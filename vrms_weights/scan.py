"""Scan cached independent GPT/DeepSeek scores without training or API calls.

Outer-label maxima are explicitly exploratory. Cached GPT batch validation is
evaluated separately, using its own batch test scores, never mixed with the
independent-request scores.
"""
import argparse
import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, confusion_matrix, roc_auc_score


CLOUD = Path("outputs/vrms_cloud/20261008_seed2026")
DEEPSEEK = Path("outputs/vrms_deepseek/20261008_seed2026")
DEFAULT_OUT = Path("outputs/vrms_weights/20261008_seed2026")
GRID = np.arange(101, dtype=float) / 100
HALF = Fraction(1, 2)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local_path(value):
    value = str(value).replace("\\", "/")
    if sys.platform != "win32" and len(value) > 2 and value[1:3] == ":/":
        value = "/mnt/" + value[0].lower() + "/" + value[3:]
    return Path(value)


def metrics(y, p, decisions=None):
    y, p = np.asarray(y, int), np.asarray(p, float)
    if not len(y) or y.shape != p.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Complete finite probabilities in [0,1] are required")
    decisions = p >= .5 if decisions is None else np.asarray(decisions, bool)
    if decisions.shape != y.shape:
        raise ValueError("Decision shape does not match labels")
    both_classes = len(np.unique(y)) == 2
    correct = int(np.count_nonzero(decisions == y))
    return dict(paths=len(y), correct_paths=correct, acc=correct / len(y),
                bacc=float(balanced_accuracy_score(y, decisions)) if both_classes else None,
                auroc=float(roc_auc_score(y, p)) if both_classes else None,
                brier=float(brier_score_loss(y, p)),
                confusion=confusion_matrix(y, decisions, labels=[0, 1]).tolist())


def exact_decisions(numeric, cloud, weight):
    """Equality is high; Fraction avoids losing isolated boundary maxima."""
    return np.asarray([a + weight * (b - a) >= HALF for a, b in zip(numeric, cloud)], bool)


def exact_cells(y, numeric, cloud):
    """Evaluate every crossing and every open interval, including 0 and 1.

    Inputs are exact fractions of the stored decimal probabilities. Threshold
    equality can make a crossing differ from both neighbouring intervals.
    """
    if len(y) != len(numeric) or len(y) != len(cloud) or not len(y):
        raise ValueError("Nonempty, matching arrays are required")
    bounds = {Fraction(0), Fraction(1)}
    for a, b in zip(numeric, cloud):
        if a != b:
            crossing = (HALF - a) / (b - a)
            if 0 <= crossing <= 1:
                bounds.add(crossing)
    bounds = sorted(bounds)
    cells = []
    for i, a in enumerate(bounds):
        count = int(np.count_nonzero(exact_decisions(numeric, cloud, a) == y))
        cells.append((a, a, True, True, count))
        if i + 1 < len(bounds):
            b = bounds[i + 1]
            count = int(np.count_nonzero(exact_decisions(numeric, cloud, (a + b) / 2) == y))
            cells.append((a, b, False, False, count))
    return cells


def best_intervals(cells):
    best = max(cell[4] for cell in cells)
    intervals = []
    for a, b, left, right, correct in cells:
        if correct != best:
            continue
        if intervals and intervals[-1][1] == a and (intervals[-1][3] or left):
            previous = intervals[-1]
            intervals[-1] = (previous[0], b, previous[2], right, best)
        else:
            intervals.append((a, b, left, right, best))
    return best, intervals


def interval_json(cell):
    a, b, left, right, correct = cell
    return dict(lower=float(a), upper=float(b), lower_inclusive=left, upper_inclusive=right,
                lower_exact_fraction=str(a), upper_exact_fraction=str(b), correct_paths=correct)


def select_grid_weight(y_validation, numeric_validation, cloud_validation):
    """Validation labels only; ACC ties prefer the lowest cloud weight."""
    y = np.asarray(y_validation, int)
    numeric = np.asarray(numeric_validation, float)
    cloud = np.asarray(cloud_validation, float)
    if not len(y) or y.shape != numeric.shape or y.shape != cloud.shape:
        raise ValueError("Nonempty validation arrays must match")
    cloud = np.where(np.isfinite(cloud), cloud, numeric)
    predictions = (1 - GRID[:, None]) * numeric + GRID[:, None] * cloud >= .5
    counts = np.count_nonzero(predictions == y, axis=1)
    index = int(np.argmax(counts))
    return float(GRID[index]), int(counts[index])


def paired_descriptive(y, subjects, a, b):
    """Subject-cluster bootstrap, conditional on the already chosen weight.

    This interval does not correct for scanning the outer labels.
    """
    groups = np.unique(subjects)
    rng = np.random.default_rng(2026)
    differences = []
    change = (a >= .5).astype(int) - (b >= .5).astype(int)
    a_correct, b_correct = (a >= .5) == y, (b >= .5) == y
    for _ in range(1000):
        sample = rng.choice(groups, len(groups), replace=True)
        ix = np.concatenate([np.flatnonzero(subjects == s) for s in sample])
        differences.append(float(np.mean(a_correct[ix]) - np.mean(b_correct[ix])))
    return dict(acc_difference=float(np.mean(a_correct) - np.mean(b_correct)),
                acc_subject_bootstrap_95ci=np.quantile(differences, [.025, .975]).tolist(),
                changed_decisions=int(np.count_nonzero(change)),
                corrected=int(np.count_nonzero(a_correct & ~b_correct)),
                harmed=int(np.count_nonzero(~a_correct & b_correct)),
                bootstrap_replicates=1000, seed=2026, selection_adjusted=False,
                interpretation="descriptive only; chosen using these same outer labels")


def audit_inputs(rows):
    """Reconcile saved scores against labels, original summaries and API logs."""
    if len(rows) != 147 or len({int(r["path_index"]) for r in rows}) != 147:
        raise ValueError("Expected 147 unique paths")
    if {int(r["path_index"]) for r in rows} != set(range(147)):
        raise ValueError("Unexpected path indices")
    rows.sort(key=lambda r: int(r["path_index"]))
    y = np.asarray([int(r["label"]) for r in rows])
    subjects = np.asarray([int(r["subject_key"]) for r in rows])
    if len(np.unique(subjects)) != 24 or np.count_nonzero(y == 0) != 71 or np.count_nonzero(y == 1) != 76:
        raise ValueError("Subject or class counts changed")
    evaluation = {r["path_index"]: r for r in read_json(CLOUD / "evaluation.json")}
    test_evidence = {r["path_index"]: r["evidence"] for fold in read_json(CLOUD / "fold_evidence.json")
                     for r in fold["records"] if r["role"] == "outer_test"}
    numeric_rows = {(int(r["outer_subject"]), int(r["path_index"])): r
                    for r in read_csv(CLOUD / "improvements/predictions.csv")}
    previous = read_json(DEEPSEEK / "summary.json")
    for name, old in previous["methods"].items():
        p = np.asarray([float(r[name]) for r in rows])
        recomputed = metrics(y, p)
        for key in ("acc", "bacc", "auroc", "brier"):
            if abs(recomputed[key] - old[key]) > 1e-12:
                raise ValueError(f"Saved metric mismatch: {name}/{key}")
        if recomputed["correct_paths"] != old["correct_paths"]:
            raise ValueError(f"Saved correct count mismatch: {name}")
    checked_files = {DEEPSEEK / "path_oof.csv", DEEPSEEK / "summary.json", DEEPSEEK / "validation.json",
                     DEEPSEEK / "protocol.json", CLOUD / "path_oof.csv", CLOUD / "summary.json",
                     CLOUD / "validation.json", CLOUD / "independent_protocol.json",
                     CLOUD / "evaluation.json", CLOUD / "fold_evidence.json",
                     CLOUD / "improvements/predictions.csv"}
    protocol = read_json(DEEPSEEK / "protocol.json")
    for name, expected in protocol["source_artifact_sha256"].items():
        path = local_path(name)
        if sha(path) != expected:
            raise ValueError(f"Frozen source hash changed: {path}")
        checked_files.add(path)
    for package in ("vrms_pilot", "vrms_cloud", "vrms_deepseek"):
        checked_files.update(Path(package).glob("*.py"))
    for row in rows:
        i, s, label = int(row["path_index"]), int(row["subject_key"]), int(row["label"])
        if evaluation[i]["subject_key"] != s or evaluation[i]["label"] != label:
            raise ValueError(f"Label/subject mismatch: {i}")
        if float(row["improved_numeric"]) != float(numeric_rows[s, i]["validated_numeric"]):
            raise ValueError(f"Numerical score mismatch: {i}")
        examples = None
        for name, folder in (("gpt", CLOUD), ("deepseek", DEEPSEEK)):
            path = folder / "independent_calls" / f"path_{i:03d}_cloud_call.json"
            checked_files.add(path)
            log = read_json(path)
            if not log["success"]:
                raise ValueError(f"A cloud result is missing: {name}/{i}")
            if name == "gpt":
                if log["model"] != "gpt-6.1-sol" or len(log["predictions"]) != 1 or len(log["mapping"]) != 1:
                    raise ValueError(f"Unexpected GPT context: {i}")
                if log["mapping"] != {"qsingle": {"role": "single_case"}} or log["mode"] != "single_evidence_assessment":
                    raise ValueError(f"GPT anonymous single-case context changed: {i}")
                prompt = json.loads(log["user_message"])
                probability = log["predictions"][0]["high_probability"]
            else:
                if log["model"] != "deepseek-flash" or log["response_model"] != "deepseek-flash":
                    raise ValueError(f"Unexpected DeepSeek model: {i}")
                probability = log["prediction"]["high_probability"]
                prompt = json.loads(log["request"]["messages"][1]["content"])
            if prompt["queries"] != [dict(id="qsingle", evidence=test_evidence[i])]:
                raise ValueError(f"Single-path evidence changed: {name}/{i}")
            if examples is None:
                examples = prompt["examples"]
            elif examples != prompt["examples"]:
                raise ValueError(f"GPT/DeepSeek examples differ: {i}")
            if float(row[name + "_cloud"]) != probability:
                raise ValueError(f"Cloud probability mismatch: {name}/{i}")
            fusion = .75 * float(row["improved_numeric"]) + .25 * probability
            if abs(float(row[name + "_fusion25"]) - fusion) > 1e-15:
                raise ValueError(f"25% score mismatch: {name}/{i}")
    return y, subjects, evaluation, numeric_rows, checked_files


def gpt_batch_internal(evaluation, numeric_rows):
    """Same-context inner selection diagnostic, distinct from the main scan."""
    y = np.asarray([evaluation[i]["label"] for i in range(147)])
    selected_oof, fixed25_oof = np.full(147, np.nan), np.full(147, np.nan)
    selected, files = [], []
    folds = read_json(CLOUD / "fold_evidence.json")
    for fold in folds:
        s, split = fold["outer_subject"], fold["split"]
        groups = [set(split[role]) for role in ("base_train", "meta_calibration", "policy_validation", "outer_test")]
        if len(set.union(*groups)) != 24 or sum(map(len, groups)) != 24 or split["outer_test"] != [s]:
            raise ValueError(f"Subject partitions are not disjoint: {s}")
        path = CLOUD / "cloud_calls" / f"improved_few_shot_fold_{s:02d}.json"
        files.append(path)
        log = read_json(path)
        if log["outer_subject"] != s or log["mode"] != "improved_few_shot" or log["model"] != "gpt-6.1-sol":
            raise ValueError("Unexpected batch context")
        if any(evaluation[i]["subject_key"] not in groups[1] for i in log["example_indices"]):
            raise ValueError("Batch example is not fold-internal meta data")
        cloud = {}
        if log["success"]:
            for prediction in log["predictions"]:
                mapping = log["mapping"][prediction["id"]]
                cloud[mapping["path_index"]] = prediction["high_probability"]
        val = [r["path_index"] for r in fold["records"] if r["role"] == "policy_validation"]
        test = [r["path_index"] for r in fold["records"] if r["role"] == "outer_test"]
        if not val or not test or set(val) & set(test):
            raise ValueError("Batch validation/test path split is invalid")
        for role, indices, allowed in (("policy_validation", val, groups[2]), ("outer_test", test, groups[3])):
            for i in indices:
                if evaluation[i]["subject_key"] not in allowed or numeric_rows[s, i]["role"] != role:
                    raise ValueError("Wrong fold/role numerical prediction supplied to selection")
                if not any(m["path_index"] == i and m["role"] == role for m in log["mapping"].values()):
                    raise ValueError("Wrong fold/role batch prediction supplied to selection")
        numeric_val = np.asarray([float(numeric_rows[s, i]["validated_numeric"]) for i in val])
        cloud_val = np.asarray([cloud.get(i, np.nan) for i in val])
        weight, correct = select_grid_weight(y[val], numeric_val, cloud_val)
        selected.append(dict(outer_subject=s, validation_paths=len(val), weight=weight,
                             validation_correct=correct, validation_acc=correct / len(val)))
        # Only after selecting a weight are this fold's test scores combined.
        for i in test:
            numeric = float(numeric_rows[s, i]["validated_numeric"])
            probability = cloud.get(i, numeric)
            selected_oof[i] = (1 - weight) * numeric + weight * probability
            fixed25_oof[i] = .75 * numeric + .25 * probability
    previous = read_json(CLOUD / "summary.json")["methods"]["improved_llm25"]
    fixed25 = metrics(y, fixed25_oof)
    if abs(fixed25["acc"] - previous["acc"]) > 1e-12:
        raise ValueError("Batch diagnostic did not reproduce the previous fixed-weight ACC")
    return dict(context="GPT per-fold batch; not independent requests", objective="validation path ACC",
                grid_step=.01, ties="smallest LLM weight", selected=selected,
                validation_selected_outer=metrics(y, selected_oof), fixed25_outer=fixed25,
                independent_context_selection_available=False,
                deepseek_independent_policy_validation_available=False), files


def make_plot(out, curves, cells_by_vendor, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    colors = {"gpt": "#0072B2", "deepseek": "#B44F00"}
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
                         "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), layout="constrained", sharey=True)
        for ax, vendor in zip(axes, ("gpt", "deepseek")):
            result, color = summary["vendors"][vendor], colors[vendor]
            cells = cells_by_vendor[vendor]
            for a, b, _, _, correct in cells:
                if a != b:
                    ax.hlines(100 * correct / 147, float(a) * 100, float(b) * 100, color=color, linewidth=1.8)
            points = [cell for cell in cells if cell[0] == cell[1]]
            ax.scatter([float(cell[0]) * 100 for cell in points], [cell[4] * 100 / 147 for cell in points],
                       color=color, s=7, zorder=3)
            for interval in result["continuous_best_intervals"]:
                ax.axvspan(interval["lower"] * 100, interval["upper"] * 100, color=color, alpha=.10)
            ax.axhline(summary["numeric"]["acc"] * 100, color="#333333", linestyle=":", linewidth=1.1,
                       label=f"Numerical only: {summary['numeric']['acc']:.2%}")
            selected = result["representative_grid_weight"]
            ax.scatter([selected * 100], [result["representative_metrics"]["acc"] * 100],
                       s=70, marker="D", color=color, edgecolor="white", zorder=5,
                       label=f"Smallest best 1% grid weight: {selected:.0%}")
            ax.set_title(f"{result['model']}\nMaximum ACC: {result['continuous_best_acc']:.2%} ({result['continuous_best_correct']}/147)")
            ax.set(xlabel="LLM weight (%)", xlim=(0, 100), ylim=(50, 75), xticks=np.arange(0, 101, 20))
            ax.grid(axis="y", color="#DDDDDD", linewidth=.6)
            ax.legend(loc="lower left", fontsize=8.5, frameon=False)
        axes[0].set_ylabel("Whole-path ACC (%)")
        fig.suptitle("Cached independent requests: exploratory weight sweep", fontsize=14)
        fig.supxlabel("147 paths / 24 subjects; fixed threshold 0.5. Maxima use already examined outer labels.", fontsize=9)
        for extension in ("png", "pdf", "svg"):
            fig.savefig(out / f"weight_acc.{extension}", dpi=240, facecolor="white", transparent=False)
        plt.close(fig)
    with Image.open(out / "weight_acc.png") as img:
        metadata = dict(pixels=list(img.size), dpi=list(img.info["dpi"]), mode=img.mode)
    manifest = dict(raw_source=str(DEEPSEEK / "path_oof.csv"), raw_source_sha256=sha(DEEPSEEK / "path_oof.csv"),
                    medium="local exploratory research report", journal_specification=None,
                    transformations=["linear probability fusion", "threshold >=0.5", "exact crossing-point enumeration"],
                    exclusions=0, missing_probabilities=0, smoothing=False,
                    curve_uncertainty="not plotted; descriptive counts of the same fixed 147 paths",
                    alt_text="GPT peaks at 98/147 and DeepSeek at 95/147; numerical baseline is 94/147. Higher LLM weights generally perform worse.",
                    png_metadata=metadata, matplotlib_version=matplotlib.__version__,
                    exports={p.name: dict(sha256=sha(p), bytes=p.stat().st_size) for p in out.glob("weight_acc.*")})
    write_json(out / "figure_manifest.json", manifest)


def pct(value):
    return f"{value * 100:.2f}%"


def interval_text(interval):
    left = "[" if interval["lower_inclusive"] else "("
    right = "]" if interval["upper_inclusive"] else ")"
    return f"{left}{interval['lower'] * 100:.4f}%, {interval['upper'] * 100:.4f}%{right}"


def report_text(summary):
    lines = ["# GPT / DeepSeek 融合权重扫描", "", "本次复用两种云端模型各 147 次独立请求的已保存概率；新增 API 调用为 0，没有重新训练模型。", "",
             "固定同一数值模型、24 个被试、147 条路径（低类 71、高类 76），保持 0.5 判决阈值。目标是整条路径问卷分数 ≥30 的高/低分类 ACC。", "",
             "融合公式：`p_final = (1 - w) × p_numeric + w × p_LLM`；`p_final >= 0.5` 判为高类。数值端包含此前内部选出的 EEG 数值模型及证据融合结果。", "",
             "**这里的最大值使用了这 147 条外层测试路径的真实标签选择权重，是事后探索结果，不能当作独立验证后的部署最优权重。**", "",
             "## 1% 步长扫描结果", "", "| 方法 | AI 权重 | 数值权重 | 判对 / 147 | ACC | 比纯数值多判对 |", "|---|---:|---:|---:|---:|---:|",
             f"| 纯数值 | 0% | 100% | {summary['numeric']['correct_paths']} | {pct(summary['numeric']['acc'])} | — |"]
    for vendor in ("gpt", "deepseek"):
        result = summary["vendors"][vendor]
        m, w = result["representative_metrics"], result["representative_grid_weight"]
        lines.append(f"| {result['model']} 最优整数权重代表 | {w:.0%} | {1-w:.0%} | {m['correct_paths']} | {pct(m['acc'])} | {m['correct_paths'] - summary['numeric']['correct_paths']} |")
        for k in (25, 100):
            m = result["checkpoints"][str(k)]
            lines.append(f"| {result['model']} | {k}% | {100-k}% | {m['correct_paths']} | {pct(m['acc'])} | {m['correct_paths'] - summary['numeric']['correct_paths']} |")
    lines += ["", "代表权重的规则是：在 0%–100%、每次增加 1% 的网格上，以 ACC 最大为目标，并列时列出全部结果，表中取最小的整数权重。这个规则没有进一步按 BACC 或 AUROC 挑选。", ""]
    for vendor in ("gpt", "deepseek"):
        r = summary["vendors"][vendor]
        grid = "、".join(f"{w * 100:.0f}%" for w in r["grid_best_weights"])
        lines += [f"- {r['model']}：最优整数权重为 **{grid}**，均为 **{r['grid_best_correct']}/147 = {pct(r['grid_best_acc'])}**。"]
    lines += ["", "## 连续权重的全部并列最优区间", "", "除了 1% 网格，还枚举了每条路径跨越阈值的权重及相邻开区间，检查是否遗漏狭窄峰值。`[`/`]` 表示包含边界，`(`/`)` 表示不包含。下面边界显示到小数点后四位；精确分数保存在 JSON。", ""]
    for vendor in ("gpt", "deepseek"):
        r = summary["vendors"][vendor]
        intervals = "；".join(interval_text(i) for i in r["continuous_best_intervals"])
        lines += [f"- **{r['model']}**：{intervals}，最高仍为 **{r['continuous_best_correct']}/147 = {pct(r['continuous_best_acc'])}**。"]
    lines += ["", "GPT 还存在约 24.36%–24.68% 的狭窄并列最优区间，1% 网格会漏掉，但它没有超过 66.67%。GPT 从原来的 25% 调到整数最优点，只多判对 1 条。DeepSeek 的两个最优区间并列；不能据此断言 3% 或 30% 是稳定可靠的最优值。", "",
              "## 收益与局限", "", "| 方法（代表权重） | ACC 增量 | 纠正 / 误改 | 被试聚类 bootstrap 95% 区间 |", "|---|---:|---:|---:|"]
    for vendor in ("gpt", "deepseek"):
        r = summary["vendors"][vendor]
        c = r["comparison_to_numeric"]
        lo, hi = c["acc_subject_bootstrap_95ci"]
        lines.append(f"| {r['model']}（{r['representative_grid_weight']:.0%}） | {c['acc_difference'] * 100:+.2f} 个百分点 | {c['corrected']} / {c['harmed']} | [{lo * 100:+.2f}, {hi * 100:+.2f}] 个百分点 |")
    lines += ["", "区间按被试整体重采样 1000 次，seed=2026。它们只描述选定权重下的样本波动，**没有校正搜索最优权重产生的乐观偏差，不能作为扫描后显著性结论**。单次 API 输出和单个随机种子也不代表重复运行后的平均性能。", "",
              "## 内部验证能否选出权重", "", "当前两种模型的独立请求模式均只有外层测试预测，没有各外层折对应的独立请求政策验证预测。不能拿其他折的 OOF 结果代替：其底层模型可能训练过当前测试被试。", "",
              "已有 GPT 政策验证预测来自旧的整折批处理。本次单独使用这些批处理验证概率，在各折 4 个政策验证被试上按 ACC 选择 0%–100% 的整数权重，再应用到该折的批处理测试概率；不与独立请求结果混用。", ""]
    d = summary["gpt_batch_internal_diagnostic"]
    m = d["validation_selected_outer"]
    old = d["fixed25_outer"]
    lines += [f"- GPT 批处理：内部选择整数权重的外层 ACC 为 **{m['correct_paths']}/147 = {pct(m['acc'])}**；同批处理固定 25% 为 **{old['correct_paths']}/147 = {pct(old['acc'])}**。",
              "- 这只是既有批处理模式的附加诊断，不能证明独立请求的最优权重，也没有对应的 DeepSeek 内部验证比较。", "",
              "## 下一步建议", "", "1. GPT 40% 和 DeepSeek 3% 可作为下一轮预先固定的候选，原来的 25% 和纯数值 0% 也保留作比较。最终部署权重应在各外层折的政策验证被试上选择，并在独立请求模式下做完整外层评估。", "",
              "2. 从本批样本看，GPT 最多增加 4 条正确路径，DeepSeek 最多增加 1 条，收益有限。优先检查边界样本和证据冲突：只在数值概率接近阈值或多路证据冲突时请求云端，门控阈值和融合权重一并在内部验证上选择，云端不确定或失败时保留数值结果。", "",
              "3. 再考虑用内部校准数据学习简单、正则化的融合器，结合数值概率、云端概率、QC 和证据冲突指标；不能把未经校准的 LLM 自报信心直接当作可靠权重。新融合器需与固定权重及纯数值基线公平比较。", "",
              "## 文件与复现", "", "- `weight_grid.csv`：每个模型 0%–100% 的 ACC、BACC、AUROC、Brier 与混淆矩阵。", "- `exact_cells.csv`：所有阈值点和区间的正确数，包含边界是否纳入。", "- `summary.json`：全部最优区间、整数权重、指标和描述性比较。", "- `representative_predictions.csv`：最小整数最优权重下的逐路径预测及纠正/误改。", "- `gpt_batch_inner_selection.csv`：附加诊断的每折验证权重。", "- `weight_acc.png/.pdf/.svg`：准确率曲线；底层数据没有平滑或插值。", "- `validation.json`：模型身份、294 条已保存调用、标签和概率核对及原始文件哈希。", "", "从项目目录执行（只读缓存，无 API 调用；目标目录必须尚不存在）：", "", "```powershell", "python -m vrms_weights.scan --out outputs/vrms_weights/recheck", "```", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Use a new output folder to preserve completed analyses")
    rows = read_csv(DEEPSEEK / "path_oof.csv")
    y, subjects, evaluation, numeric_rows, checked_files = audit_inputs(rows)
    inner, batch_files = gpt_batch_internal(evaluation, numeric_rows)
    checked_files.update(batch_files)
    before = {str(p): sha(p) for p in sorted(checked_files, key=str)}
    numeric = np.asarray([float(r["improved_numeric"]) for r in rows])
    numeric_exact = [Fraction(r["improved_numeric"]) for r in rows]
    summary = dict(status="completed_exploratory_posthoc_weight_scan", paths=147, subjects=24,
                   objective="path ACC", fusion="(1-w)*improved_numeric + w*llm", threshold=.5,
                   cloud_models=dict(gpt="gpt-6.1-sol", deepseek="deepseek-flash"),
                   added_api_calls=0, models_retrained=False, source_results_rewritten=False,
                   weights_selected_using_outer_labels=True, deployment_weight_changed=False,
                   numeric=metrics(y, numeric), vendors={}, gpt_batch_internal_diagnostic=inner,
                   seed=2026, grid_step=.01, independent_paths_per_vendor=147,
                   created_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version())
    curves, cells_by_vendor, cell_rows, predictions = [], {}, [], [dict(r) for r in rows]
    for vendor, model in (("gpt", "gpt-6.1-sol"), ("deepseek", "deepseek-flash")):
        cloud = np.asarray([float(r[vendor + "_cloud"]) for r in rows])
        cloud_exact = [Fraction(r[vendor + "_cloud"]) for r in rows]
        cells = exact_cells(y, numeric_exact, cloud_exact)
        cells_by_vendor[vendor] = cells
        continuous_correct, intervals = best_intervals(cells)
        vendor_grid = []
        for k, weight in enumerate(GRID):
            p = (1 - weight) * numeric + weight * cloud
            decisions = exact_decisions(numeric_exact, cloud_exact, Fraction(k, 100))
            if not np.array_equal(decisions, p >= .5):
                raise ValueError(f"Floating point/grid boundary disagreement: {vendor}/{k}")
            result = metrics(y, p, decisions)
            vendor_grid.append(result)
            curves.append(dict(vendor=vendor, model=model, llm_weight=weight, numeric_weight=1-weight,
                               **{key: value for key, value in result.items() if key != "confusion"},
                               tn=result["confusion"][0][0], fp=result["confusion"][0][1],
                               fn=result["confusion"][1][0], tp=result["confusion"][1][1]))
        grid_correct = max(m["correct_paths"] for m in vendor_grid)
        best_k = [k for k, m in enumerate(vendor_grid) if m["correct_paths"] == grid_correct]
        w = best_k[0] / 100
        representative = (1 - w) * numeric + w * cloud
        summary["vendors"][vendor] = dict(model=model, grid_best_correct=grid_correct,
                   grid_best_acc=grid_correct / len(y), grid_best_weights=[k / 100 for k in best_k],
                   continuous_best_correct=continuous_correct, continuous_best_acc=continuous_correct / len(y),
                   continuous_best_intervals=[interval_json(i) for i in intervals],
                   representative_grid_weight=w, representative_tie_rule="smallest best 1% grid weight",
                   representative_metrics=vendor_grid[best_k[0]],
                   comparison_to_numeric=paired_descriptive(y, subjects, representative, numeric),
                   checkpoints={str(k): vendor_grid[k] for k in (0, 5, 10, 20, 25, 30, 40, 50, 60, 75, 100)})
        for cell in cells:
            cell_rows.append(dict(vendor=vendor, kind="point" if cell[0] == cell[1] else "open_interval",
                                  **interval_json(cell), acc=cell[4] / len(y)))
        for i, row in enumerate(predictions):
            baseline_ok, chosen_ok = (numeric[i] >= .5) == y[i], (representative[i] >= .5) == y[i]
            change = "corrected" if chosen_ok and not baseline_ok else "harmed" if baseline_ok and not chosen_ok else "unchanged_correctness"
            row.update({vendor + "_best_grid_weight": w, vendor + "_best_grid_probability": float(representative[i]),
                        vendor + "_best_grid_state": "high" if representative[i] >= .5 else "low",
                        vendor + "_best_grid_correct": int(chosen_ok), vendor + "_change_vs_numeric": change})
    after = {name: sha(name) for name in before}
    if after != before:
        raise ValueError("An original input changed during the analysis")
    args.out.mkdir(parents=True)
    (args.out / "source_predictions.csv").write_bytes((DEEPSEEK / "path_oof.csv").read_bytes())
    write_csv(args.out / "weight_grid.csv", curves)
    write_csv(args.out / "exact_cells.csv", cell_rows)
    write_csv(args.out / "representative_predictions.csv", predictions)
    write_csv(args.out / "gpt_batch_inner_selection.csv", inner["selected"])
    write_json(args.out / "summary.json", summary)
    (args.out / "权重扫描报告.md").write_text(report_text(summary), encoding="utf-8")
    make_plot(args.out, curves, cells_by_vendor, summary)
    if {name: sha(name) for name in before} != before:
        raise ValueError("An original input changed while generating outputs")
    write_json(args.out / "validation.json", dict(status="passed", original_input_hashes_unchanged=True,
               checked_original_files=len(before), original_sha256=before, reconciled_independent_requests=294,
               saved_metrics_reproduced=True, complete_finite_paths_per_vendor=147, unique_subjects=24,
               exact_crossings_and_threshold_equality_checked=True, integer_grid_float_exact_agreement=True,
               inner_diagnostic_fold_specific_predictions_checked=True, inner_diagnostic_subject_disjointness_checked=True,
               batch_and_independent_scores_not_mixed=True, api_calls_added=0, deployment_weight_changed=False,
               implementation_sha256={p.name: sha(p) for p in Path("vrms_weights").glob("*.py")}))
    print(json.dumps(dict(status=summary["status"], output=str(args.out),
               vendors={name: {key: r[key] for key in ("grid_best_weights", "continuous_best_acc", "representative_grid_weight", "comparison_to_numeric")}
                        for name, r in summary["vendors"].items()},
               gpt_batch_validation_selected=inner["validation_selected_outer"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
