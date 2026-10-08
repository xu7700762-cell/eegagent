"""Select fusion/calibration on policy subjects; score outer subjects once."""
import argparse
import csv
from collections import Counter
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from vrms_weights.scan import metrics
from .common import CLOUD, DEEPSEEK, DEFAULT_OUT, api_job, job_hash, read_json, request_path, sha, write_json


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def subset(data, indices):
    return {k: np.asarray(v)[indices] for k, v in data.items()}


def feature_matrix(data, name):
    def logit(p):
        p = np.clip(p, 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p))
    if name == "baseline25_only":
        return logit(data["baseline25"])[:, None]
    two = np.c_[logit(data["numeric"]), logit(data["enhanced"])]
    if name == "two_probabilities":
        return two
    if name == "probabilities_quality_conflict":
        return np.c_[two, np.abs(data["numeric"] - data["enhanced"]), data["qc"], data["reference"], data["spread"]]
    raise ValueError("Unknown, non-predeclared fusion features")


def fit_predict(spec, train, y_train, target, weights):
    """No target labels are accepted by this interface."""
    if spec["kind"] == "fixed":
        return target[spec["source"]].copy(), dict(spec=spec)
    if spec["kind"] == "blend":
        cloud = train[spec["source"]]
        counts = [np.count_nonzero(((1-w)*train["numeric"] + w*cloud >= .5) == y_train) for w in weights]
        w = float(weights[int(np.argmax(counts))])
        return (1-w)*target["numeric"] + w*target[spec["source"]], dict(spec=spec, weight=w)
    if spec["kind"] != "logistic":
        raise ValueError("Unknown candidate")
    if len(np.unique(y_train)) < 2:
        return target["numeric"].copy(), dict(spec=spec, numeric_fallback=True)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=spec["C"], solver="liblinear",
                          max_iter=2000, class_weight=None, random_state=2026))
    model.fit(feature_matrix(train, spec["features"]), y_train)
    return model.predict_proba(feature_matrix(target, spec["features"]))[:, 1], dict(spec=spec, estimator=model)


def cv_predict(spec, data, y, groups, weights):
    p = np.full(len(y), np.nan)
    for subject in np.unique(groups):
        training = np.flatnonzero(groups != subject)
        target = np.flatnonzero(groups == subject)
        if set(groups[training]) & set(groups[target]):
            raise ValueError("Fusion CV subject leakage")
        p[target], _ = fit_predict(spec, subset(data, training), y[training], subset(data, target), weights)
    if not np.isfinite(p).all():
        raise ValueError("Fusion CV did not score every validation path")
    return p


def predict_saved(bundle, data):
    spec = bundle["spec"]
    if spec["kind"] == "fixed":
        return data[spec["source"]].copy()
    if spec["kind"] == "blend":
        w = bundle["weight"]
        return (1-w)*data["numeric"] + w*data[spec["source"]]
    if bundle.get("numeric_fallback"):
        return data["numeric"].copy()
    return bundle["estimator"].predict_proba(feature_matrix(data, spec["features"]))[:,1]


def choose_accuracy(specs, data, y, groups, weights):
    probabilities, diagnostics = [], []
    for spec in specs:
        p = cv_predict(spec, data, y, groups, weights)
        m = metrics(y, p)
        probabilities.append(p)
        diagnostics.append(dict(name=spec["name"], **m))
    chosen = min(range(len(specs)), key=lambda k: (-diagnostics[k]["correct_paths"], diagnostics[k]["brier"], k))
    return specs[chosen], probabilities[chosen], diagnostics


def choose_calibration(specs, data, y, groups, weights):
    probabilities = [cv_predict(spec, data, y, groups, weights) for spec in specs]
    diagnostics = [dict(name=spec["name"], **metrics(y, p)) for spec, p in zip(specs, probabilities)]
    eligible = [k for k, d in enumerate(diagnostics) if d["correct_paths"] >= diagnostics[0]["correct_paths"] - 1]
    chosen = min(eligible, key=lambda k: (diagnostics[k]["brier"], k))
    return specs[chosen], probabilities[chosen], diagnostics


def selective_threshold(y, p, plan):
    confidence = np.maximum(p, 1-p)
    for threshold in plan["selective_confidence_thresholds"]:
        accept = confidence >= threshold
        if (accept.sum() >= plan["selective_minimum_validation_cases"] and
            accept.mean() >= plan["selective_minimum_validation_coverage"] and
            np.mean((p[accept] >= .5) == y[accept]) >= plan["selective_target_validation_accuracy"]):
            return float(threshold)
    return None


def scored(y, p):
    result = metrics(y, p)
    bins = np.minimum((p * 10).astype(int), 9)
    ece = 0.
    for b in range(10):
        selected = bins == b
        if selected.any():
            ece += selected.mean() * abs(float(y[selected].mean()) - float(p[selected].mean()))
    result.update(log_loss=float(log_loss(y, np.clip(p, 1e-8, 1-1e-8), labels=[0, 1])), ece_probability_10bins=float(ece))
    return result


def paired(y, subjects, candidate, baseline):
    groups = np.unique(subjects)
    sizes = np.asarray([np.count_nonzero(subjects == s) for s in groups])
    acc_gain = np.asarray([np.count_nonzero((candidate[subjects == s] >= .5) == y[subjects == s]) -
                           np.count_nonzero((baseline[subjects == s] >= .5) == y[subjects == s]) for s in groups])
    brier_gain = np.asarray([np.sum((baseline[subjects == s] - y[subjects == s]) ** 2 -
                                   (candidate[subjects == s] - y[subjects == s]) ** 2) for s in groups])
    draws = np.random.default_rng(2026).integers(0, len(groups), size=(2000, len(groups)))
    totals = sizes[draws].sum(axis=1)
    corrected = (candidate >= .5) == y
    original = (baseline >= .5) == y
    return dict(acc_gain=float(acc_gain.sum() / len(y)), brier_gain=float(brier_gain.sum() / len(y)),
                acc_gain_subject_bootstrap_95ci=np.quantile(acc_gain[draws].sum(axis=1) / totals, [.025, .975]).tolist(),
                brier_gain_subject_bootstrap_95ci=np.quantile(brier_gain[draws].sum(axis=1) / totals, [.025, .975]).tolist(),
                corrected=int(np.count_nonzero(corrected & ~original)), harmed=int(np.count_nonzero(~corrected & original)),
                subjects_net_improved=int(np.count_nonzero(acc_gain > 0)),
                subjects_net_worsened=int(np.count_nonzero(acc_gain < 0)),
                bootstrap_replicates=2000, seed=2026, already_examined_outer_data=True)


def acceptance(summary, protocol):
    gates = protocol["acceptance"]
    methods, comparisons = summary["methods"], summary["comparisons"]
    base, main, cal = methods["baseline25"], methods["selected_accuracy"], methods["calibrated_baseline25"]
    a, c = comparisons["selected_minus_baseline25"], comparisons["calibrated_minus_baseline25"]
    api_ok = summary["api"]["successful_logical_requests"] / summary["api"]["logical_requests"] >= gates["api_minimum_success_rate"]
    accuracy = dict(gain_at_least_five_paths=main["correct_paths"]-base["correct_paths"] >= gates["accuracy_minimum_gain_paths"],
                    positive_subject_ci=a["acc_gain_subject_bootstrap_95ci"][0] > 0,
                    bacc_not_worse=main["bacc"] >= base["bacc"],
                    brier_guard=main["brier"] <= base["brier"]+gates["brier_max_increase"], api_coverage=api_ok)
    calibration = dict(relative_brier_gain=c["brier_gain"] / base["brier"] >= gates["calibration_minimum_relative_brier_improvement"],
                       positive_subject_ci=c["brier_gain_subject_bootstrap_95ci"][0] > 0,
                       lost_at_most_one_path=cal["correct_paths"] >= base["correct_paths"]-gates["calibration_max_lost_correct_paths"],
                       api_coverage=api_ok)
    return dict(accuracy_promoted=all(accuracy.values()), calibration_promoted=all(calibration.values()),
                accuracy_checks=accuracy, calibration_checks=calibration)


def audit_calls(out, vendor, jobs, status):
    rows = {(r["mode"], r["outer_subject"], r["path_index"]): r for r in status["results"]}
    if status["status"] != "completed" or status["completed_requests"] != 1309 or len(rows) != 1309:
        raise ValueError("The full 1309-request run is not complete")
    attempts, returned_models, token_usage, paths = [], Counter(), Counter(), []
    for job in jobs:
        key = job["mode"], job["outer_subject"], job["path_index"]
        row = rows[key]
        original_path = request_path(out, vendor, job)
        path = Path(row.get("selected_log_path", str(original_path))).as_posix().replace("\\", "/")
        if sys.platform != "win32" and len(path) > 2 and path[1:3] == ":/":
            path = "/mnt/" + path[0].lower() + "/" + path[3:]
        path = Path(path)
        log = read_json(path)
        if sha(path) != row["log_sha256"] or log["success"] != row["success"]:
            raise ValueError("Cloud log/status mismatch")
        if vendor == "gpt":
            if log["model"] != "gpt-6.1-sol" or log["request_sha256"] != job["message_sha256"] or log["user_message"] != job["message"]:
                raise ValueError("GPT request changed")
            if log["mapping"] != api_job(job)["mapping"]:
                raise ValueError("GPT request is not single-path")
            prediction = log["predictions"][0] if log["success"] else None
        else:
            message = log["request"]["messages"][1]["content"]
            if log["model"] != "deepseek-flash" or message != job["message"]:
                raise ValueError("DeepSeek request changed")
            prediction = log["prediction"] if log["success"] else None
            if log["success"] and log["response_model"] != "deepseek-flash":
                raise ValueError("DeepSeek returned another model")
        if prediction is not None and prediction["high_probability"] != row["probability"]:
            raise ValueError("Cloud probability changed")
        logical_paths = [original_path]
        if path != original_path:
            logical_paths.append(path)
        for log_path in logical_paths:
            if not log_path.exists():
                continue
            paths.append(log_path)
            for attempt in read_json(log_path)["attempts"]:
                attempts.append(attempt)
                response = attempt.get("response", {})
                if response:
                    returned_models[response.get("model", "unknown")] += 1
                usage = response.get("usage", {})
                token_usage["input"] += usage.get("input_tokens", usage.get("prompt_tokens", 0))
                token_usage["output"] += usage.get("output_tokens", usage.get("completion_tokens", 0))
    return rows, dict(logical_requests=1309, successful_logical_requests=sum(r["success"] for r in rows.values()),
                      attempted_requests=len(attempts), failed_attempts=sum(a.get("status") != "success" for a in attempts),
                      returned_models=dict(returned_models), tokens=dict(token_usage),
                      mean_seconds=float(np.mean([r["seconds"] for r in rows.values()])),
                      monetary_cost_verified=False), paths


def evaluate_vendor(out, vendor, protocol, plan, jobs):
    status = read_json(out / vendor / "run_status.json")
    cloud_rows, api, log_paths = audit_calls(out, vendor, jobs, status)
    evaluation = {r["path_index"]: r for r in read_json(CLOUD / "evaluation.json")}
    numerical = {(int(r["outer_subject"]), int(r["path_index"])): r for r in read_csv(CLOUD / "improvements/predictions.csv")}
    original_rows = sorted(read_csv(DEEPSEEK / "path_oof.csv"), key=lambda r: int(r["path_index"]))
    y = np.asarray([int(r["label"]) for r in original_rows])
    subjects = np.asarray([int(r["subject_key"]) for r in original_rows])
    names = ("numeric", "baseline_cloud", "enhanced_cloud", "baseline25", "enhanced25", "baseline_selected_blend",
             "enhanced_selected_blend", "selected_accuracy", "calibrated_baseline25")
    methods = {name: np.full(147, np.nan) for name in names}
    published = np.zeros(147, bool)
    thresholds, diagnostics, models = {}, [], {}
    for fold in read_json(CLOUD / "fold_evidence.json"):
        s = fold["outer_subject"]
        val = [r for r in fold["records"] if r["role"] == "policy_validation"]
        test = [r for r in fold["records"] if r["role"] == "outer_test"]
        records = val + test
        data = {k: [] for k in ("numeric", "baseline", "enhanced", "qc", "reference", "spread")}
        for record in records:
            i = record["path_index"]
            n = float(numerical[s, i]["validated_numeric"])
            b = (cloud_rows["baseline", s, i]["probability"] if record["role"] == "policy_validation"
                 else float(original_rows[i][vendor + "_cloud"]))
            e = cloud_rows["enhanced", s, i]["probability"]
            probability = record["evidence"]
            data["numeric"].append(n)
            data["baseline"].append(n if b is None else b)
            data["enhanced"].append(n if e is None else e)
            data["qc"].append(probability["qc_accepted_fraction"])
            data["reference"].append(float(probability["reference_available"]))
            data["spread"].append(float(np.std([v for k, v in probability["improved_probabilities"].items()
                                                if k not in ("validated_numeric_probability", "candidate_mean_probability") and v is not None])))
        data = {k: np.asarray(v) for k, v in data.items()}
        data["baseline25"] = .75 * data["numeric"] + .25 * data["baseline"]
        data["enhanced25"] = .75 * data["numeric"] + .25 * data["enhanced"]
        vdata, tdata = subset(data, np.arange(len(val))), subset(data, np.arange(len(val), len(records)))
        vy = np.asarray([evaluation[r["path_index"]]["label"] for r in val])
        vg = np.asarray([evaluation[r["path_index"]]["subject_key"] for r in val])
        if set(vg) & set(subjects[[r["path_index"] for r in test]]):
            raise ValueError("Outer subject entered fusion training")
        spec, cv, cv_diagnostics = choose_accuracy(plan["primary_candidates"], vdata, vy, vg, plan["blend_weight_grid"])
        main, bundle = fit_predict(spec, vdata, vy, tdata, plan["blend_weight_grid"])
        cal_spec, cal_cv, cal_diagnostics = choose_calibration(plan["secondary_calibration_candidates"], vdata, vy, vg, plan["blend_weight_grid"])
        calibrated, cal_bundle = fit_predict(cal_spec, vdata, vy, tdata, plan["blend_weight_grid"])
        # Reload the fitted policy without fitting or consulting target labels.
        temp_path = out / vendor / f"policy_reload_{s:02d}.joblib"
        joblib.dump(dict(accuracy=bundle, calibration=cal_bundle), temp_path)
        reloaded = joblib.load(temp_path)
        np.testing.assert_array_equal(main,predict_saved(reloaded["accuracy"],tdata))
        np.testing.assert_array_equal(calibrated,predict_saved(reloaded["calibration"],tdata))
        threshold = selective_threshold(vy, cv, plan)
        thresholds[s] = threshold
        local = dict(numeric=tdata["numeric"], baseline_cloud=tdata["baseline"], enhanced_cloud=tdata["enhanced"],
                     baseline25=tdata["baseline25"], enhanced25=tdata["enhanced25"],
                     selected_accuracy=main, calibrated_baseline25=calibrated)
        for source in ("baseline", "enhanced"):
            spec_blend = next(p for p in plan["primary_candidates"] if p["name"] == source + "_validated_blend")
            local[source + "_selected_blend"], _ = fit_predict(spec_blend, vdata, vy, tdata, plan["blend_weight_grid"])
        indices = [r["path_index"] for r in test]
        for name, p in local.items():
            methods[name][indices] = p
        if threshold is not None:
            published[indices] = np.maximum(main, 1-main) >= threshold
        models[s] = dict(accuracy=bundle, calibration=cal_bundle, threshold=threshold,
                         policy_subjects=np.unique(vg).tolist(), numerical_candidate=numerical[s, indices[0]]["selected_numeric"])
        diagnostics.append(dict(outer_subject=s, policy_paths=len(val), selected_accuracy=spec["name"],
                                accuracy_cv=cv_diagnostics, selected_calibration=cal_spec["name"],
                                calibration_cv=cal_diagnostics, selective_threshold=threshold))
    summary = dict(vendor=vendor, status="completed_exploratory_147_outer_paths", paths=147, subjects=24,
                   methods={name: scored(y, p) for name, p in methods.items()}, comparisons={}, api=api,
                   selection_counts=dict(Counter(d["selected_accuracy"] for d in diagnostics)),
                   calibration_counts=dict(Counter(d["selected_calibration"] for d in diagnostics)),
                   selective=dict(accepted_paths=int(published.sum()), coverage=float(published.mean()),
                        correct_accepted=int(np.count_nonzero((methods["selected_accuracy"][published] >= .5) == y[published])),
                        accepted_acc=float(np.mean((methods["selected_accuracy"][published] >= .5) == y[published])) if published.any() else None,
                        thresholds=thresholds, validation_target=.80, validation_min_coverage=.70),
                   outer_labels_used_for_selection=False, already_examined_outer_data=True)
    for name, candidate, reference in (("selected_minus_baseline25", "selected_accuracy", "baseline25"),
                                      ("selected_minus_numeric", "selected_accuracy", "numeric"),
                                      ("enhanced25_minus_baseline25", "enhanced25", "baseline25"),
                                      ("calibrated_minus_baseline25", "calibrated_baseline25", "baseline25")):
        summary["comparisons"][name] = paired(y, subjects, methods[candidate], methods[reference])
    summary["retention"] = acceptance(summary, protocol)
    previous = read_json(DEEPSEEK / "summary.json")["methods"][vendor + "_fusion25"]
    if summary["methods"]["baseline25"]["correct_paths"] != previous["correct_paths"]:
        raise ValueError("The fixed baseline comparison changed")
    directory = out / vendor
    write_json(directory / "evaluation.json", summary)
    write_json(directory / "selection_diagnostics.json", diagnostics)
    joblib.dump(models, directory / "fitted_policies.joblib")
    write_csv(directory / "path_oof.csv", [dict(path_index=i, subject_key=int(subjects[i]), label=int(y[i]),
        **{name: float(p[i]) for name, p in methods.items()},
        accepted=int(published[i]), publication_state=("high" if methods["selected_accuracy"][i] >= .5 else "low") if published[i] else "uncertain")
        for i in range(147)])
    return summary, log_paths


def report(summary):
    complete = not summary["incomplete_vendors"]
    promoted = any(s["retention"]["accuracy_promoted"] or s["retention"]["calibration_promoted"] for s in summary["vendors"].values())
    outcome = ("GPT 与 DeepSeek 均已完成全部请求。" if complete else "目前只评价已完成的模型。")
    if not promoted:
        outcome += "已完成方案均未通过试验前固定的保留门槛，原默认流程继续保留；新增代码和结果作为试验记录保存。"
    else:
        outcome += "以下采用决定按试验前固定的门槛分别判断。"
    lines = ["# 云端证据改进试验", "", outcome, "",
             "评价对象是24名被试的147条路径级问卷标签，分数<30为低类、≥30为高类；不是每5秒的实时症状真值。每家新增1309个匿名单路径请求，原提示的外层预测复用既有记录。", "",
             "原 EEG 模型、被试划分和既有结果保留。本轮联合新增相似案例、校准头留一被试预测及内部可靠性信息，并在各折 4 个政策被试上选择简单融合／校准。这不能单独归因于某一个组件。", "",
             "选择规则在完整 API 运行前固定；外层测试标签不用于新提示、检索、融合拟合或阈值选择。数据此前已经被多次查看，结果仍属探索。", "",
             "## 准确率与概率质量", "", "| 模型 | 方法 | 判对 / 147 | ACC | BACC | Brier（越低越好） |", "|---|---|---:|---:|---:|---:|"]
    labels = {"numeric": "纯数值", "baseline_cloud": "原云端单独判断", "baseline25": "原提示固定云端25%", "enhanced_cloud": "改进云端单独判断",
              "enhanced25": "改进提示固定云端25%", "baseline_selected_blend": "原提示内部选权重",
              "enhanced_selected_blend": "改进提示内部选权重", "selected_accuracy": "内部选择融合方案（主要结果）",
              "calibrated_baseline25": "原25%结果加内部校准"}
    for vendor, s in summary["vendors"].items():
        for name, label in labels.items():
            m = s["methods"][name]
            lines.append(f"| {vendor} | {label} | {m['correct_paths']} | {m['acc']:.2%} | {m['bacc']:.2%} | {m['brier']:.4f} |")
    lines += ["", "## 是否保留", "", "准确率采用门槛：比原固定25%多判对至少5条、被试聚类95%增量区间下界大于0、BACC不降、Brier增加不超过0.005、API成功率至少99%。", "",
              "校准单独采用门槛：Brier相对下降至少10%、改善区间下界大于0、全样本最多少判对1条、API成功率至少99%。所有方法完整计入失败回退，未删除难例。", ""]
    for vendor, s in summary["vendors"].items():
        d = s["comparisons"]["selected_minus_baseline25"]
        lo, hi = d["acc_gain_subject_bootstrap_95ci"]
        keep = s["retention"]
        lines += [f"- **{vendor}**：主要结果比原25%增加 {d['acc_gain']*100:+.2f} 个百分点；95%区间 [{lo*100:+.2f}, {hi*100:+.2f}]。纠正 {d['corrected']} 条，误改 {d['harmed']} 条。",
                  f"  准确率改动：{'达到门槛，可保留' if keep['accuracy_promoted'] else '未达到门槛，不替换默认流程'}；校准改动：{'达到门槛，可保留' if keep['calibration_promoted'] else '未达到门槛，不替换默认置信度'}。"]
        fixed = s["comparisons"]["enhanced25_minus_baseline25"]
        lines += [f"  提示改进固定25%：纠正 {fixed['corrected']} 条，误改 {fixed['harmed']} 条；{fixed['subjects_net_improved']} 名被试净改善、{fixed['subjects_net_worsened']} 名净变差。此项为预列对照，不能看完外层结果后替换主要方案。"]
        cal = s["comparisons"]["calibrated_minus_baseline25"]
        clo, chi = cal["brier_gain_subject_bootstrap_95ci"]
        relative = cal["brier_gain"] / s["methods"]["baseline25"]["brier"]
        lines += [f"  校准使Brier相对下降 {relative:.2%}，改善量95%区间 [{clo:+.4f}, {chi:+.4f}]；区间包含0时，不能认定概率可靠性已稳定提高。"]
        sel = s["selective"]
        acc = "无法估计" if sel["accepted_acc"] is None else f"{sel['accepted_acc']:.2%}"
        lines += [f"  内部选择的明确发布门槛在外层接受 {sel['accepted_paths']}/147 条（覆盖率 {sel['coverage']:.2%}），接受样本准确率 {acc}。这不等于全样本准确率，也不是置信度认证。"]
    lines += ["", "选择性发布的目标是至少70%覆盖率、接受样本准确率至少80%，需在外层同时达到。拒判后的条件准确率不能算作全样本ACC提高。", "",
              "## 后续改进方向", "",
              "当前证据支持先增加独立被试和内部验证样本，再判断复杂融合是否值得保留。每折只有5名参考被试和4名政策被试，验证集选出的方案未必适用于新的被试；本轮无法证明这一因素是性能变化的唯一原因。", "",
              "下一轮优先检查数值模型在不同被试上的概率漂移和误判，并验证任务前无标签基线及更强的数值模型。若继续使用LLM，可预先限制其处理模型分歧或低把握案例，保留理由和证据，仍由内部被试选择复核策略。以上属于后续候选，需要新的独立验证，尚未实施或证明有效。", ""]
    for vendor, reason in summary["incomplete_vendors"].items():
        lines += [f"- **{vendor} 未完成**：{reason}。不得将部分请求或失败回退的结果描述为完整改进模型效果。"]
    lines += ["", "## 协议限制", "", "- 参考案例只来自本折5个元校准被试；检索尺度只由这部分数据确定，查询和测试标签不参与。", "- 示例概率的校准头进行了留一元被试重拟合；冻结MIL编码器选训练轮次时曾使用元被试标签，所以这不是整个深度模型完全交叉拟合的可靠性估计。", "- 原数值候选已经在4个政策被试上选择；本轮融合交叉验证以这个冻结选择为条件，没有重跑整个数值模型的内层选择。", "- 置信度仍需独立数据检验。LLM自报不确定标志的减少不算效果；没有外层标签搜索阈值或删除失败案例。", "",
              "## 实际 API 调用", ""]
    for vendor, s in summary["vendors"].items():
        a = s["api"]
        lines += [f"- {vendor}：{a['successful_logical_requests']}/{a['logical_requests']} 个逻辑请求成功；实际尝试 {a['attempted_requests']} 次，输入token {a['tokens']['input']}、输出token {a['tokens']['output']}。金额未核实。"]
    lines += ["", "GPT实际返回模型为`gpt-6.1-sol`，DeepSeek为原项目配置的`deepseek-flash`。DeepSeek充值恢复只重试基础设施失败请求，原失败记录保留，规则见`operational_amendment.json`。", "",
              "逐路径结果、每折选择、模型参数和API原始记录均保存在相应vendor子目录。`analysis_plan.json`和`protocol.json`记录预先固定的方法与门槛；`promotion.json`记录采用决定。`validation.json`记录来源哈希、匿名单路径合同和48次保存参数重载检查；`independent_verification.json`另行从CSV重算ACC/BACC/Brier并核对调用统计。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--vendors", nargs="+", choices=("gpt", "deepseek"), default=["gpt", "deepseek"])
    args = parser.parse_args()
    protocol, plan = read_json(args.out / "protocol.json"), read_json(args.out / "analysis_plan.json")
    jobs = read_json(args.out / "jobs.json")
    if read_json(args.out / "structural_validation.json")["status"] != "passed":
        raise ValueError("The request/source structural audit must pass")
    for name, expected in protocol["original_sha256"].items():
        if sha(name) != expected:
            raise ValueError("A protected original changed")
    summary = dict(status="completed_available_vendors", vendors={}, incomplete_vendors={}, analysis_plan_sha256=sha(args.out / "analysis_plan.json"))
    logs = []
    for vendor in ("gpt", "deepseek"):
        path = args.out / vendor / "run_status.json"
        if vendor not in args.vendors or not path.exists() or read_json(path)["status"] != "completed":
            summary["incomplete_vendors"][vendor] = "独立请求未全部完成；查看run_status及基础设施诊断"
            continue
        summary["vendors"][vendor], paths = evaluate_vendor(args.out, vendor, protocol, plan, jobs)
        logs += paths
    if not summary["vendors"]:
        raise ValueError("No complete vendor can be evaluated")
    for name, expected in protocol["original_sha256"].items():
        if sha(name) != expected:
            raise ValueError("A protected original changed during evaluation")
    write_json(args.out / "summary.json", summary)
    write_json(args.out / "promotion.json", dict(scope="exploratory local deployment only", decisions={v:s["retention"] for v,s in summary["vendors"].items()},
                incomplete_vendors=summary["incomplete_vendors"], original_entrypoints_modified=False))
    write_json(args.out / "validation.json", dict(status="passed_for_completed_vendors", completed_vendors=list(summary["vendors"]),
        incomplete_vendors=summary["incomplete_vendors"], audited_cloud_logs=len(logs), original_hashes_unchanged=True,
        single_path_requests_checked=True, no_new_outer_label_selection=True, baseline_metrics_reproduced=True,
        fitted_policy_reloads_checked=24*len(summary["vendors"]),
        analysis_plan_sha256=summary["analysis_plan_sha256"], implementation_sha256={p.name:sha(p) for p in Path("vrms_refine").glob("*.py")}))
    (args.out / "改进试验报告.md").write_text(report(summary), encoding="utf-8")
    print(__import__("json").dumps({v:dict(methods={n:(m["correct_paths"],m["acc"],m["brier"]) for n,m in s["methods"].items()}, retention=s["retention"],selective=s["selective"])
                                  for v,s in summary["vendors"].items()},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
