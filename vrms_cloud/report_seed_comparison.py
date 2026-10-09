"""Report the three preserved training-seed runs without selecting a winner."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil

import numpy as np

from .cloud import write_json
from .evaluate_seed_retest import successful_log
from .second_judgment import sha256
from .seed_retest import read, OLD_GPT
from .seed_retest_2028 import DEFAULT_OUT


def report(out=DEFAULT_OUT, desktop=Path("reports")):
    out, desktop = Path(out), Path(desktop)
    latest = read(out / "summary.json")
    assert latest["status"] == "completed_and_audited" and latest["seed"] == 2028
    for filename in ("training_repeat_audit.json", "outer_label_poison_audit.json"):
        assert read(out / filename)["status"] == "passed"
    roots = [OLD_GPT, Path("outputs/vrms_agent_seed_retest/seed2027_20261008"), out]
    numeric_roots = [Path("outputs/vrms_agent_loso/v6_selector_20261008"), roots[1] / "selector", out / "selector"]
    summaries = [read(root / "summary.json") for root in roots]
    records, tables = [], []
    for seed, root, numeric_root, s in zip((2026, 2027, 2028), roots, numeric_roots, summaries):
        numeric = read(numeric_root / "outer_numeric_controls/summary.json")["controls"]["selector_raw"]["metrics"]
        if seed != 2026:
            assert numeric == s["numeric_controls"]["selector_raw"]["metrics"]
        passed = s["goal"]["metrics_pass"] if seed == 2026 else s["goal_metric_pass"]
        ratio = s["corrected"] / s["damaged"] if s["damaged"] else None
        record = dict(seed=seed, mil=s["baseline"], numeric_fusion=numeric, gpt=s["agent"],
            corrected=s["corrected"], damaged=s["damaged"], correction_damage_ratio=ratio,
            metric_pass=passed, source=str(root.resolve()), summary_sha256=sha256(root / "summary.json"))
        records.append(record)
        cell = lambda m: f"{100*m['accuracy']:.2f}%（{m['correct']}/146）"
        ratio_text = "∞" if ratio is None else f"{ratio:.2f}:1"
        tables.append(f"| {seed} | {cell(s['baseline'])} | {cell(numeric)} | {cell(s['agent'])} | {s['corrected']} | {s['damaged']} | {ratio_text} | {'是' if passed else '否'} |")
    with (out / "path_comparison.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 146
    with (roots[1] / "path_comparison.csv").open(encoding="utf-8-sig", newline="") as f:
        previous_rows = list(csv.DictReader(f))
    for a, b in zip(rows, previous_rows):
        assert all(a[key] == b[key] for key in ("path_index", "subject_key", "true_class"))
    subject_table = []
    for subject in sorted({int(r["subject_key"]) for r in rows}):
        indices = [i for i, r in enumerate(rows) if int(r["subject_key"]) == subject]
        counts = [sum(rows[i]["original_agent_class"] == rows[i]["true_class"] for i in indices),
                  sum(previous_rows[i]["agent_class"] == previous_rows[i]["true_class"] for i in indices),
                  sum(rows[i]["agent_class"] == rows[i]["true_class"] for i in indices)]
        subject_table.append(f"| {subject:02d} | {len(indices)} | {' | '.join(map(str, counts))} |")
    ci = latest["paired_subject_bootstrap"]["accuracy_difference_95ci"]
    num = latest["vs_same_seed_numeric_fusion"]
    fixed = latest["vs_original_fixed_baseline"]
    nested = read(out / "selector/nested_audit/summary.json")
    means = {key: dict(mean=float(np.mean([r[key]["accuracy"] for r in records])),
                      sample_sd=float(np.std([r[key]["accuracy"] for r in records], ddof=1)))
             for key in ("mil", "numeric_fusion", "gpt")}
    pass_count = sum(r["metric_pass"] for r in records)
    title = "EEGAgent_三随机种子对照报告_20261008.md"
    content = f"""# EEGAgent：2026、2027、2028随机种子对照

同146条完整路径、24名被试、24折LOSO；问卷分数<30为Low、≥30为High，评价单位是整条路径。本轮在查看结果前指定训练seed=2028，实际重新训练MIL头和辅助数值模型，再进行146条真实GPT调用。最终全部High/Low，uncertain=0，回退=0。

| 训练Seed | 冻结FEMBA＋MIL ACC | 纯数值融合 ACC | GPT工具Agent ACC | 纠正 | 误改 | 纠正/误改 | 达到双标准 |
|---|---:|---:|---:|---:|---:|---:|:---:|
{chr(10).join(tables)}

纠正和误改均相对同一seed的MIL统计。“达到双标准”要求ACC≥70%，且纠正数≥2×误改数。三轮有{pass_count}/3达到；这三个seed在同一开发数据上运行，不能据此宣称稳定的外部泛化。

三轮GPT平均ACC={100*means['gpt']['mean']:.2f}%，样本标准差={100*means['gpt']['sample_sd']:.2f}个百分点；这是三次完整运行ACC的描述统计，不是24折标准差，也不是独立样本置信区间。

## seed2028的结果含义

MIL正确{latest['baseline']['correct']}/146，GPT正确{latest['agent']['correct']}/146；相对同seed MIL纠正{latest['corrected']}条、误改{latest['damaged']}条，净增加{latest['net']}条正确预测。另与最初固定60.96%的MIL基线对照：纠正{fixed['corrected']}条、误改{fixed['damaged']}条，净增加{fixed['net']}条。这两种参照不能混写。

GPT相对同seed纯数值融合额外纠正{num['corrected']}条、误改{num['damaged']}条，净变化{num['net']}条；预测不同的路径共{num['corrected']+num['damaged']}条。因此必须保留数值融合对照，不能将整个增益归因于GPT。

GPT的BACC={100*latest['agent']['balanced_accuracy']:.2f}%，Macro-F1={latest['agent']['macro_f1']:.4f}。MIL合并AUROC={latest['baseline_score_diagnostics']['auroc']:.4f}，Brier={latest['baseline_score_diagnostics']['brier']:.4f}。GPT相对同seed MIL的被试配对bootstrap ACC差值95%区间为[{100*ci[0]:+.2f},{100*ci[1]:+.2f}]个百分点。统计bootstrap保持原seed=2026。

## 固定的实验条件

预训练FEMBA、EEG、特征缓存和输入归一化不变；24折outer/base14/meta5/policy4角色不变。MIL仅用base14训练，固定12个epoch和原AdamW配置。辅助分类器使用meta5被试OOF选择与校准，最终使用base14+meta5的19名内部被试；policy4继续独立用于策略评估，融合保留原来的候选、margin和嵌套审核规则。外层测试标签不参与拟合或参数选择。

只改变下游训练随机性；提示词、工具、JSON格式及gpt-6.1-sol保持不变。接口请求没有seed参数，所以训练seed不能固定GPT采样，本轮变化不能单独归因于LLM随机性。没有依据外层结果调整模型、提示词或阈值，也没有为了更好的ACC重跑合法回复。

本轮数值融合{nested['approved_folds']}/24折通过嵌套纠正审核。强制High/Low是此次实验口径，不能解释为每条预测都已通过可靠发布门控。全部路径都调用8路EEG证据与纠正融合，这轮没有证明自适应节省计算。

## 实际调用与核验

完成146/146真实GPT回复；真实API轮次{latest['api']['rounds']}、尝试{latest['api']['attempts']}、失败尝试{latest['api']['failed_attempts']}。失败记录保留；返回usage是接口记录，不等同实际账单。

最终分类由原始API JSON重新解析；146条工具输出逐条重放并比较，冻结源码、特征、模型和角色哈希通过核验。24折MIL训练样本及12个epoch均与前两轮一致，初始/最终权重哈希均不同，保存重载预测差异为0。24折再做外层标签内存污染检查：翻转当前外层被试全部标签与问卷值，初始请求、8路工具及融合输出不变；没有额外调用API或修改原数据。

辅助工具的训练被试数多于MIL基线；同一数据已经多轮开发；预训练重叠未排除。本表用于随机种子敏感性观察，仍需独立数据和公平训练样本对照来判断泛化及GPT贡献。

## 逐被试GPT正确数

| 被试 | 路径数 | 2026 | 2027 | 2028 |
|---|---:|---:|---:|---:|
{chr(10).join(subject_table)}

## 数据与复现入口

本轮完整产物：`{out.resolve()}`。保留前两轮输出。使用独立新目录复现：先`python -m vrms_cloud.seed_retest_2028 --stage freeze --out NEW_OUTPUT`，再依次执行同入口的`models`、`numeric`、`gpt`阶段。MIL阶段使用原WSL CUDA环境，数值训练与API使用原Windows虚拟环境；GPT从环境变量读取项目已有接口配置，密钥不写入报告。

评分：`python -m vrms_cloud.evaluate_seed_retest --out NEW_OUTPUT`。本报告从已审计summary与原始逐路径结果生成；报告脚本不参与训练、选择或预测。
"""
    transcript = ["# EEGAgent seed2028：全部真实GPT回复\n\n真值只用于本地评价及本报告，没有发送给GPT。\n"]
    category = lambda value: "High" if int(value) else "Low"
    for r in rows:
        i = int(r["path_index"])
        log = successful_log(out / "gpt", i)
        answer = log["final"]
        transcript.append(f"\n## 路径 {i:03d}\n\n真值={category(r['true_class'])}；MIL={category(r['baseline_class'])}；GPT={answer['final_class']}；定性置信度={answer['confidence']}。\n\n执行工具：{', '.join(log['actual_tool_calls'])}。\n\n{answer['explanation']}\n")
        for key, label in (("supporting_evidence", "支持证据"), ("conflicting_evidence", "冲突证据"), ("missing_evidence", "缺失证据")):
            transcript.append(f"\n{label}：\n")
            transcript.extend(f"\n- {value}\n" for value in answer[key])
    (out / title).write_text(content, encoding="utf-8")
    reply_name = "EEGAgent_seed2028全部GPT回复_20261008.md"
    (out / reply_name).write_text("".join(transcript), encoding="utf-8")
    write_json(out / "three_seed_comparison.json", dict(paths=146, folds=24, records=records,
        accuracy_description=means, seeds_meeting_both_criteria=pass_count,
        reporter_sha256=sha256(__file__)))
    desktop.mkdir(parents=True, exist_ok=True)
    deliveries = []
    for source, name in ((out / title, title), (out / reply_name, reply_name),
            (out / "path_comparison.csv", "EEGAgent_seed2028逐条结果_20261008.csv"),
            (out / "summary.json", "EEGAgent_seed2028结果与审计_20261008.json"),
            (out / "three_seed_comparison.json", "EEGAgent_三随机种子对照_20261008.json")):
        destination = desktop / name
        assert not destination.exists(), f"Preserve existing delivery: {destination}"
        shutil.copy2(source, destination)
        assert sha256(source) == sha256(destination)
        deliveries.append(dict(source=str(source.resolve()), destination=str(destination), sha256=sha256(destination)))
    write_json(out / "report_delivery.json", dict(status="passed", sha256_verified=True,
        files=deliveries, reporter_sha256=sha256(__file__)))
    print("\n".join(tables))
    print(f"Delivered {len(deliveries)} verified files to {desktop}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--desktop", type=Path, default=Path("reports"))
    args = parser.parse_args()
    report(args.out, args.desktop)
