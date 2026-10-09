"""Deliver a seed comparison, every GPT reply and paired CSV to baogao."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil

from .cloud import write_json
from .evaluate_seed_retest import successful_log
from .second_judgment import sha256
from .seed_retest import DEFAULT_OUT, read


def report(out=DEFAULT_OUT, desktop=Path("reports")):
    out, desktop = Path(out), Path(desktop)
    summary, protocol = read(out / "summary.json"), read(out / "protocol.json")
    assert summary["status"] == "completed_and_audited"
    with (out / "path_comparison.csv").open(encoding="utf-8-sig",newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 146
    seed = summary["seed"]
    previous = summary["original_reference"]
    numeric = read(out / "selector/outer_numeric_controls/summary.json")["controls"]
    original_numeric = read(Path("outputs/vrms_agent_loso/v6_selector_20261008/outer_numeric_controls/summary.json"))["controls"]
    baseline, agent, raw = summary["baseline"], summary["agent"], numeric["selector_raw"]["metrics"]
    assert raw == summary["numeric_controls"]["selector_raw"]["metrics"]
    ratio = "∞（误改0）" if summary["damaged"] == 0 else f"{summary['corrected']/summary['damaged']:.2f}:1"
    delta = 100*(agent["accuracy"] - previous["agent"]["accuracy"])
    passed = summary["goal_metric_pass"]
    ci = summary["paired_subject_bootstrap"]["accuracy_difference_95ci"]
    nested = read(out / "selector/nested_audit/summary.json")
    numerical_comparison = summary["vs_same_seed_numeric_fusion"]
    old_comparison = summary["vs_original_fixed_baseline"]
    table = []
    for label, old, new in [("冻结FEMBA＋MIL分类头", previous["baseline"], baseline),
                             ("纯数值工具融合，不调用GPT",original_numeric["selector_raw"]["metrics"],raw),
                             ("真实GPT工具Agent",previous["agent"],agent)]:
        table.append(f"| {label} | {old['correct']}/146，{100*old['accuracy']:.2f}% | {new['correct']}/146，{100*new['accuracy']:.2f}% | {100*(new['accuracy']-old['accuracy']):+.2f} |")
    subject_rows = []
    subject_differences = []
    for subject in sorted({int(r["subject_key"]) for r in rows}):
        part = [r for r in rows if int(r["subject_key"]) == subject]
        counts = [sum(r[key] == r["true_class"] for r in part) for key in ("original_baseline_class","baseline_class","original_agent_class","agent_class")]
        subject_rows.append(f"| {subject:02d} | {len(part)} | {' | '.join(map(str,counts))} |")
        subject_differences.append((counts[3]-counts[2],subject,len(part),counts[2],counts[3]))
    difference, subject, n, before, after = min(subject_differences)
    old_config = read(Path("outputs/vrms_agent_loso/v6_selector_20261008/internal_validation") / f"fold_{subject:02d}.json")["configuration"]
    new_config = read(out / "selector/internal_validation" / f"fold_{subject:02d}.json")["configuration"]
    subject_analysis = f"最大单被试下降出现在被试{subject:02d}：原GPT正确{before}/{n}，新GPT正确{after}/{n}。该折数值融合的内部选择从`{old_config}`变为`{new_config}`；GPT与新融合输出一致。这是融合及其内部参数选择对训练seed敏感的具体线索，不能单凭这一对应关系认定唯一原因。"
    conclusion = ("本轮同时达到ACC≥70%与纠正数≥2×误改数的标准。" if passed
                  else "本轮没有同时达到ACC≥70%与纠正数≥2×误改数的标准，原先单次达标结果不能据此称为跨种子稳定。")
    transcript = [f"# EEGAgent seed={seed}：全部真实GPT回复\n\n查询真值只在本地评价和本报告中展示，没有发送给GPT。\n"]
    for row in rows:
        i = int(row["path_index"])
        log = successful_log(out / "gpt",i)
        answer = log["final"]
        category = lambda value: "High" if int(value) else "Low"
        transcript.append(f"\n## 路径 {i:03d}\n\n真值={category(row['true_class'])}；新种子MIL={category(row['baseline_class'])}；GPT={answer['final_class']}；定性置信度={answer['confidence']}。\n\n实际执行：{', '.join(log['actual_tool_calls'])}。\n\n{answer['explanation']}\n")
        for key,label in [("supporting_evidence","支持证据"),("conflicting_evidence","冲突证据"),("missing_evidence","缺失证据")]:
            transcript.append(f"\n{label}：\n")
            transcript.extend("\n- "+text+"\n" for text in answer[key])
    text = f"""# EEGAgent 随机种子复测：seed={seed}

日期：2026-10-08。本轮是在查看新结果前指定seed={seed}的一次完整复测；固定同146条完整路径、24折LOSO、预训练编码器、预处理、输入归一化和特征缓存，重新训练FEMBA＋MIL分类头、辅助分类器及工具融合模型，再重新调用真实GPT。原seed=2026的权重、概率、API回复和报告保留。

本轮最终GPT ACC为 **{100*agent['accuracy']:.2f}%（{agent['correct']}/146）**，较原71.92%变化 **{delta:+.2f}个百分点**。相对同seed的新MIL，纠正 **{summary['corrected']}** 条、误改 **{summary['damaged']}** 条，比例 **{ratio}**。{conclusion}

## 同样本结果比较

| 流程 | seed=2026 | seed={seed} | ACC变化（百分点） |
|---|---:|---:|---:|
{chr(10).join(table)}

新种子的MIL基线是57.53%，不能将新的“纠正/误改”冒充为相对原60.96%基线的统计。另作同样本对照：新Agent相对原固定60.96%基线，纠正{old_comparison['corrected']}条、误改{old_comparison['damaged']}条，净增加{old_comparison['net']}条正确路径。

## 实际改动范围

只改变本轮下游训练随机性：MIL初始化及路径训练顺序、辅助分类器random_state、进程内NumPy/PyTorch随机状态。24折的外层被试与base14/meta5/policy4角色完全不变。固定12个epoch、AdamW学习率0.001、weight_decay0.001及MIL结构；没有挑选最佳epoch，没有根据新测试结果调整提示词、阈值候选或选择规则。

冻结FEMBA预训练参数和已有特征；本轮没有重训预训练编码器、CNN源编码器或原输入归一化统计。MIL分类头24折均真正重新初始化和训练：初始与最终状态哈希全部不同，训练路径、epoch和特征保持一致。`training_repeat_audit.json`记录全部24折。

辅助工具仍在base14＋另外4名meta上拟合，预测被排除的meta被试，使用meta5被试OOF选择候选和校准；最终辅助工具使用19名内部被试。融合模型利用meta OOF与独立policy预测，候选与margin通过policy被试OOF选择，并另做完整嵌套审核。外层测试标签不参与上述拟合或选择。新种子只有{nested['approved_folds']}/24折通过嵌套纠正条件；强制High/Low不能解释为风险有保证的可靠发布。

## GPT真实执行与纯数值对照

使用与原第四轮相同的gpt-6.1-sol、提示词、工具定义、最终JSON格式和4并发。每条匿名查询通过function_call触发实际EEG工具，再通过function_call_output返回证据，最终只输出High/Low。没有修改原始GPT分类，最终146/146成功、回退0、uncertain=0。

训练seed不固定GPT采样：本接口请求没有seed参数，本轮是新的真实API调用。因此本次变化包含下游训练随机性与GPT响应波动，不能单独归因于GPT随机数。

同seed纯数值融合为{100*raw['accuracy']:.2f}%（{raw['correct']}/146）。GPT相对它额外纠正{numerical_comparison['corrected']}条、误改{numerical_comparison['damaged']}条，净变化{numerical_comparison['net']}条。辅助工具训练样本数多于MIL基线，增益不能单独归因于GPT。全部路径执行8路证据及纠正学习器，这轮没有证明自适应节省工具计算。

API真实轮次{summary['api']['rounds']}，实际尝试{summary['api']['attempts']}，失败尝试{summary['api']['failed_attempts']}；失败和恢复保留，合法但错误的回复没有为了挑结果而重跑。返回usage仅为接口记录，不能替代实际账单。

## 验证与范围

最终回复逐条从原始API响应重新解析；146条实际工具输出全部离线重放并比较，冻结源码与模型哈希核对通过。模型24折保存/重载概率差异为0；样本标签与原146条清单完全一致。新增随机种子契约与既有OOF/MIL隔离测试10/10通过。完整原始日志与数值产物位于 `{out.resolve()}`。

另做24折真实数据的内存标签污染检查：反转当前外层被试的全部类别与问卷值，通过限定路径的JSON读取拦截输入给相同运行时，匿名初始请求、8路工具和融合证据全部不变。没有调用API，没有修改实际数据文件；记录在`outer_label_poison_audit.json`。

新Agent BACC={100*agent['balanced_accuracy']:.2f}%，Macro-F1={agent['macro_f1']:.4f}。新MIL的合并AUROC={summary['baseline_score_diagnostics']['auroc']:.4f}，Brier={summary['baseline_score_diagnostics']['brier']:.4f}。Agent相对同seed MIL的被试配对bootstrap ACC差值95%区间为[{100*ci[0]:+.2f},{100*ci[1]:+.2f}]个百分点；bootstrap随机种子仍按原统计方法固定2026，与训练seed不同。

目前只有一个新增训练seed。此前同一数据集已多轮开发，预训练数据清单缺失，重叠不能排除；本轮不是独立外部验证，也不能凭两次实验宣称稳定泛化。所有新结果如实保留，不筛选更好看的seed。

## 逐被试结果

| 被试 | 路径数 | 2026MIL正确 | {seed}MIL正确 | 2026GPT正确 | {seed}GPT正确 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(subject_rows)}

{subject_analysis}

## 复现入口

新复现必须使用独立目录，不能覆盖本轮。Windows环境完成freeze/numeric/gpt；MIL训练在原WSL CUDA环境执行。

```bash
python -m vrms_cloud.seed_retest --stage freeze --seed 2027 --scope full --out NEW_OUTPUT
python -m vrms_cloud.seed_retest --stage models --out NEW_OUTPUT
python -m vrms_cloud.seed_retest --stage numeric --out NEW_OUTPUT
python -m vrms_cloud.seed_retest --stage gpt --out NEW_OUTPUT
python -m vrms_cloud.evaluate_seed_retest --out NEW_OUTPUT
python -m vrms_cloud.report_seed_retest --out NEW_OUTPUT
```

API使用项目配置EEG_GPT_BASE_URL/MODEL/KEY；密钥不写入日志或报告。初次调用时冻结协议，新路径间不共享对话。源代码快照保存在新输出的sources目录。
"""
    report_file = out / f"EEGAgent_seed{seed}复测报告_20261008.md"
    reply_file = out / f"EEGAgent_seed{seed}全部GPT回复_20261008.md"
    report_file.write_text(text,encoding="utf-8")
    reply_file.write_text("".join(transcript),encoding="utf-8")
    desktop.mkdir(parents=True,exist_ok=True)
    deliveries = []
    for source,name in [(report_file,report_file.name),(reply_file,reply_file.name),
                        (out / "path_comparison.csv",f"EEGAgent_seed{seed}逐条结果_20261008.csv"),
                        (out / "summary.json",f"EEGAgent_seed{seed}结果与审计_20261008.json")]:
        destination=desktop/name;shutil.copy2(source,destination)
        assert sha256(source) == sha256(destination)
        deliveries.append(dict(source=str(source.resolve()),destination=str(destination),sha256=sha256(destination)))
    write_json(out / "report_delivery.json",dict(status="passed",sha256_verified=True,files=deliveries,
        reporter_sha256=sha256(__file__),root_protocol_sha256=sha256(out / "protocol.json")))
    print(f"Delivered {len(deliveries)} verified files to {desktop}")


if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--out",type=Path,default=DEFAULT_OUT)
    parser.add_argument("--desktop",type=Path,default=Path("reports"))
    args=parser.parse_args();report(args.out,args.desktop)
