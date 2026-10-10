# 优化 RAG：已完成的 146 路径配对实验

本记录对应 2026-10-11 已完成的 GPT 对照，不替换网页的 2026-10-09 评价。两组均有 146 个真实、已完成的 LLM 分类，最终结果如下：

| 设置 | 正确 / 总数 | ACC | balanced ACC |
| --- | ---: | ---: | ---: |
| 无 RAG | 104 / 146 | 71.23% | 71.41% |
| `percentile_group` RAG | 107 / 146 | 73.29% | 73.21% |

配对纠正 12 条、损坏 9 条，净增 3 条，ACC 增加 2.0548 个百分点。原定净增至少 4 条的目标未达到。24 条试点为 17 对 20，净增 3 条；其余 122 条净增 0 条。12 条 `compact_same` 候选没有扩展为全集，其结果和成本不计入最终任一组的 146 条 ACC。

## LLM 实际看到什么

两组固定同一份系统提示、MIL 原始 sigmoid、八个工具原始决策分数、可用校准值和内部验证资料。无 RAG 表示去掉检索的参考案例，仍有 EEG 工具证据，LLM 根据这些读数独立判定。类别定义是整路径结束时问卷值 ≥30 为 High，否则为 Low；当前查询问卷值、真实类别、旧回答与融合类别/分数不进入模型输入。

固定配置为 `gpt-6.1-sol`、Responses、`reasoning_effort=high`、`max_output_tokens=8192`。最终六字段为 `final_class`、`confidence`、`explanation`、`supporting_tools`、`counterevidence_tools`、`rag_citations`。评分直接读取实际 LLM 的 `final_class`，没有按置信度或算法投票替换类别。

## 检索实现

[decisive_rag.py](../eeg_agent/decisive_rag.py) 保留冻结实验的数值与选择逻辑。`percentile_group` 使用八个工具原始分数在同折参考库中的无标签经验百分位，平分四组权重：VRMSModel 特征/均值/时间、频谱、空间/CSP、训练案例检索。百分位对并列值取中秩，不对极端分数做裁剪；MIL 只作为读数展示，不参与案例距离或选择。

每个参考类别先取各被试最近的至多两条候选，从中选择每类两条核心案例，先最大化不同被试覆盖，再最小化距离；若被试仍未覆盖，追加最近的缺失被试，至多六条。上下文保留案例原始读数、校准值、整库类别中位数/IQR、工具方向错误计数和按被试列出的相同来源组符号模式。距离和标签计数用于展示证据，代码不生成当前查询的最终类别或推荐类别。

参考被试与查询被试分离，且未用于该折工具头和校准器拟合。不过这些参考被试参与过内部选择器开发，当前 146 条数据也经过多轮开发，属于开发集评价。每折只有四个参考被试，多个路径和共享来源工具存在依赖，不能当作独立外部验证。

## 置信度与引用处理

| 置信度 | 无 RAG 正确 / 条数 | ACC | 有 RAG 正确 / 条数 | ACC |
| --- | ---: | ---: | ---: | ---: |
| low | 8 / 15 | 53.33% | 6 / 15 | 40.00% |
| medium | 64 / 93 | 68.82% | 67 / 91 | 73.63% |
| high | 32 / 38 | 84.21% | 34 / 40 | 85.00% |

这里存在置信度与正确率的关联，分组由模型判断后产生，组内样本也会变化；不能据此认定改变 confidence 会使类别更准确。所有置信度组都计入最终 ACC。

请求 schema 的工具与案例引用是字符串数组。无 RAG 有 10 条已完成六字段输出包含未知工具字符串，这些原生六字段完整保留，未知引用另记警告，不能称为已核验引用；没有因此重采样分类。此前两个明确的 `baseline`→`MIL` 引用别名修复有独立历史记录，只涉及元数据。公开的 [rag_experiment.py](../eeg_agent/rag_experiment.py) 将分类有效性和引用警告分开，保留原生字段，并拒绝未完成或拒答响应。未知名称不会被自动改成真实工具。

## 补跑来源与成本

| 设置 | CPA 有效判断 | 中转有效判断 | 启动尝试 | 保存原始响应 | 保存响应 input / output tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| 无 RAG | 105 | 41 | 152 | 146 | 487249 / 106103 |
| 优化 RAG | 127 | 19 | 186 | 146 | 909425 / 169088 |

接口归属来自执行环境与任务记录的声明，原生响应没有记录 endpoint；返回同一模型名不证明后端权重相同。因此无法把净增 3 条完全解释为 RAG 的因果效果。Tokens 只覆盖保存的原始响应；未保存响应的尝试是否收费及供应商实际费率未知。完整试点/恢复成本见[汇总](../results/optimized_rag_summary.json)。

## 离线核验与使用

安装项目后运行：

```bash
python -m scripts.verify_optimized_rag
python -m pytest tests/test_optimized_rag.py -q
```

核验器检查文件哈希、被试角色分离、参考数值与方向计数、配对分类、试点成员、置信度分组和声明接口统计，并重建全部 146 条 RAG 上下文。这些上下文与经公开工具命名转换后的冻结 API 输入摘要逐条完全一致。只读取本地文件，不访问模型。

可导出一个查询的输入检查 LLM 证据：

```bash
python -m scripts.verify_optimized_rag --payload q001 --arm percentile_group
python -m scripts.verify_optimized_rag --payload q001 --arm without_rag
```

自行使用时，先用 `validate_bank(bank, fold)` 核验同折参考库，再调用 `context_from_bank(prediction, bank, provenance, variant="percentile_group")`。`prediction` 仅需要 MIL 读数，以及 `numeric_fusion` 容器里的 `reference_available=False` 和八个 `actual_tools`；容器名称不表示需提供融合类别或分数。新数据的参考库需在本地重新准备，不能把此实验的固定折与任意新被试混用。

## 公开证据范围

[匿名路径 CSV](../results/optimized_rag_paths.csv)保存真实最终类别、confidence、试点归属与声明接口。[数值输入](../results/optimized_rag_inputs.json)和[24 折参考库](../results/optimized_rag_banks.json)支持检索回放；[固定提示](../results/optimized_rag_protocol.json)、[文件及上下文哈希](../results/optimized_rag_manifest.json)和[源输出审计索引](../results/optimized_rag_source_audit.json)支持核对。

公开版把工具名称统一为仓库的 VRMSModel 命名，将被试按全局顺序替换为匿名 ID，移除私人路径与不参与检索的查询元数据。数值、参考类别、顺序和实际 LLM 最终分类保留。历史源代码、请求和响应 SHA256 指向原始本地字节；公开文件 SHA256 与变换后的上下文摘要另列，二者不能混用。

原始 EEG、问卷、权重、API 密钥、原生响应及私有推理未公开。公开数据能复算结果并回放检索，不能独立重新编码原始 EEG 或核验供应商后端；原生响应真实性的独立复查仍需受控访问本地源文件。发布前本地原始审计已通过，已有成功回答、试点记录和补跑前快照保持不变；本次发布没有新增模型调用。
