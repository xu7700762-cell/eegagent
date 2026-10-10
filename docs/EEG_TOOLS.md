# 当前 EEG 工具清单

本清单对应当前 `eeg_agent` / `vrms_model` 版本。VRMS 判断使用 **8 个 EEG 证据工具**；加上 `corrective_evidence` 聚合接口，GPT 可调用的函数共 **9 个**。

| 工具名 | 作用 |
| --- | --- |
| `vrms_features` | 使用冻结 VRMSModel 编码特征的路径均值和标准差进行分类。 |
| `spectral` | 从实际 EEG 窗口计算频谱及路径内变化，返回训练好的频谱分类器证据。 |
| `spatial_covariance` | 从实际 EEG 窗口计算归一化空间协方差，返回空间分类器证据。 |
| `case_retrieval` | 检索不同内部被试的相似训练案例，返回距离、类别计数和描述性投票。 |
| `vrms_mean` | 使用冻结 VRMSModel 编码特征的路径均值进行分类。 |
| `vrms_temporal` | 使用编码特征的路径均值、标准差和后段减前段变化进行分类。 |
| `spectral_compact` | 使用紧凑脑区频谱比例、频带间相对功率和路径内变化进行分类。 |
| `filterbank_csp` | 使用 delta/theta/alpha/beta 频带协方差及内部训练的 CSP 空间滤波器进行分类。 |

注册表见 [raw_tool_agent.py](../eeg_agent/raw_tool_agent.py)，数值计算见 [numeric.py](../vrms_model/numeric.py)。工具来自共享波形与特征，证据存在相关性。

`corrective_evidence` 聚合 MIL 与上述 8 个工具的证据，返回已训练的组合结果及各工具读数。它不增加新的 EEG 证据通道。当前版本没有 `combined_evidence` 调用接口。

MIL 是单独的路径基线，最终 High/Low 由 LLM 返回。优化 RAG 的案例选择见 [decisive_rag.py](../eeg_agent/decisive_rag.py)，不另算一个 EEG 分类工具。

网页 Supervisor 另外调度 4 个领域入口，它们与上述 9 个 GPT 函数处于不同调用层级：

| 领域入口 | 作用 |
| --- | --- |
| `vrms.raw_recording` | 原始事件、完整路径与时间区间统计。 |
| `vrms.raw_model` | 重新编码原始 EEG，运行 MIL、8 个证据工具与 GPT 路径判断。 |
| `fatigue.raw_spectrum` | 完整路径与休息段的 θ/α 指标及变化。 |
| `emotion.raw_workload` | 额区 θ/顶区 α 工作负荷指标及 FAA。 |

领域入口定义见 [domain_tools.py](../eeg_agent/domain_tools.py)。
