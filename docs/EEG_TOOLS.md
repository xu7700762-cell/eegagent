# 当前 EEG 工具清单

本清单对应当前 `eeg_agent` / `vrms_model` 版本，包含三个领域 Agent 的工具。按已注册的不同接口名称统计，共 **17 个**：4 个领域入口、4 个共享基础工具、VRMS 判断内部的 8 个证据工具与 1 个聚合接口。不同层级存在包含关系，不能解释为 17 路独立 EEG 证据。

三个领域 Agent 的领域入口共 **4 个**：

| Agent | 工具名 | 作用 |
| --- | --- | --- |
| VRMSAgent | `vrms.raw_recording` | 原始事件、完整与未完成路径、休息段和时间区间统计。 |
| VRMSAgent | `vrms.raw_model` | 重新编码原始 EEG，运行 MIL、8 个证据工具与 GPT 路径判断。 |
| FatigueAgent | `fatigue.raw_spectrum` | 计算全头皮 θ/α（TAR），分别汇总路径与休息段指标，并返回逐段值、首末变化及最高片段和时间。 |
| EmotionAgent | `emotion.raw_workload` | 计算额区 θ/顶区 α 工作负荷指标及额区 alpha 不对称性（FAA），分别返回路径与休息段汇总、逐段工作负荷值、首末变化及最高片段和时间。 |

FatigueAgent 与 EmotionAgent 各有 **1 个已注册 EEG 测量工具**。同一工具返回多个指标和时段比较，不能按输出字段再增加工具数。定义见 [domain_tools.py](../eeg_agent/domain_tools.py)，实际计算见 [recordings.py](../eeg_agent/recordings.py)。

三个 Agent 在原始 EEG 会话中还共享以下 **4 个基础工具**；它们由 Agent 自主规划，但由本地控制器强制执行加载、预处理和质量检查顺序：

| 工具名 | 作用 |
| --- | --- |
| `EEGFileLoader` | 在受限本地目录加载选定记录，返回匿名格式、通道、采样率、事件和完整性摘要。 |
| `EEGPreprocessor` | 按冻结输入合同执行重参考、50 Hz 陷波、带通、低通、重采样、断流预热和五秒分窗。 |
| `EEGQualityAssessor` | 以五秒窗口聚合逐导联有限值、峰峰值、标准差、稳健振幅离群和中位参考相关性；区分持续坏导联与短暂/共享伪迹，并统计窗口覆盖率。默认需在至少 50% 可用窗口持续的硬故障才进入 `blocking_bad_channels`。 |
| `EEGChannelRepair` | 对明确坏导联执行 `drop` 删除或 `interpolate` 邻近导联均值插值，然后重新检查质量和模型合同。 |

共享基础工具只返回本地计算的摘要。原始波形、路径、标签和被试身份不会进入云端。质量状态为 `review` 时，VRMSModel 分类会被本地质量门禁止，描述性指标仍可运行；中位参考相关性只作为软性警告，不会把正常的频率差异误判成硬故障。`EEGChannelRepair` 的插值供体来自固定邻接表；没有足够明确供体时拒绝插值，不使用任意通道平均。

`vrms.raw_model` 内部的 GPT 判断可调用以下 **8 个 EEG 证据工具**：

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

因此，VRMS 内部 GPT 函数共 **9 个**；三个领域入口 **4 个**、共享基础工具 **4 个**，全项目按唯一接口名称合计 **17 个**。在原始 EEG 会话中，Agent 实际可见的是“本领域入口 + 共享基础工具”，而共享工具结果只以本地摘要进入云端。
