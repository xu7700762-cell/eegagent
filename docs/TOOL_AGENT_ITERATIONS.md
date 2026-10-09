# 锁定 FEMBA＋MIL 的工具 Agent 迭代协议

用户验收目标：原基线锁定 **60.96%（89/146）**；真实 GPT 二次判断的最终 ACC 至少 70%，纠正数至少为误改数两倍；同 146 条完整路径、24 折 LOSO，最终类别只能是 High/Low。

每个外层测试被试均不能参加该折工具拟合、工具参数选择、纠正学习器或门槛选择。请求只有一条匿名查询。既有数据已被反复检查，本轮是持续开发中的探索性 LOSO，不能等同于未查看过的独立外部验证。

## 数值证据与动态执行

保留原 14 名 base 被试训练的 FEMBA＋MIL 权重和 0.5 判定门槛。本轮辅助分类器利用冻结 FEMBA 特征、频谱、协方差、路径内变化和滤波器组 CSP。其 meta5 被试 OOF：每次训练 base14＋另外4名 meta，预测被排除的第5名 meta；最终工具拟合 base14＋meta5，共19名被试。

工具参数只按 meta OOF 结果选择；校准拟合 OOF 分数，不使用已拟合头的自身预测作校准验证。policy4 独立用于比较原始与校准判定、纠正和误改。同一折的外层标签没有参与上述选择。

辅助工具使用19名被试，原固定基线使用14名。该数据量差异必须报告；提升不能单独归因于 GPT 或预训练。保留不调用 GPT 的各工具及组合对照。

真实 GPT Responses `function_call` 选择工具；收到调用后才计算当前查询的频谱、协方差及 CSP，再把 `function_call_output` 发回 GPT。执行复用已有 `ToolReplay.tool` 的缓存和计时。共享冻结编码器特征可以缓存，未请求的替代分类头、当前查询频谱及空间计算不会提前运行。

第三轮增加纠正学习器：输入原模型及8路工具证据，在 meta5 OOF 与 policy4 独立预测上拟合；参数和改判 margin 通过 policy 被试 OOF 选择。额外做嵌套 policy 被试验证，内层选择参数/margin，外层被试仅审计。选择 OOF 和完整嵌套验证分开报告；未通过的折不能称为可靠发布策略。最终仍强制 High/Low，证据弱时只降低定性置信度。

## 已完成实验

| 实验 | 最终 ACC | 纠正 | 误改 | 是否达到用户目标 |
|---|---:|---:|---:|---|
| 原固定 FEMBA＋MIL | 60.96% | — | — | 基线 |
| 原先离线打包证据＋GPT | 63.01% | 15 | 12 | 否 |
| 第一轮真实动态工具 Agent | 61.64% | 17 | 16 | 否 |
| 第二轮扩展工具＋方向支持 Agent | 63.70% | 21 | 17 | 否 |
| 第三轮纠正学习器＋GPT | 69.18% | 21 | 9 | ACC未达到70% |
| 第四轮证据优先＋GPT | **71.92%（105/146）** | **32** | **16** | 达到：ACC≥70%，纠正/误改=2.00 |

第二轮最终146条都有实际 GPT 分类。6条原失败记录另存补测；其中2条最后仅因 `spectral / spatial_covariance` 等组合名称引用被拒，严格拆为已执行名称后可解析，分类与解释完全没有改动。该修复使最终正确数从95降至93，不能只保留更好看的回退结果。

纯数值对照：内部选定 margin 的组合分类器为70.55%（103/146，纠正25、误改11）；未加该 margin 的原组合模型为73.29%（107/146，纠正32、误改14）；只允许完整嵌套审核通过的折改判为67.81%（99/146，纠正15、误改5）。这些均不是 GPT 的最终结果。

第四轮146条均有实际 GPT 最终回复，回退0条、uncertain=0。相对固定基线增加16条正确路径，即10.96个百分点。GPT实际请求纠正学习器，由该工具执行8路EEG证据并返回融合结果；本轮全部路径都执行相同8路工具，不能据此宣称自适应减少了工具成本。297轮成功响应、306次实际API尝试中有9次传输失败，失败均在同一查询内恢复，全部记录保留。

第四轮纯数值融合比GPT多对2条；提升不能单独归因于GPT。按被试配对bootstrap，ACC差值95%区间为[-0.71,+23.75]个百分点，跨过0。本轮达到用户设定的数值标准，仍是反复开发数据上的探索性结果；严格可靠发布和独立泛化尚未证明。

离线审计入口为 `python -m vrms_cloud.audit_tool_agent_v4`：独立重算CSV统计，检查冻结源码、原概率、权重、EEG缓存和划分哈希，核对24折的角色隔离，并重新执行146条查询所有已调用工具与保存的API证据逐项比较。审计不调用API，不修改GPT分类。结果保存在第四轮目录的 `completion_audit.json`。

运行实际Agent：配置本项目 `EEG_GPT_BASE_URL/MODEL/KEY` 后执行 `python -m vrms_cloud.tool_agent_v4`；已有记录会复用，协议变化会被拒绝。随后运行 `python -m vrms_cloud.evaluate_tool_agent --out outputs/vrms_agent_loso/v7_gpt_r1_20261008`。新实验应使用独立目录，不能覆盖历史响应。

## 运行位置

- 第一轮：`outputs/vrms_agent_loso/v4_gpt_r1_20261008`
- 第二轮：`outputs/vrms_agent_loso/v5_gpt_r1_20261008`
- 第三轮：`outputs/vrms_agent_loso/v6_gpt_r1_20261008`
- 第四轮：`outputs/vrms_agent_loso/v7_gpt_r1_20261008`
- 原始工具：`outputs/vrms_agent_loso/v4_tools_r1_20261008`
- 扩展工具：`outputs/vrms_agent_loso/v5_tools_portable_20261008`
- 纠正学习器及嵌套审核：`outputs/vrms_agent_loso/v6_selector_20261008`

原 FEMBA 预训练清单缺失，预训练数据与下游被试重叠不能排除。原始 API 日志、重试、引用修复、每轮输出和不利结果全部保留。报告与逐条结果复制到桌面 baogao。

随后指定 seed2027、seed2028，保持预训练特征、划分、训练配方、工具及提示词，重新训练下游 MIL/工具/融合并重新调用 GPT。GPT 分别为 69.18%（101/146）和 71.92%（105/146）；相对各自 MIL 的纠正/误改为 35/18、28/11。三种子全部结果及代码协作关系见 [GPT_MULTI_AGENT](GPT_MULTI_AGENT.md)。原固定基线的历史统计与新 seed 自身基线分别报告。

接口依据：[OpenAI Docs — Function calling](https://developers.openai.com/api/docs/guides/function-calling)、[Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)。
