# 冻结 FEMBA＋MIL 的 GPT 二次分类实验

这是用户单独要求的探索实验。生产 V2 Supervisor 继续只让 LLM 解释证据；本实验使用独立模块和输出目录，让 GPT 直接返回 high/low。结果不能替代 Reliability V3 的发布门控。

固定比较同一批 146 条完整路径、24 个 LOSO 折。原基线为原始 FEMBA＋MIL 分数 >=0.5 判 high；每次 API 请求只有一个匿名查询，隐藏查询问卷值、真实类别、被试和路径标识。使用 `store=false` 的独立 Responses 请求，不传对话历史。

GPT 可读取同折模型原始未校准分数、有效完整窗口比例、客观频谱/协方差变化，以及 5 名 meta 被试各自最近的一条路径。距离采用冻结 FEMBA 窗口特征的整路径均值，以原 MIL 的 base14 均值/标准差标准化、裁剪后计算 RMS 距离。按距离排序，不按类别配额；检索可靠范围尚未验证。

所有例子均来自同折 meta5；这些被试没有参与该 MIL 分类头拟合。例子的观察类别可以发送，查询真实标签不能发送。缺少参考不作 low 证据，频谱与无方向协方差距离不作已验证的方向性证据。

提示词、输出 schema、输入请求和模型来源在调用前冻结并保存哈希。不训练新模型、不放宽门控、不对外层标签调整提示词或融合权重。GPT 的离散类别直接作为二次分类结果，定性 confidence 不是校准概率。

全部请求失败时回退原模型类别；同时报告调用成功样本的比较和全 146 条含回退的比较。失败日志保留，恢复请求另存。原模型、V2/V3 结果与生产代码均不覆盖。

准备在已有科学 Python 环境中运行：

```bash
python -m vrms_cloud.second_judgment --prepare
python -m unittest vrms_cloud.test_second_judgment -v
```

设置已有项目显式 API 环境变量后，以下命令实际调用 GPT：

```bash
python -m vrms_cloud.second_judgment --probe
python -m vrms_cloud.second_judgment --workers 4
python -m vrms_cloud.second_judgment --recover --workers 4
```

本实验复用既有生理缓存，属于离线二次判断，不是懒执行部署或延时基准。原 FEMBA 预训练清单缺失，预训练数据与下游被试重叠无法排除。单种子、已被查看过的数据和固定提示词的一次实验只能提供探索证据。

## 2026-10-08 实测

真实 `gpt-6.1-sol`：146/146 成功；共 150 次尝试，4 次连接断开后同请求重试成功。原模型 ACC=60.96%（89/146）、BACC=61.15%；二次判断 ACC=63.01%（92/146）、BACC=62.89%。27 条改判中，纠正 15 条、误改 12 条。

被试级配对 bootstrap 的 ACC 差值 95% 区间为 -5.96～+11.19 个百分点，跨过 0。事后计算的同 5 案例多数投票也为 92/146；不能把当前微小差值认定为稳定的 LLM 贡献。结果保存于 `outputs/vrms_second_judgment/femba_gpt_20261008`，桌面 baogao 内有中文总结、逐条 CSV 和全部真实回复。

离线评分与输入/回复审计：

```bash
python -m vrms_cloud.evaluate_second_judgment
```

接口依据：[OpenAI Docs — Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)。
