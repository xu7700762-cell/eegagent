# 当前入口复现

## 安装与本地配置

```bash
python -m pip install -e .
```

Python 至少 3.10。描述性工具和网页可在 CPU 上运行。冻结 VRMSModel 工作进程需要兼容的 NVIDIA CUDA、PyTorch 和原生 `mamba-ssm`；本次模型核对使用 Python 3.10.20、PyTorch 2.11.0+cu128、mamba-ssm 2.3.1、scikit-learn 1.7.2、joblib 1.5.3。私有估计器最初在 scikit-learn 1.6.1 保存；迁移到本次环境后做了逐路径数值核对。不同环境须重新核对，不能仅凭成功反序列化宣称等价。

将 `.env.example` 复制为 `.env`，填写 `EEG_API_BASE_URL`、`EEG_API_MODEL` 和 `EEG_API_KEY`。也可启动网页后在 `/settings` 保存。密钥不会由 API 返回，配置不会读取其他应用的账号。

`configs/local.yaml` 中的 `data_root` 目录应包含：

```text
private_data/
  data/raw/Acquisition 01.cdt
  data/raw/Acquisition 01.cdt.dpo
  ...
  data/labels/task_segments_with_path_scores.csv
```

分析只读取波形及 DPO；全路径评价额外需要原标签表的边界、完整性、评分可用标记及评分列。CDT 事件从原始 Trigger 读取，不读取 CEO。

## 冻结资产

权重不随仓库发布。数据持有者将已有 seed2026 资产导出到 `private_assets/seed2026/`。清单由编码器和 24 个留一折组成，每折包含 `head`、`normalization`、`first`、`second`、`selector`、`nested_audit`；每个文件均须有 SHA256。示例见 [模型清单](../configs/model_manifest.example.json)。

提供私有导入 JSON，列出已有资产源文件，必要时用 `string_aliases` 指定原工具名到当前工具名的映射、`module_aliases` 指定序列化模块到 `vrms_model.features` 的映射。只导入自己持有且可信的 torch/joblib 文件。导出不训练、不读取查询数据：

```bash
python -m scripts.export_model_assets --import-manifest /path/to/private_import.json --out private_assets/seed2026
```

私有导入结构见 [导入示例](../configs/model_import.example.json)。清单中的路径相对清单文件解析；编码器必须完整加载 83 个张量。运行时核验资产哈希及留出被试隔离。

Linux 直接配置 CUDA Python 为 `worker_command`。Windows 可设置本地私有 YAML：

```yaml
worker_command: [wsl.exe, -d, YOUR_CUDA_DISTRO]
worker_python: /path/to/cuda/environment/bin/python
model_manifest: ./private_assets/seed2026/manifest.json
data_root: ./private_data
output_root: ./outputs
seed: 2026
```

WSL 工作进程须能访问项目和资产所在盘。`EEG_DATA_ROOT`、`EEG_MODEL_MANIFEST` 可以覆盖配置；从准备到评分须保持一致。

## 网页

```bash
python -m eeg_agent --config configs/local.yaml serve --host 127.0.0.1 --port 8880
```

打开 <http://127.0.0.1:8880>，选择三个领域并输入“请分析被试 10 的原始 EEG”。上传 EDF 后也可比较路径或休息段指标。兼容性、窗口规则和 GPT 输出状态会在报告中显示。首次高类定位使用整路径结果，其时间解释见[协议](PROTOCOL.md)。

## 全路径评价

保持源代码、配置和资产冻结，对新输出目录依次执行：

```bash
python -m scripts.evaluate_raw_gpt_all --config configs/local.yaml --stage prepare --out outputs/new_raw_evaluation
python -m scripts.evaluate_raw_gpt_all --config configs/local.yaml --stage predict --out outputs/new_raw_evaluation
python -m scripts.evaluate_raw_gpt_all --config configs/local.yaml --stage score --out outputs/new_raw_evaluation
```

准备阶段锁定 146 条路径并记录哈希；预测阶段不载入评分值，重新计算波形证据并实际请求 GPT；评分阶段输出 `summary.json`、`path_results.csv` 和 `report.md`，同时计算 GPT 与 VRMSModel ACC。网页“实验评价”调用相同脚本。

只有全部首次预测完成后才能补跑 GPT 失败分类，原输出保持原样：

```bash
python -m scripts.recover_raw_gpt_failures --config configs/local.yaml --source outputs/new_raw_evaluation --out outputs/recovered_raw_evaluation
python -m scripts.evaluate_raw_gpt_all --config configs/local.yaml --stage score --out outputs/recovered_raw_evaluation
```

补跑必须继续使用原冻结代码、配置、清单和波形证据。补跑后的评分读取原输出复制的协议，原有有效决策逐项核对，不能重试已判错的有效分类。网页仅为符合这些条件的记录显示补跑按钮。

公开的 `current_seed2026_paths.csv` 去除了被试编号、时间、原始波形和 API 文本，只保留匿名逐路径真实类别、两种决策及本地证据摘要哈希。它可复算主表，完整波形复现仍需私有原始文件、评分表、冻结参数及云端接口。

## 验证

```bash
python -m pip install -e ".[test]"
python -m pytest -q
node --check eeg_agent/web/brain.js
node --test tests/web_contracts.cjs
python -m scripts.verify_release
```

自动测试验证数据隔离、事件边界、真实工具调用、失败不替代 GPT、报告事实和评价配置传播。测试通过不等于新一轮 ACC 已完成；结果记录明确区分已完成评价与打包验证。
