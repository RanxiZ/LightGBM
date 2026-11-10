# Ascend NPU：lightgbm_npu_npa0 复现说明

本分支固化 **精度逻辑集成完成、后续性能优化之前** 的 NPU 版本，源自历史 `npu_integrated_source`。它与 `lightgbm_npu_npa1` 性能优化版分开发布。
默认 `device=npu, gpu_use_dp=false` 使用 FP32 分块直方图、FP64 合并和 SIMT 完全并列候选 FP64 精修；`gpu_use_dp=true` 选择 FP64 路径。无需设置实验性 `LGBM_NPU_*` 开关。

## 环境要求与第一次安装

- Linux、Ascend **950PR / dav-3510**，匹配的驱动和 **CANN 9.1.0**。
- Python **3.12**、Git、C/C++ 编译器、磁盘空间和构建依赖的网络访问。
- Qlib CN 日线数据：`~/.qlib/qlib_data/cn_data`。数据不包含在 Git；不同数据版本不能保证相同指标与时间。
- Qlib 固定源码基线 `be338d3c2ee8f0aa462015e4ba888a5a7682a10e` 已随分支打包，安装前校验 SHA256，不依赖该提交能否从公网获取。

先安装一次，随后才能直接运行 qrun。仅克隆 LightGBM 不会自动安装驱动、原生库、Qlib 和数据。
以下命令适用于新的复现目录；已有 `/workspace/LightGBM` 时请使用其他克隆目录，避免覆盖已有工作。

```bash
git clone --branch lightgbm_npu_npa0 https://github.com/RanxiZ/LightGBM.git /workspace/LightGBM
cd /workspace/LightGBM
NPU_PYTHON=/usr/local/python3.12.13/bin/python \
CANN_ROOT=/usr/local/Ascend/cann-9.1.0 \
bash npu/setup.sh --download-data
```

若 Python 3.12 已为 `python3` 且 CANN 在默认位置，可简写为 `bash npu/setup.sh --download-data`。
已有数据时无需 `--download-data`。下载的数据可能与历史测试数据不同，请先核验指纹。

安装脚本将：

1. 解压和安装固定 Qlib 源码快照，安装 `requirements-reproduce.txt` 的核心依赖。
2. 初始化 LightGBM 子模块，使用 `USE_NPU=ON` 构建，再安装预编译原生库的 Python 包。
3. 在 Qlib 源码和实际导入位置恢复三个**优化前原始模块**，不引入 npa1 的索引、行情与延迟导入优化。
4. 安装 Alpha158 YAML：`device=npu`、`gpu_use_dp=false`、`max_depth=-1`、`num_threads=20`，保留 SignalRecord、SigAnaRecord、PortAnaRecord。
5. 安装 qrun 包装入口，自动加载 CANN、选择安装时的 Python，未设置时使用 `PYTHONHASHSEED=0`，调用原 Qlib CLI。
6. 校验实际原生库、配置和模块，验证干净环境下的 `qrun --help`，执行默认与 FP64 小训练。

脚本拒绝覆盖不匹配的 Qlib 自定义修改。若现有环境已安装 npa1，建议使用独立 Python 环境和 Qlib 目录：

```bash
NPU_PYTHON=/path/to/venv/bin/python bash npu/setup.sh \
  --qlib-dir /workspace/qlib-npa0 --qrun-bin-dir /path/to/venv/bin
```

此时应在 `/workspace/qlib-npa0/examples` 运行。要使用用户要求的 `/workspace/qlib/examples`，首次安装使用默认目录即可。
`--skip-python-install` 仅适用于依赖完全匹配且已有对应 Python 包的环境；`--skip-build` 仅复用当前检出版本已构建的 NPU 库。

## 平时直接运行

完成首次安装后，在普通终端执行：

```bash
cd /workspace/qlib/examples
qrun benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml
```

无需手动 source CANN 或设置额外 Python 环境变量。同名 qrun 若被其他目录遮蔽，应调整 PATH 或使用安装报告给出的绝对入口。

## 历史结果与精度边界

精度集成版历史完整 qrun **106.86790183093399 秒**，其中训练 **61.3867675550282 秒**，best_iteration=13。
这是完整 SignalRecord / SigAnaRecord / PortAnaRecord 的单次观测，不能保证其他机器耗时相同。
更早精度候选的 105.23 秒不是本分支集成版的同一次计时。

独立精度回归的测试集指标：

| MSE | IC | RankIC | ICIR | RankICIR |
| --- | --- | --- | --- | --- |
| 0.004563398847794972 | 0.04306053455317936 | 0.04533504235969814 | 0.35601199261799293 | 0.3850378400331159 |

- 默认混合精度路径保留历史跨次浮点波动，不能承诺每次预测或聚合指标逐位一致，固定哈希种子也不能消除所有设备累加差异。
- FP64 集成回归相对既有 NPU FP64 参考，valid/test 最大预测差均为 0。
- 不声称与 CUDA 全部严格一致。六次 CUDA 观测范围不是统计置信区间；部分聚合指标越界，回测持仓与收益也未完成 CUDA 全量一致性验收。
- 本分支发布只整理源码、安装与复现材料，**没有覆盖或安装到 HiDeVLab 当前版本，也没有为本次分支发布重新进行硬件训练**。
- 本次保留的是历史源码和历史证据；新的路径构建可能产生不同库 SHA，不能把历史二进制 SHA 当作重建必须相等的值。

历史证据：`reference/integrated_qrun.json`、`integration_validation.json`、`cross_acceptance.json`、`clean_candidate.json`。
`integrated_source_manifest.json` 记录导出时 610 个原始跟踪文件的 SHA256；发布所需的 README 和 .gitignore 单独调整，NPU 实现与历史源码逐文件相同。`publication_validation.json` 记录本次本地核验范围。

## 数据与结果核验

使用与安装相同的 Python，并先加载 CANN：

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
python /workspace/LightGBM/npu/verify_reference.py
python /workspace/LightGBM/npu/verify_reference.py \
  --artifacts /workspace/qlib/examples/mlruns/EXPERIMENT/RUN/artifacts
```

输入与 `reference/input_summary.json` 指纹不符时停止。模型核验检查 NPU 参数、best_iteration、保存信号与模型预测一致性，并输出十项指标相对历史单次结果的差值。
默认指标差值仅作诊断，不宣称精度验收通过；增加 `--strict-metrics` 可要求与历史单次结果差值不超过 1e-10，但旧版跨次波动可能导致失败。
仅加载自己生成且可信的 pickle 模型。

## 备份与恢复

安装覆盖项保存在 `npu/_state/YYYYMMDD_HHMMSS_PID/`。发生覆盖阶段失败会恢复文件；也可手动执行：

```bash
python /workspace/LightGBM/npu/install.py --rollback /workspace/LightGBM/npu/_state/STATE_DIRECTORY
```

恢复范围为原生库、被覆盖的 Qlib 模块、YAML 和 qrun 入口；不包括 pip 依赖、源码目录或下载数据。建议独立 Python 环境。
QLib 配套源码遵循 MIT，见 `QLIB_LICENSE`。库、原始数据、MLflow 模型与安装备份不提交 Git。

安装工具本地测试：

```bash
python -m unittest discover -s npu/tests -v
```
