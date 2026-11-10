# Ascend NPU：lightgbm_npu_npa1 复现说明

本分支包含完整 Ascend NPU 后端，以及 2026-11-10 验证的精度/耗时平衡版本。
默认是 **FP32 块内累加 + FP64 合并/复核的混合精度路径**；`gpu_use_dp=true` 为单独的 FP64 路径。

## 环境要求

- Linux、Ascend **950PR / dav-3510**，已安装匹配驱动及 **CANN 9.1.0**。
- Python **3.12**、Git、C/C++ 编译器、足够的磁盘空间，以及网络访问 GitHub/GitLab/PyPI 的能力。
- 固定 Qlib 源码基线：`be338d3c2ee8f0aa462015e4ba888a5a7682a10e`。
- Qlib CN 日线数据位于 `~/.qlib/qlib_data/cn_data`。数据不放入 Git；下载的数据若与参考指纹不同，指标和时间也可能不同。

这是指定 Ascend/CANN 环境上的复现方案，不是对所有型号 NPU、CPU-only 机器或任意数据版本的兼容承诺。

## 第一次安装

```bash
git clone --branch lightgbm_npu_npa1 https://github.com/RanxiZ/LightGBM.git /workspace/LightGBM
cd /workspace/LightGBM
NPU_PYTHON=/usr/local/python3.12.13/bin/python \
CANN_ROOT=/usr/local/Ascend/cann-9.1.0 \
bash npu/setup.sh --download-data
```

如果 Python 3.12 已是 `python3`，CANN 在默认路径，可简写：

```bash
bash npu/setup.sh --download-data
```

安装脚本会：

1. 在 `/workspace/qlib` 解压随分支提供的固定 Qlib 源码快照（先核验 SHA256）；已有匹配提交或快照则复用，不重置用户仓库。
2. 安装固定的核心 Python 依赖和 Qlib，初始化构建需要的 LightGBM 子模块。源码快照不依赖不可公开获取的本地 Git 提交，使用版本号 0.9.8 构建。
3. 用 `USE_NPU=ON` 构建原生库，并通过 `build-python.sh install --precompile` 安装 Python 包，避免误装 CPU 版。
4. 将三个 Qlib 优化模块应用到源码位置和**实际导入位置**，保留其他模块。
5. 安装 Alpha158 YAML：`device=npu`、`gpu_use_dp=false`、`max_depth=-1`、`num_threads=20`。
6. 在所选 Python 的命令目录安装 `qrun` 包装；可写时也安装到 `/usr/local/bin`。包装只加载 CANN、选择所选 Python 和默认哈希种子，然后调用原来的 Qlib CLI，不替换训练逻辑。
7. 验证实际库 SHA、Qlib 模块、无预先加载 CANN 环境时的 `qrun --help`、默认和 FP64 NPU 小训练。

有其他同名 `qrun` 排在 PATH 前面时，使用 `--qrun-bin-dir` 指定命令目录，或调整 PATH。安装结果列出所有入口。

已有相同依赖且只需构建、应用配置时：

```bash
NPU_PYTHON=/usr/local/python3.12.13/bin/python bash npu/setup.sh --skip-python-install
```

`--skip-build` 只用于复用当前检出的已构建 NPU 库；不能拿 CPU 库代替。

## 平时直接运行

第一次安装完成后，在新的普通终端运行：

```bash
cd /workspace/qlib/examples
qrun benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml
```

无需手工执行 CANN `source` 或额外设置 Python 环境变量。`qrun` 包装会自动准备运行环境；`PYTHONHASHSEED` 未设置时默认为 0。
工作流完整保留 SignalRecord、SigAnaRecord、PortAnaRecord。

## 输入和结果核验

对照参考数据指纹：

```bash
python /workspace/LightGBM/npu/verify_reference.py
```

核对自己生成的可信 MLflow 模型、保存信号和十项精确指标：

```bash
python /workspace/LightGBM/npu/verify_reference.py \
  --artifacts /workspace/qlib/examples/mlruns/EXPERIMENT/RUN/artifacts
```

上面的核验命令应在相同 Python 环境且已加载 CANN 的 shell 中运行。
`npu/reference/input_summary.json` 固定数据、标签、索引和列指纹。输入不一致时会明确停止，不把新数据结果冒充参考结果。
结果核验只读取用户自己生成的可信 pickle 模型。

## 已验证的结果与边界

完整正式 qrun 五次：**70.2478、72.7084、71.5449、71.9724、70.2870 秒**；
中位数 **71.5449 秒**。环境版本、逐次指标、FP64 回归和回测严格等价证据在 `npu/reference/`。
时间依赖硬件、系统负载和数据版本，不能保证其他机器也恰好为该数值。

- 默认：五次实际模型和预测与选定稳定参考一致，best_iteration=13，evaluated_iterations=63。
- FP64：对既有 NPU FP64 基线预测逐位一致，树结构一致；不是重新取得 A100 原始数组的全量配对。
- Qlib 回测：五次报告、871 天持仓和订单指标与隔离参考严格相等。
- 严格 CUDA 对齐**尚未全部闭合**：三项指标在六次范围内，七项略向更好方向越界；42 个样本边界和 11 天 Top50 边界尚无原始 CUDA 数组补证。原始 A100 临时文件已缺失。
- 这是按精度优先原则选择的平衡版；约 66 秒目标尚未达到。

## 安装记录与恢复

每次应用覆盖前，脚本将原生库、被覆盖的 Qlib 文件、工作流和 qrun 入口保存到：

```
npu/_state/YYYYMMDD_HHMMSS_PID/
```

恢复这些覆盖项：

```bash
python /workspace/LightGBM/npu/install.py --rollback /workspace/LightGBM/npu/_state/STATE_DIRECTORY
```

失败会自动恢复这些文件。pip 依赖安装、新克隆的仓库和下载的数据不属于文件覆盖回滚范围，建议在独立 Python 环境安装。
构建目录、安装状态、预编译 `.so`、MLflow 模型和原始数据不提交到 Git。
Qlib 配套代码沿用其 MIT 许可证，见 `QLIB_LICENSE`。

## 安装工具测试

无需 NPU 的备份、恢复和入口生成测试：

```bash
python -m unittest discover -s npu/tests -v
```

原生算子测试需先完成 NPU 构建：

```bash
./npu_histogram_probe stable_fp32
```

完整复现还必须运行前述 qrun 和输入/指标核对，单元测试不代替 Ascend 硬件验证。

QLib 源码快照 `qlib_source.tar.gz` 由上述 Git 基线导出，仅包含 Git 跟踪文件，不包含用户未提交修改、原始数据、模型或访问令牌。完整性与来源见 `reference/qlib_source.json`。

发布前除既有依赖环境测试外，还新建了 Python 环境（共享匹配的系统依赖），实际执行 Qlib 源码快照构建、LightGBM Python 包安装和无预载 CANN 环境的完整 qrun；输入指纹与十项指标核对通过。结果见 `reference/fresh_install_validation.json`。
