# LightGBM / XGBoost 昇腾 NPU 迁移进展

工作汇报 · 截至 2026-10-19 · 已有运行证据汇总 · 本次未启动设备任务

**两模型已在完整 Alpha158 工作负载打通训练、预测、信号分析、回测与产物验证。当前阶段成果是关键训练链路迁移和端到端功能闭环；主机保留部分逻辑，尚非全功能、全设备实现。**

## LightGBM · 指定汇报基准

**3m40.740 / 220.740 秒**

采用 timing-histogram 运行；训练耗时 175.935 秒；74/74 项产物检查通过。开启融合直方图与融合分裂，最终保留 23 棵树。

IC 0.047030 · 扣费后超额年化 11.0624% · 信息比率 1.305132

## XGBoost · CUDA 逻辑对齐 v1

**5m36.637 / 336.637 秒**

采用 final-alpha158-npu-cuda-alignment-20261019-v1；qrun 与原生 NPU 后端验收通过。最终 73 棵树，预测按既有 Qlib 逻辑使用全部树。

IC 0.060891 · 扣费后超额年化 7.1286% · 信息比率 0.805570

> **说明：**汇报口径已修正：LightGBM 主结果仅用 220.740 秒那次数据及其 175.935 秒训练耗时。其他运行的 243.706 秒总耗时、198.209 秒训练耗时均不作为本报告主结果。qrun 是 Qlib 的端到端流程，包含数据准备、训练、预测、信号分析及回测。

## 01 · 工作负载与主要结果

Alpha158 / 158 特征 / CSI300 股票池 / SH000300 基准；训练 2008-01-01—2014-12-31，验证 2015-01-01—2016-12-31。测试配置 2017-01-01—2020-08-01，实际预测覆盖 2017-01-03—2020-07-31，共 871 个交易日、261,207 条预测；未裁剪测试范围。

| 指标（单位 / 定义） | LightGBM · 3m40.740 那次 | XGBoost · 10-19 v1 |
| --- | ---: | ---: |
| 端到端 qrun（秒） | 220.740 | 336.637 |
| 已单独记录的训练耗时（秒） | 175.935 | 该结果未单独记录，不由墙钟推算 |
| IC（预测与收益相关系数） | 0.047030 | 0.060891 |
| Rank IC（秩相关系数） | 0.048694 | 0.051528 |
| ICIR / Rank ICIR（原始日志值） | 0.381641 / 0.405690 | 0.435843 / 0.411568 |
| 扣费后超额年化收益（%） | 11.0624 | 7.1286 |
| 扣费后超额信息比率 | 1.305132 | 0.805570 |
| 扣费后超额最大回撤（%） | −8.5821 | −10.3409 |
| 未扣费超额年化收益（%） | 14.7334 | 10.5224 |
| 未扣费超额信息比率 | 1.737630 | 1.188975 |
| 未扣费超额最大回撤（%） | −7.8346 | −9.5045 |
| 保存模型树数 / 最佳迭代 | 23 / 23（从 1 开始） | 73 / 22（从 0 开始） |

来源：LightGBM previous-histogram-artifacts.json / execution、qrun_log、required_mlflow_metrics；XGBoost comparison.json / runs.after。收益与回撤均为超额口径、百分比按原值乘 100；ICIR 是原始日志值，不另行年化。不同模型与预测树数语义不支持直接推导模型间加速比。

## 02 · 迁移完成情况与 CPU / NPU 边界

| 模块 | LightGBM | XGBoost |
| --- | --- | --- |
| 框架集成 | NPUTreeLearner 与 device=npu 路由；保留 Qlib 原始非量化 Alpha158 配置 | device=npu:0、tree_method=hist、grow_npu_hist 与原生预测路由；接入独立 Python 环境 |
| 数据准备与分箱 | 数据准备/初始分箱由主机组织，分箱列、梯度与叶行驻留设备 | CPU SketchOnDMatrix 生成切分点；NativeBinRowsKernel 在 NPU 编码分箱并缓存 |
| 目标函数与采样 | 梯度/Hessian 由主机生成并上传，保留 boosting 编排和采样控制 | 平方误差梯度、行采样和梯度量化在设备；初始截距与小特征集列采样仍在主机 |
| 直方图构建 | 自定义融合 AICPU 算子：FP32 梯度/Hessian 转 FP64 累加，融合原 gather/cast/scatter 链 | 自定义整数直方图内核，梯度量化和直方图统计在设备执行，复用不可变分箱缓存 |
| 数值分裂搜索 | 融合 AICPU 阈值扫描、增益计算和赢家选择，回传选中结果；类别路径存在主机兼容逻辑 | 设备批处理前缀、精确评分、候选选择与结果收尾；主机负责树生长调度及选中节点元数据 |
| 叶行划分与更新 | 设备 INT32 前缀和、scatter 划分驻留叶行，训练分数更新在设备；主机维护兼容布局 | 设备 gather / prefix / scatter 维护行成员和左右子节点；复用缓冲区与批量调度 |
| 模型预测 | 训练分数更新已用设备算子；一般模型推理仍在主机逐树遍历，不能称通用推理已迁移 | NativePredictKernel 执行普通 DMatrix 标量树预测，接入预测缓存及标准模型保存/加载 |
| 验证闭环 | 指定 220.740 秒运行通过 74 项产物检查；融合开关对照全量预测差为 0、树结构一致 | 原生训练/预测/保存加载和完整 qrun 通过；UBJ/JSON/raw 重载预测全量一致 |

来源：两模型原生树更新、运行时、预测源码及保存的验收产物。这里的“设备”含 NPU 板上 AICPU 和加速核心；不等于全部在向量核心执行。源码存在某能力不等于已通过全部配置的端到端验收。

## 03 · 算子规模与热点结论

| 独立性能采样统计 | LightGBM · 同融合库 | XGBoost · 10-19 v1 |
| --- | ---: | ---: |
| 合并 Core Type 后的 OP Type 数 | 18 类 | 28 类 |
| 原表“算子 × 核心”记录数 | 21 条 | 30 条 |
| 设备算子调用总次数 | 141,229 | 779,649 |
| 设备算子累计任务耗时（秒） | 150.923351 | 246.714615 |
| 采样中出现的自定义内核类型 | 2 类融合 AICPU 内核 | 9 类自定义内核 |
| 首要设备热点 | 融合直方图 91.615% | 整数直方图 58.692% + ScatterElementsV2 25.371% |

> **注意：**算子耗时来自独立 msprof 性能采样，均不是上述非采样 qrun 的阶段拆分。占比的分母是该采样的设备算子累计时长，不能用 150.923 / 220.740 或 246.715 / 336.637 当作设备利用率，也不能叠加主机 API 时长。LightGBM 的 AI_CPU 表示设备侧 AICPU，不是主机 CPU，更不是向量核心。

LightGBM 采样墙钟为 260.786 秒，clean_device=true；实际加载库 SHA256 与 220.740 秒基准相同，融合开关一致，但不是同次运行且 Qlib 加载路径不同。两份重复导出的算子统计 CSV 经 SHA256 比对字节相同，只计一份；旧库 341.929 秒算子统计未混入。

LightGBM 的设备算子时间主要落在 AICPU：150.589480 秒；向量核心与 MIX_AIV 合计 0.333871 秒。XGBoost 的 MIX_AIC 为 181.562613 秒、MIX_AIV 为 62.967935 秒、AI_VECTOR_CORE 为 2.184067 秒。以上为 profiler 原始分类，不把混合核心标签解释为某个物理核心独占执行。

## 04 · 自定义算子与职责

**LightGBM：融合直方图与融合数值分裂**

| 自定义算子（精确名称） | 职责 | 采样执行单元 | 调用次数 | 累计耗时（秒） |
| --- | --- | --- | ---: | ---: |
| LGBMNPUFusedHistogramKernel_interface | 融合特征取值、FP64 梯度/Hessian 直方图累加 | AI_CPU | 4,700 | 138.268880 |
| LGBMNPUFusedSplitKernel_interface | 数值阈值搜索、增益计算、最优分裂选择 | AI_CPU | 9,327 | 6.742906 |

**XGBoost：分箱、采样、目标函数、直方图、批量分裂、预测**

| 自定义算子（精确名称） | 职责 | 采样执行单元 | 调用次数 | 累计耗时（秒） |
| --- | --- | --- | ---: | ---: |
| IntegerHistogramBuildKernel | 量化梯度/Hessian 的整数直方图累加 | MIX_AIC | 8,381 | 144.801699 |
| BatchedSplitScoreKernel | 分裂候选精确评分 | MIX_AIC | 16,689 | 17.534225 |
| BatchedSplitSelectKernel | 最优候选选择 | MIX_AIC | 16,689 | 10.619963 |
| BatchedSplitPrefixKernel | 分裂候选前缀统计 | MIX_AIC | 16,689 | 6.806689 |
| BatchedSplitFinishKernel | 选中分裂摘要与结果收尾 | MIX_AIC | 16,689 | 1.003575 |
| NativePredictKernel | 原生树模型批量预测 | MIX_AIC | 147 | 0.692385 |
| NativeSampleRowsKernel | 均匀行采样及梯度掩码 | MIX_AIC | 73 | 0.047183 |
| NativeBinRowsKernel | 按已有切分点进行分箱编码 | MIX_AIC | 1 | 0.047007 |
| NativeSquareErrorKernel | 平方误差梯度/Hessian 计算 | MIX_AIC | 73 | 0.009888 |

来源：2026-10-19 独立 msprof op_statistic 与对应源码；仅统计采样实际出现的自定义 OP Type，不代表源码里全部内核数量。XGBoost 的 ScatterElementsV2 为通用算子，不纳入自定义内核数量。

## 05 · 已完成优化与验证

**LightGBM**

同库、原始工作负载单次开关对照：仅融合分裂 311.957 秒 → 融合直方图与分裂 220.740 秒，减少 91.218 秒（29.24%）。全测试集 261,207 条预测最大绝对差 0，树结构一致，最佳迭代均为 23。结果描述为本次对照观察，不泛化为相对 CPU / CUDA 的正式加速比。

**XGBoost**

完成分箱及分配缓存复用、批量分裂驱动、驻留行成员与预测缓存；本批对齐行/列采样、随机数推进、预测累加顺序及 depthwise 最大 1024 节点批大小。原生训练、预测、标准模型保存加载与完整 qrun 验证形成闭环。

10-17 v4 → 10-19 v1 的单次墙钟 342.667 → 336.637 秒，减少 1.76%，但树数 74 → 73；IC 上升 0.001750，扣费后超额年化下降 1.9897 个百分点。跨版本预测相关约 0.8960，不声称数值结果不变或收益无回退。

## 06 · 当前边界与待完成项

**LightGBM**

本次验收是非量化 Alpha158。公开 NPU 量化训练入口仍拒绝；单调约束、线性树、CEGB、强制分裂及分布式训练未作为已支持能力。类别分裂、refit 等存在主机兼容路径，通用推理仍在主机；生产验收标记仍为 false。

新采集流程在发现两个相同算子导出文件时断言中断，导致完整新旧对照未完成；本报告核对哈希后只读取一份已有 CSV，没有修改采集程序或重启实验。

**XGBoost**

范围限于单卡、内存内、数值特征、标量平方误差、gbtree；类别/分布式/外存/多目标/加权特征采样，以及原地预测、叶索引与 SHAP 贡献等不在当前原生支持范围。

切分点仍用 CPU SketchOnDMatrix，尚未迁移为 CUDA DeviceSketch 等价算法；初始截距归约、量化尺度边界与编译舍入仍有未对齐或未验证项。尚无冻结相同输入下的 CUDA 逐轮模型/预测等价证明。

## 07 · 下一阶段工作建议

优先级 1：围绕已验证热点优化。LightGBM 优先优化融合直方图的设备 AICPU 循环；向量化/并行化须保持 FP64 累加与预测一致性。XGBoost 优先优化整数直方图与行成员相关 scatter，二者占采样设备算子时间约 84.06%。

优先级 2：固定输入、参数、训练轮数与运行环境，在独占设备上重复计时，补充分位数和方差；LightGBM 对重复导出文件去重后补齐完整前后算子比较。

优先级 3：XGBoost 补齐切分点、初始截距、量化边界与 CUDA 同输入逐轮对照；两模型分别扩展功能回归、打包部署和生产验收，避免把基准通过写成全功能迁移完成。

## 附录 A · LightGBM 全部采样算子

| OP Type | Core Type（合并显示） | 调用次数 | 累计（秒） | 设备算子时间占比 | 平均（微秒/次） |
| --- | --- | ---: | ---: | ---: | ---: |
| LGBMNPUFusedHistogramKernel_interface | AI_CPU | 4,700 | 138.268880 | 91.615% | 29418.911 |
| LGBMNPUFusedSplitKernel_interface | AI_CPU | 9,327 | 6.742906 | 4.468% | 722.945 |
| ScatterElements | AI_CPU | 6,441 | 2.260724 | 1.498% | 350.990 |
| Sub | AI_CPU / AI_VECTOR_CORE | 21,851 | 1.466243 | 0.972% | 67.102 |
| Cumsum | AI_CPU / AI_VECTOR_CORE / MIX_AIV | 10,856 | 1.063297 | 0.705% | 97.946 |
| ArgMax | AI_CPU | 6,441 | 0.558038 | 0.370% | 86.638 |
| BroadcastTo | AI_CPU | 6,441 | 0.331899 | 0.220% | 51.529 |
| GatherV2 | AI_VECTOR_CORE | 10,856 | 0.063361 | 0.042% | 5.836 |
| Cast | AI_VECTOR_CORE | 12,809 | 0.032720 | 0.022% | 2.554 |
| ScatterElementsV2 | MIX_AIV | 6,368 | 0.029203 | 0.019% | 4.586 |
| Equal | AI_VECTOR_CORE | 12,495 | 0.025706 | 0.017% | 2.057 |
| LogicalOr | AI_VECTOR_CORE | 10,491 | 0.019790 | 0.013% | 1.886 |
| SelectV2 | AI_VECTOR_CORE | 6,368 | 0.016606 | 0.011% | 2.608 |
| Add | AI_VECTOR_CORE | 6,368 | 0.016377 | 0.011% | 2.572 |
| LessEqual | AI_VECTOR_CORE | 6,368 | 0.015090 | 0.010% | 2.370 |
| TensorMove | AI_VECTOR_CORE | 577 | 0.006401 | 0.004% | 11.094 |
| LogicalNot | AI_VECTOR_CORE | 1,236 | 0.003332 | 0.002% | 2.696 |
| LogicalAnd | AI_VECTOR_CORE | 1,236 | 0.002779 | 0.002% | 2.248 |

来源：op_statistic_20261019071225.csv，同库独立采样，2026-10-19。按 OP Type 合并不同 Core Type，平均耗时按总微秒 / 总次数加权计算；所有占比分母为 150.923351239 秒。保留原始 CSV 以便追溯不同核心记录。

## 附录 B · XGBoost 全部采样算子

| OP Type | Core Type（合并显示） | 调用次数 | 累计（秒） | 设备算子时间占比 | 平均（微秒/次） |
| --- | --- | ---: | ---: | ---: | ---: |
| IntegerHistogramBuildKernel | MIX_AIC | 8,381 | 144.801699 | 58.692% | 17277.377 |
| ScatterElementsV2 | MIX_AIV | 38,847 | 62.593063 | 25.371% | 1611.271 |
| BatchedSplitScoreKernel | MIX_AIC | 16,689 | 17.534225 | 7.107% | 1050.646 |
| BatchedSplitSelectKernel | MIX_AIC | 16,689 | 10.619963 | 4.305% | 636.345 |
| BatchedSplitPrefixKernel | MIX_AIC | 16,689 | 6.806689 | 2.759% | 407.855 |
| BatchedSplitFinishKernel | MIX_AIC | 16,689 | 1.003575 | 0.407% | 60.134 |
| NativePredictKernel | MIX_AIC | 147 | 0.692385 | 0.281% | 4710.099 |
| Cumsum | AI_VECTOR_CORE / MIX_AIV | 60,786 | 0.474056 | 0.192% | 7.799 |
| GatherV2 | AI_VECTOR_CORE | 62,242 | 0.332969 | 0.135% | 5.350 |
| ReduceSum | MIX_AIV / AI_VECTOR_CORE | 74,938 | 0.307243 | 0.125% | 4.100 |
| AxpyV2 | AI_VECTOR_CORE | 95,332 | 0.304228 | 0.123% | 3.191 |
| Cast | AI_VECTOR_CORE | 86,890 | 0.282681 | 0.115% | 3.253 |
| TensorMove | AI_VECTOR_CORE | 11,257 | 0.178272 | 0.072% | 15.837 |
| SelectV2 | AI_VECTOR_CORE | 63,560 | 0.152728 | 0.062% | 2.403 |
| Greater | AI_VECTOR_CORE | 33,394 | 0.111089 | 0.045% | 3.327 |
| Less | AI_VECTOR_CORE | 48,481 | 0.109356 | 0.044% | 2.256 |
| Add | AI_VECTOR_CORE | 38,561 | 0.107979 | 0.044% | 2.800 |
| LogicalNot | AI_VECTOR_CORE | 28,611 | 0.057955 | 0.023% | 2.026 |
| NativeSampleRowsKernel | MIX_AIC | 73 | 0.047183 | 0.019% | 646.341 |
| NativeBinRowsKernel | MIX_AIC | 1 | 0.047007 | 0.019% | 47006.550 |
| LogicalOr | AI_VECTOR_CORE | 26,138 | 0.042799 | 0.017% | 1.637 |
| Equal | AI_VECTOR_CORE | 16,328 | 0.033660 | 0.014% | 2.061 |
| LogicalAnd | AI_VECTOR_CORE | 14,035 | 0.023509 | 0.010% | 1.675 |
| RightShift | AI_VECTOR_CORE | 2,190 | 0.019944 | 0.008% | 9.107 |
| BitwiseAnd | AI_VECTOR_CORE | 1,752 | 0.010564 | 0.004% | 6.030 |
| NativeSquareErrorKernel | MIX_AIC | 73 | 0.009888 | 0.004% | 135.456 |
| ClipByValueV2 | AI_VECTOR_CORE | 584 | 0.007471 | 0.003% | 12.793 |
| Mul | AI_VECTOR_CORE | 292 | 0.002436 | 0.001% | 8.341 |

来源：op_comparison.csv 的 after 列，原始 op_statistic_20261019053119.csv，2026-10-19 独立采样。按 OP Type 合并不同 Core Type，平均耗时按总微秒 / 总次数计算；所有占比分母为 246.714615286 秒。

## 原始证据与源码

静态工作汇报快照，不随后台新实验自动更新。点击打开对应文件。

以下为原始证据和源码的本地绝对路径，未随本 Markdown 上传；GitHub 页面不能直接打开这些本地文件。

- **LightGBM 指定 3m40.740 计时**：`/workspace/z30084427/lightgbm-round-comparison-20261019/previous-timing-comparison.json`
- **LightGBM 指定运行指标及 74 项检查**：`/workspace/z30084427/lightgbm-round-comparison-20261019/previous-histogram-artifacts.json`
- **LightGBM 融合对照预测一致性**：`/workspace/z30084427/lightgbm-round-comparison-20261019/previous-fused-parity.json`
- **LightGBM 独立采样算子原表**：`/workspace/z30084427/lightgbm-round-comparison-20261019/profile-after-fused/raw/PROF_000001_20261019070826589_00671324CABJARMH/mindstudio_profiler_output/op_statistic_20261019071225.csv`
- **LightGBM 采样加载库身份**：`/workspace/z30084427/lightgbm-round-comparison-20261019/profile-after-fused/loaded-environment.json`
- **LightGBM 采样运行元数据**：`/workspace/z30084427/lightgbm-round-comparison-20261019/profile-after-fused/time.json`
- **XGBoost 全量指标与版本比较**：`/workspace/xgb-npu-runs/cuda-alignment-comparison-20261019/comparison.json`
- **XGBoost 原始算子对照表**：`/workspace/xgb-npu-runs/cuda-alignment-comparison-20261019/op_comparison.csv`
- **XGBoost CUDA 对齐审计与剩余项**：`/workspace/xgb-npu-runs/cuda-alignment-comparison-20261019/alignment-audit.json`
- **LightGBM 设备与主机边界源码**：`/workspace/LightGBM/src/treelearner/npu_tree_learner.cpp`
- **XGBoost 原生树更新源码**：`/workspace/xgboost-npu/src/tree/updater_npu_hist.cc`
- **XGBoost 预测能力边界源码**：`/workspace/xgboost-npu/src/predictor/npu_predictor.cc`

---

格式说明：本文由同目录的 `LightGBM-XGBoost-qrun-summary.canvas.tsx` 转换而来，保留原始静态快照的文字、指标、表格和证据路径；仅做格式转换，未重新训练、计时或采集数据。Markdown 可直接在 GitHub 阅读，无需 `cursor/canvas` 运行环境。
