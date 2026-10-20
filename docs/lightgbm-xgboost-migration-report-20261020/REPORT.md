# LightGBM / XGBoost 迁移至 Ascend NPU：全量代码修改报告

**采集时间（UTC）**：2026-10-20T08:03:07.660496+00:00

## 1. 范围、基线与分类口径

本报告列出当前实现相对原始训练库/Qlib 的全部净源码修改，包括已提交改动、工作区改动和限定源码路径内未跟踪文件。目录不使用通配符代替文件；逐文件状态、职责、增删行和 SHA256 见后续表格及 inventory.csv/json。历史中间版、原文件备份、第三方依赖、构建目录、mlruns、profiling 数据不重复作为生产源码。

外部部署/验收脚本没有统一原始 Git 基线，单独列出当前可定位文件，不声称覆盖机器所有 /tmp 历史临时文件。原始基线不是最近一次性能优化快照，也不是当前 HEAD。

| 仓库 | 原始基线 | 当前HEAD | 基线说明 |
| --- | --- | --- | --- |
| LightGBM | 2b59a44a894da144322f96c6bd68f89c5086e378 | e4ea1f2b461aaeed92aeb55cc3aea69ac8d04310 | Baseline: pristine LightGBM 4.7.0 source (unmodified) |
| xgboost-npu | 6fe8c547bdd21c73e4555d85b087d9260595d30d | 15afd2af9fa66f96e6731073c57f9d62e26dc0d8 | Bump version to 3.4.1. (#12468) |
| qlib | 54355232463878d2eebb91fe0ee5fa7fa1f5976c | 715b4993104232b3a7399927f5254f5321d782f8 | fix(security): constrain config-driven code execution (#2340) |
| qlib-xgb-npu | 54355232463878d2eebb91fe0ee5fa7fa1f5976c | 715b4993104232b3a7399927f5254f5321d782f8 | fix(security): constrain config-driven code execution (#2340) |

- **框架侧**：Qlib 配置/入口、设备抽象、Learner/Boosting/预测器接入、运行时资源和缓存生命周期、构建、安装部署。
- **算子侧**：直方图、采样、量化、分裂评分/选择、行分区、设备预测等计算；包括主机算子封装、共享数学规则和 ABI（Application Binary Interface，主机与设备二进制接口），不要求文件本身一定在 NPU 上执行。
- 每个文件有唯一主分类；跨层文件在职责中注明次职责。测试文件仍按被测层分类，但与生产实现分列。文件数不是算子数，头文件/ABI/测试不能重复算成已执行设备算子。

## 2. 完整统计

| 仓库 | 主分类 | 生产/构建/配置 | 测试 | 验收工具 | 合计 |
| --- | --- | --- | --- | --- | --- |
| LightGBM | 框架侧 | 14 | 8 | 0 | 22 |
| LightGBM | 算子侧 | 14 | 11 | 0 | 25 |
| xgboost-npu | 框架侧 | 24 | 20 | 0 | 44 |
| xgboost-npu | 算子侧 | 34 | 53 | 0 | 87 |
| qlib | 框架侧 | 11 | 2 | 0 | 13 |
| qlib | 算子侧 | 0 | 0 | 0 | 0 |
| qlib-xgb-npu | 框架侧 | 5 | 1 | 2 | 8 |
| qlib-xgb-npu | 算子侧 | 0 | 0 | 0 | 0 |

基线差分清单合计 **199 个物理路径**；另列系统入口 1 个、外部辅助源文件 26 个和构建/安装产物 6 个。Qlib 主/隔离副本是不同物理路径，统计不能当作独立算法改动数量相加。

## 3. LightGBM 迁移方案与边界

基于 LightGBM 4.7.0，新增 NPUTreeLearner 继承 SerialTreeLearner，工厂以 device_type=npu 接入；保留原 Boosting、树结构、early stopping 和模型格式。NPU 执行直方图构建/相减、分裂搜索、行分区及训练分数更新；CPU 仍负责数据准备/分箱、目标函数梯度/Hessian、树控制、评估及普通预测。分类排序/增益及部分 refit/叶更新仍有主机路径。

最新融合直方图/分裂使用 FP64 INLINE AICPU：AI CPU 是 Ascend 设备上的 CPU 执行单元，不是主机 CPU，也不能描述为 AI Core 矩阵加速。训练分数更新后仍可下载给主机，因此不是全流程设备常驻。

NPU 公开 use_quantized_grad 入口明确拒绝；内部量化代码和测试仍完整列出。单机 serial 之外、linear tree、单调约束、CEGB、forced splits 等能力仍受限制。MAPE/量化相关公共文件修改可能同时改变 CPU 数值行为，不能当作仅 NPU 私有改动。

## 4. XGBoost 迁移方案与边界

基于 XGBoost 3.4.1，在原 learner/GBTree 内新增 npu 设备、grow_npu_hist updater 和 NPU predictor；标准 RegTree、early stopping 和模型 IO 保留。AscendC/ASC（CANN 设备编译语言）与 SIMT（单指令多线程）内核及 ACLNN 承担训练和预测计算。CPU 当前仍执行 SketchOnDMatrix 分箱、初始截距和允许的 RMSE/MAE 评估，以及小特征集合列采样和树控制。

已实现共享只读分箱、连续直方图、兄弟相减、行索引双缓冲、层级批量分裂、有界分配缓存及共享分配生命周期。行/列采样参考固定 CCCL 3.3.3（CUDA C++ 核心库）；修复实际 learner RNG（随机数状态）推进和预测累加顺序。

当前生产内核入口是 src/common/npu/kernel/production_kernels.asc，通过单一翻译单元合并七组设备实现；native_kernels.asc 和 native_training_kernel/ 属保留/独立构建路径，清单保留但不称当前主库同时执行。regression_obj.cu 的修改是 NPU 条件分支接入，不能凭 .cu 后缀认定完整 CUDA 迁移。

尚未完成 CUDA 等价：CPU/CUDA 分箱算法不同；量化证书接受域较窄；初始截距归约及 CUDA 设备舍入未全链路验证。当前支持范围为单设备、内存内数值特征、单目标平方误差训练；不支持项应明确失败。

## 5. 当前 Qlib、部署与 profiling 接入

当前默认 qrun 仅将 XGBModel 且设备缺省/NPU 的工作流分流到 /workspace/xgb-npu-venv 和固定源码/动态库；显式 CPU/CUDA 和非 XGB 工作流走备份原入口。LightGBM 使用默认 Python 包中的已部署 .so。Qlib 原 qlib/cli/run.py 相对原始基线未修改。

XGBoost Alpha158/Alpha360 YAML 已显式指定 npu:0/hist；LightGBM Alpha158 显式指定 npu/verbose:2，LightGBM Alpha360 相对基线未改。日志区分选择后端与第一轮实际训练确认，模型配置不符立即失败。两个后端共用机器不意味着可并发占用同一 NPU。

外层 msprof 直接包默认 XGBoost qrun 时，分流器会清理 LD_/ASCEND_/PYTHON 等环境；手动 profiling 应先建立固定环境，再由 msprof 启动私有 qrun，并自行保持设备互斥。算子统计来自独立 profiling，不能代替非追踪墙钟计时。

## 6. 逐文件清单（全部路径）

状态 A=相对原始基线新增，M=修改，?=当前未跟踪；新增并已提交的文件仍为 A。增删行为对原始基线的净差分，新增文件整文件计入。下列表格路径相对各节给出的源码根目录。

### 6.1 LightGBM

根目录：`/workspace/LightGBM`。

#### 归属：直接迁移

##### 框架侧 — 生产代码/构建/配置（14 文件）

| 状态 | 文件 | 增/删行 | 模块 | 修改内容/职责 |
| --- | --- | --- | --- | --- |
| M | `CMakeLists.txt` | +226 / −2 | 构建 | 新增 USE_NPU、CANN/Bisheng/INLINE AICPU 依赖检测、编译链接和隔离输出目录。 |
| M | `src/io/config.cpp` | +17 / −0 | 设备配置 | 识别 npu/ascend；强制列式直方图；在配置归一化前拒绝量化梯度和 linear_tree，避免静默改用 CPU。 |
| M | `src/objective/regression_objective.hpp` | +4 / −1 | 目标函数兼容 | 修复加权 MAPE 的 IsConstantHessian：有权重时不再把逐行 Hessian 当作常量；影响通用 CPU/NPU 语义。 |
| M | `src/treelearner/data_partition.hpp` | +32 / −0 | 分区接口 | 扩展叶行成员访问/状态接口，为设备分区及兼容性主机物化提供接入。 |
| M | `src/treelearner/feature_histogram.hpp` | +14 / −12 | 通用量化兼容 | 修正打包量化统计的有符号位移表达，避免未定义行为；次职责属于算子数值规则。 |
| M | `src/treelearner/gradient_discretizer.cpp` | +4 / −2 | 通用量化兼容 | 处理零梯度/Hessian 的量化缩放，避免除零、NaN 和无效 int8 转换；不代表公开量化 NPU 训练已支持。 |
| M | `src/treelearner/leaf_splits.hpp` | +24 / −10 | 叶统计接口 | 支持设备叶无主机行索引初始化，并修正打包梯度/Hessian 的算术表达。 |
| A | `src/treelearner/npu_runtime.cpp` | +3149 / −0 | 跨层运行时 | 管理 ACL 资源、传输与缓存；同时承载 ACLNN 直方图/分裂/分区/分数更新实现，具有算子侧次职责。 |
| A | `src/treelearner/npu_runtime.h` | +265 / −0 | 跨层运行时 | 管理 ACL 资源、传输与缓存；同时承载 ACLNN 直方图/分裂/分区/分数更新实现，具有算子侧次职责。 |
| A | `src/treelearner/npu_tree_learner.cpp` | +621 / −0 | 设备训练编排 | 接入 SerialTreeLearner；配置/数据校验、缓存、叶生长与设备直方图/分裂/分区/分数更新调度；必要时物化主机行。 |
| A | `src/treelearner/npu_tree_learner.h` | +114 / −0 | 设备训练编排 | 接入 SerialTreeLearner；配置/数据校验、缓存、叶生长与设备直方图/分裂/分区/分数更新调度；必要时物化主机行。 |
| M | `src/treelearner/serial_tree_learner.cpp` | +28 / −30 | 训练生命周期 | 把叶选择、分区、叶统计及量化输出更新接入可覆写接口，复用原串行树训练过程。 |
| M | `src/treelearner/serial_tree_learner.h` | +32 / −0 | 训练生命周期 | 声明 NPU 派生学习器所需虚接口；跨设备公共扩展点。 |
| M | `src/treelearner/tree_learner.cpp` | +7 / −0 | 后端注册 | 为单机 serial/npu 注册 NPUTreeLearner；不支持的学习器类型明确报错。 |

##### 框架侧 — 测试/验收（8 文件）

| 状态 | 文件 | 增/删行 | 模块 | 修改内容/职责 |
| --- | --- | --- | --- | --- |
| A | `tests/cpp_tests/test_npu_lifecycle_probe.cpp` | +90 / −0 | 测试/实验 | lifecycle_probe 的探针/回归或精度性能实验；列入测试支持，不计作生产实现。 |
| A | `tests/cpp_tests/test_npu_quantized_tree_probe.cpp` | +549 / −0 | 测试/实验 | quantized_tree_probe 的探针/回归或精度性能实验；列入测试支持，不计作生产实现。 |
| A | `tests/test_npu_categorical.py` | +364 / −0 | 测试/实验 | categorical 的探针/回归或精度性能实验；列入测试支持，不计作生产实现。 |
| A | `tests/test_npu_extra_trees.py` | +309 / −0 | 测试/实验 | extra_trees 的探针/回归或精度性能实验；列入测试支持，不计作生产实现。 |
| A | `tests/test_npu_feature_sampling.py` | +179 / −0 | 测试/实验 | feature_sampling 的探针/回归或精度性能实验；列入测试支持，不计作生产实现。 |
| A | `tests/test_npu_leaf_rules.py` | +240 / −0 | 测试/实验 | leaf_rules 的探针/回归或精度性能实验；列入测试支持，不计作生产实现。 |
| A | `tests/test_npu_objectives.py` | +525 / −0 | 测试/实验 | objectives 的探针/回归或精度性能实验；列入测试支持，不计作生产实现。 |
| A | `tests/test_npu_training.py` | +150 / −0 | 测试/实验 | training 的探针/回归或精度性能实验；列入测试支持，不计作生产实现。 |

##### 算子侧 — 生产代码/构建/配置（14 文件）

| 状态 | 文件 | 增/删行 | 模块 | 修改内容/职责 |
| --- | --- | --- | --- | --- |
| A | `src/treelearner/npu_best_split_finder.cpp` | +37 / −0 | 主机算子封装 | 设备数值/内部量化最佳分裂搜索的主机封装和参数/结果转换。 |
| A | `src/treelearner/npu_best_split_finder.h` | +51 / −0 | 主机算子封装 | 设备数值/内部量化最佳分裂搜索的主机封装和参数/结果转换。 |
| A | `src/treelearner/npu_data_partition.cpp` | +83 / −0 | 主机算子封装 | 管理设备叶行索引和数值/分类分区；是算子主机封装，不等于所有分类增益计算均在设备。 |
| A | `src/treelearner/npu_data_partition.h` | +48 / −0 | 主机算子封装 | 管理设备叶行索引和数值/分类分区；是算子主机封装，不等于所有分类增益计算均在设备。 |
| A | `src/treelearner/npu_fused_histogram.aicpu` | +42 / −0 | 设备融合内核 | FP64 多特征直方图融合累加，按设备行索引顺序执行，运行于 Ascend 设备 AI CPU。 |
| A | `src/treelearner/npu_fused_histogram.h` | +34 / −0 | 算子ABI | 定义 npu_fused_histogram 的主机/设备共享参数、结果和 ABI。 |
| A | `src/treelearner/npu_fused_split.aicpu` | +214 / −0 | 设备融合内核 | 融合 FP64 前缀统计、增益计算和最优分裂筛选，运行于设备 AI CPU。 |
| A | `src/treelearner/npu_fused_split.h` | +76 / −0 | 算子ABI | 定义 npu_fused_split 的主机/设备共享参数、结果和 ABI。 |
| A | `src/treelearner/npu_gradient_stager.cpp` | +32 / −0 | 主机算子封装 | 上传并暂存主机生成的量化梯度/Hessian；用于内部探针路径，公开 NPU 量化入口仍被拒绝。 |
| A | `src/treelearner/npu_gradient_stager.h` | +36 / −0 | 主机算子封装 | 上传并暂存主机生成的量化梯度/Hessian；用于内部探针路径，公开 NPU 量化入口仍被拒绝。 |
| A | `src/treelearner/npu_histogram_constructor.cpp` | +42 / −0 | 主机算子封装 | 封装 FP64/内部量化设备直方图的构建、修正和兄弟直方图相减。 |
| A | `src/treelearner/npu_histogram_constructor.h` | +30 / −0 | 主机算子封装 | 封装 FP64/内部量化设备直方图的构建、修正和兄弟直方图相减。 |
| A | `src/treelearner/npu_score_updater.cpp` | +23 / −0 | 主机算子封装 | 按设备叶行成员和叶输出更新训练分数；运行时随后可将分数下载给主机。 |
| A | `src/treelearner/npu_score_updater.h` | +33 / −0 | 主机算子封装 | 按设备叶行成员和叶输出更新训练分数；运行时随后可将分数下载给主机。 |

##### 算子侧 — 测试/验收（11 文件）

| 状态 | 文件 | 增/删行 | 模块 | 修改内容/职责 |
| --- | --- | --- | --- | --- |
| A | `tests/cpp_tests/test_npu_histogram_probe.cpp` | +1574 / −0 | 测试/实验 | histogram_probe 的探针/回归或精度性能实验；列入测试支持，不计作生产实现。 |
| A | `tests/cpp_tests/test_npu_split_ops_probe.cpp` | +380 / −0 | 测试/实验 | split_ops_probe 的探针/回归或精度性能实验；列入测试支持，不计作生产实现。 |
| A | `tests/cpp_tests/test_npu_split_probe.cpp` | +367 / −0 | 测试/实验 | split_probe 的探针/回归或精度性能实验；列入测试支持，不计作生产实现。 |
| A | `tests/npu_port/test_bincount.cpp` | +109 / −0 | 测试/实验 | bincount 的探针/回归或精度性能实验；列入测试支持，不计作生产实现。 |
| A | `tests/npu_port/test_chunked.cpp` | +118 / −0 | 测试/实验 | chunked 的探针/回归或精度性能实验；列入测试支持，不计作生产实现。 |
| A | `tests/npu_port/test_dtype_perf.cpp` | +135 / −0 | 测试/实验 | dtype_perf 的探针/回归或精度性能实验；列入测试支持，不计作生产实现。 |
| A | `tests/npu_port/test_dtype_scale.cpp` | +163 / −0 | 测试/实验 | dtype_scale 的探针/回归或精度性能实验；列入测试支持，不计作生产实现。 |
| A | `tests/npu_port/test_error_metric.cpp` | +125 / −0 | 测试/实验 | error_metric 的探针/回归或精度性能实验；列入测试支持，不计作生产实现。 |
| A | `tests/npu_port/test_exact_fast.cpp` | +179 / −0 | 测试/实验 | exact_fast 的探针/回归或精度性能实验；列入测试支持，不计作生产实现。 |
| A | `tests/npu_port/test_fp16_hypothesis.cpp` | +108 / −0 | 测试/实验 | fp16_hypothesis 的探针/回归或精度性能实验；列入测试支持，不计作生产实现。 |
| A | `tests/npu_port/test_scale_hist.cpp` | +139 / −0 | 测试/实验 | scale_hist 的探针/回归或精度性能实验；列入测试支持，不计作生产实现。 |

### 6.2 xgboost-npu

根目录：`/workspace/xgboost-npu`。

#### 归属：直接迁移

##### 框架侧 — 生产代码/构建/配置（24 文件）

| 状态 | 文件 | 增/删行 | 模块 | 修改内容/职责 |
| --- | --- | --- | --- | --- |
| M | `CMakeLists.txt` | +14 / −0 | 构建 | 新增 USE_NPU 开关、构建约束、生产内核链接和安装接入。 |
| A | `cmake/NPU.cmake` | +75 / −0 | 构建部署 | 查找 CANN 运行库；构建或验证生产内核 .so；设置 RPATH 并将内核库放在 libxgboost 旁。 |
| A | `cmake/NPUConfig.cmake` | +31 / −0 | 构建约束 | 固定 CANN 9.1.0、Linux x86_64、dav-3510、Bisheng 和 ASC 构建依赖。 |
| M | `include/xgboost/context.h` | +22 / −2 | 设备抽象 | 新增 NPU 设备枚举/构造/分发并保持旧设备编号；未实现路径拒绝 CPU 回退。 |
| M | `include/xgboost/gradient.h` | +4 / −0 | 设备梯度所有权 | 为梯度容器增加独立拥有生命周期的 NPU 梯度对象，不以主机 gpair 替代。 |
| A | `include/xgboost/npu.h` | +10 / −0 | 设备数据接口 | 声明 NPU 梯度、预测、运行时等对象和框架桥接接口。 |
| M | `include/xgboost/predictor.h` | +4 / −1 | 设备预测缓存 | 为预测缓存增加驻留 NPU 对象；主机预测向量是显式指标/API 导出。 |
| M | `src/CMakeLists.txt` | +12 / −0 | 源文件构建 | 在 USE_NPU 下纳入后端 C++ 生产源文件及依赖。 |
| M | `src/c_api/c_api.cc` | +5 / −0 | 能力查询 | 向 XGBuildInfo 输出 USE_NPU，供 Python/Qlib 拒绝错误训练库。 |
| A | `src/common/npu/acl_api.h` | +54 / −0 | 设备资源适配 | 生产 ACL API 适配、context/stream/内存传输、缓存与调用统计，供可注入运行时使用。 |
| A | `src/common/npu/kernel/CMakeLists.txt` | +31 / −0 | 生产内核构建 | 独立 ASC 工程编译 libxgboost_npu_kernels.so，不改变主机 CXX 编译器。 |
| A | `src/common/npu/native_prediction.cc` | +594 / −0 | 预测数据桥接 | 标准模型/输入数据到设备预测结构的转换、上传和预测缓存生命周期；调用设备算子。 |
| A | `src/common/npu/native_prediction.h` | +155 / −0 | 预测桥接接口 | 声明设备预测缓存及模型/输入管理接口；次职责为算子编排。 |
| A | `src/common/npu/native_training.cc` | +162 / −0 | 训练数据桥接 | 绑定 DMatrix/SparsePage，CPU SketchOnDMatrix 分箱并上传，共享不可变分箱缓存及训练算子调度。 |
| A | `src/common/npu/native_training.h` | +51 / −0 | 训练缓存接口 | 训练输入缓存及分箱所有权接口；按 SparsePage 共享所有权识别数据生命周期。 |
| A | `src/common/npu/native_training_kernel/CMakeLists.txt` | +17 / −0 | 独立内核构建项目 | 保留的独立训练内核工程；主生产 cmake/NPU.cmake 当前使用 common/npu/kernel 工程。 |
| A | `src/common/npu/runtime.h` | +274 / −0 | 内存和生命周期 | 单主机线程/单 stream 运行时、共享分配与 slice 生命周期、有界分配缓存、清理和重试；不做全局设备重置/静默回退。 |
| M | `src/context.cc` | +81 / −16 | 设备解析 | 解析和校验 npu/npu:ordinal；未编译支持或非法设备配置显式失败。 |
| M | `src/gbm/gbtree.cc` | +113 / −1 | Boosting接入 | NPU updater/predictor 选择、训练梯度对象传递及不支持能力检查。 |
| M | `src/gbm/gbtree.h` | +3 / −0 | Boosting状态 | 保存 NPU predictor 等后端状态并接入树提升生命周期。 |
| M | `src/learner.cc` | +148 / −17 | Learner接入 | 接入设备驻留梯度和预测；主机截距预处理及 RMSE/MAE 评估上下文；校验单目标能力和状态生命周期。 |
| M | `src/objective/regression_obj.cu` | +46 / −1 | 目标函数接入 | 给 reg:squarederror 增加 NPU 配置限制、主机截距和恒等链接分支；拒绝旧主机梯度入口；并非整体移植 CUDA 文件。 |
| A | `src/predictor/npu_predictor.cc` | +68 / −0 | 预测器接入 | 注册 NPU predictor，连接标准树模型与设备预测/缓存，显式导出主机 API 结果。 |
| A | `src/tree/updater_npu_hist.cc` | +277 / −0 | 树更新器编排 | 注册 grow_npu_hist；编排行列采样、整数直方图、层级分裂/分区、兄弟相减与标准 RegTree 提交。 |

##### 框架侧 — 测试/验收（20 文件）

| 状态 | 文件 | 增/删行 | 模块 | 修改内容/职责 |
| --- | --- | --- | --- | --- |
| A | `tests/npu_port/build_native_probe.sh` | +53 / −0 | 测试/验收 | 原生后端：构建/隔离 Python 测试入口。 |
| A | `tests/npu_port/check_native_backend.py` | +388 / −0 | 测试/验收 | 原生后端：测试驱动、结果校验与身份/设备保护。 |
| A | `tests/npu_port/check_runtime.py` | +174 / −0 | 测试/验收 | 运行时生命周期：测试驱动、结果校验与身份/设备保护。 |
| A | `tests/npu_port/check_transfer_audit.py` | +470 / −0 | 测试/验收 | 主机设备传输审计：测试驱动、结果校验与身份/设备保护。 |
| A | `tests/npu_port/context_host_test.cc` | +129 / −0 | 测试/验收 | 设备上下文：主机参考、生命周期或契约回归（不是 CUDA 设备执行证明）。 |
| A | `tests/npu_port/context_regression.py` | +146 / −0 | 测试/验收 | 设备上下文：回归、设备探针或验收支持。 |
| A | `tests/npu_port/cpu_baseline.py` | +142 / −0 | 测试/验收 | NPU 组件：回归、设备探针或验收支持。 |
| A | `tests/npu_port/learner_npu_regression.py` | +120 / −0 | 测试/验收 | Learner 接入：回归、设备探针或验收支持。 |
| A | `tests/npu_port/native_backend_cases.py` | +251 / −0 | 测试/验收 | 原生后端：测试用例和边界输入。 |
| A | `tests/npu_port/native_probe.cc` | +410 / −0 | 测试/验收 | 原生后端：回归、设备探针或验收支持。 |
| A | `tests/npu_port/run_python.sh` | +11 / −0 | 测试/验收 | NPU 组件：构建/隔离 Python 测试入口。 |
| A | `tests/npu_port/run_transfer_audit.py` | +78 / −0 | 测试/验收 | 主机设备传输审计：回归、设备探针或验收支持。 |
| A | `tests/npu_port/runtime_host_test.cc` | +491 / −0 | 测试/验收 | 运行时生命周期：主机参考、生命周期或契约回归（不是 CUDA 设备执行证明）。 |
| A | `tests/npu_port/runtime_int64_probe.cc` | +320 / −0 | 测试/验收 | 运行时生命周期：回归、设备探针或验收支持。 |
| A | `tests/npu_port/runtime_no_device_guard.cc` | +32 / −0 | 测试/验收 | 运行时生命周期：回归、设备探针或验收支持。 |
| A | `tests/npu_port/test_check_native_backend_host.py` | +292 / −0 | 测试/验收 | 原生后端：主机参考、生命周期或契约回归（不是 CUDA 设备执行证明）。 |
| A | `tests/npu_port/test_check_transfer_audit_host.py` | +243 / −0 | 测试/验收 | 主机设备传输审计：主机参考、生命周期或契约回归（不是 CUDA 设备执行证明）。 |
| A | `tests/npu_port/test_transfer_audit_host.py` | +134 / −0 | 测试/验收 | 主机设备传输审计：主机参考、生命周期或契约回归（不是 CUDA 设备执行证明）。 |
| A | `tests/npu_port/transfer_audit.cc` | +160 / −0 | 测试/验收 | 主机设备传输审计：回归、设备探针或验收支持。 |
| A | `tests/npu_port/transfer_audit_host_fixture.cc` | +97 / −0 | 测试/验收 | 主机设备传输审计：主机参考、生命周期或契约回归（不是 CUDA 设备执行证明）。 |

##### 算子侧 — 生产代码/构建/配置（34 文件）

| 状态 | 文件 | 增/删行 | 模块 | 修改内容/职责 |
| --- | --- | --- | --- | --- |
| A | `src/common/npu/acl_ops.h` | +308 / −0 | ACLNN封装 | 张量/工作区管理及 Add/Sub/Scatter/Reduce/Cast/位运算等 ACLNN 调用；次职责为运行时资源管理。 |
| A | `src/common/npu/candidate_selection.asc` | +34 / −0 | 候选选择内核 | 设备候选分裂选择及归约。 |
| A | `src/common/npu/candidate_selection.h` | +38 / −0 | 候选选择封装 | 候选选择主机接口、参数检查及内核调用。 |
| A | `src/common/npu/candidate_selection_abi.h` | +49 / −0 | 算子ABI | 候选选择内核共享参数/结果布局。 |
| A | `src/common/npu/candidate_selection_device.h` | +19 / −0 | 设备选择逻辑 | 候选比较和设备选择规则，供内核共享。 |
| A | `src/common/npu/certified_scale.h` | +84 / −0 | 量化数值契约 | 可接受量化缩放的边界证明/证书；可能拒绝 CUDA 可处理的模糊指数边界。 |
| A | `src/common/npu/cuda_column_sampler.h` | +258 / −0 | 主机采样算法 | 参考固定 CCCL 3.3.3 的 Thrust/Feistel 列采样及排序；在主机处理小特征集，推进实际 learner RNG。 |
| A | `src/common/npu/cuda_row_sampler.h` | +105 / −0 | 共享采样算法 | 固定 CCCL 3.3.3 Philox 行随机数、float canonical 转换及采样边界规则，供主机参考和 ASC 内核共享。 |
| A | `src/common/npu/device_row_partitioner.h` | +181 / −0 | 行分区封装 | 节点行区间、双缓冲索引及设备分区工作区复用。 |
| A | `src/common/npu/dynamic_quantizer.h` | +52 / −0 | 动态量化封装 | 设备统计界、缩放拟合与 float32 梯度/Hessian 到整数通道转换。 |
| A | `src/common/npu/fixed_quantizer.h` | +94 / −0 | 固定量化封装 | 按给定二进制幂缩放进行位精确 float32→int64 转换；完整缩放拟合在其他组件完成。 |
| A | `src/common/npu/histogram_contract.h` | +28 / −0 | 直方图契约 | 直方图输入、布局、数值/一致性约束。 |
| A | `src/common/npu/integer_histogram.asc` | +73 / −0 | 整数直方图内核 | 设备整数直方图构建/操作的 ASC 实现。 |
| A | `src/common/npu/integer_histogram.h` | +252 / −0 | 直方图封装 | 共享只读分箱、连续梯度/Hessian 直方图、兄弟相减及工作区复用。 |
| A | `src/common/npu/integer_histogram_abi.h` | +28 / −0 | 算子ABI | 整数直方图内核参数/结果布局。 |
| A | `src/common/npu/kernel/production_kernels.asc` | +10 / −0 | 生产内核汇编入口 | 当前生产单翻译单元，包含评分/分裂/候选/直方图/预测/训练内核，避免 CANN 重复全局符号。 |
| A | `src/common/npu/native_kernels.asc` | +124 / −0 | 保留实现 | 较早设备预测/梯度实现；未被当前 production_kernels.asc 包含，不能作为当前部署内核重复计数。 |
| A | `src/common/npu/native_prediction.asc` | +263 / −0 | 设备预测内核 | 标准树遍历和预测累加；按树区间先求和再加缓存/初始 margin。 |
| A | `src/common/npu/native_prediction_abi.h` | +91 / −0 | 算子ABI | 树、节点、稀疏输入及预测/梯度内核共享布局。 |
| A | `src/common/npu/native_training.asc` | +97 / −0 | 训练设备内核 | 设备梯度/行采样、量化、行分区等训练计算入口。 |
| A | `src/common/npu/native_training_abi.h` | +19 / −0 | 算子ABI | 训练/采样/分区内核共享布局。 |
| A | `src/common/npu/native_training_kernel/native_training_kernels.asc` | +8 / −0 | 独立内核汇编入口 | 独立训练内核工程的合并入口，非当前主生产库入口；文件仍完整纳入清单。 |
| A | `src/common/npu/numerical_split_evaluator.asc` | +301 / −0 | 数值分裂内核 | 批量前缀统计、评分、选择和完成四阶段设备计算。 |
| A | `src/common/npu/numerical_split_evaluator.h` | +151 / −0 | 批量分裂封装 | 层级批量数值分裂、工作区管理及内核调度。 |
| A | `src/common/npu/numerical_split_evaluator_abi.h` | +67 / −0 | 算子ABI | 批量数值分裂输入/输出与工作区约定。 |
| A | `src/common/npu/precise_scoring.asc` | +67 / −0 | 精确评分内核 | 软件精确算术/评分入口和设备计算。 |
| A | `src/common/npu/precise_scoring.h` | +185 / −0 | 评分封装 | 精确评分参数校验和设备调用。 |
| A | `src/common/npu/precise_scoring_abi.h` | +14 / −0 | 算子ABI | 精确评分共享参数/结果布局。 |
| A | `src/common/npu/precise_split_scorer.h` | +61 / −0 | 共享评分规则 | 精确增益/权重等分裂数学规则，供主机参考及设备实现使用。 |
| A | `src/common/npu/quantizer_scale.h` | +59 / −0 | 量化缩放规则 | 量化缩放表示、范围和辅助数值规则。 |
| A | `src/common/npu/row_partition_contract.h` | +27 / −0 | 行分区契约 | 行分区输入、结果和一致性约束。 |
| A | `src/common/npu/split_finalization.asc` | +69 / −0 | 分裂完成内核 | 从候选统计计算最终分裂结果。 |
| A | `src/common/npu/split_finalization.h` | +65 / −0 | 分裂完成封装 | 最终分裂计算的主机校验及调用。 |
| A | `src/common/npu/split_statistics.h` | +81 / −0 | 分裂统计规则 | 梯度/Hessian 汇总、分裂统计和数值检查。 |

##### 算子侧 — 测试/验收（53 文件）

| 状态 | 文件 | 增/删行 | 模块 | 修改内容/职责 |
| --- | --- | --- | --- | --- |
| A | `tests/npu_port/candidate_selection_cases.h` | +86 / −0 | 测试/验收 | 候选选择：测试用例和边界输入。 |
| A | `tests/npu_port/candidate_selection_contract.json` | +56 / −0 | 测试/验收 | 候选选择：参考契约/固定期望值。 |
| A | `tests/npu_port/candidate_selection_host_test.cc` | +6 / −0 | 测试/验收 | 候选选择：主机参考、生命周期或契约回归（不是 CUDA 设备执行证明）。 |
| A | `tests/npu_port/check_cuda_contract.py` | +214 / −0 | 测试/验收 | CUDA 源码语义契约：测试驱动、结果校验与身份/设备保护。 |
| A | `tests/npu_port/check_cuda_row_sampler.py` | +173 / −0 | 测试/验收 | 固定 CUDA 参考行采样：测试驱动、结果校验与身份/设备保护。 |
| A | `tests/npu_port/check_dynamic_quantizer.py` | +202 / −0 | 测试/验收 | 动态量化：测试驱动、结果校验与身份/设备保护。 |
| A | `tests/npu_port/check_fixed_quantizer.py` | +193 / −0 | 测试/验收 | 固定量化：测试驱动、结果校验与身份/设备保护。 |
| A | `tests/npu_port/check_integer_histogram.py` | +227 / −0 | 测试/验收 | 整数直方图：测试驱动、结果校验与身份/设备保护。 |
| A | `tests/npu_port/check_numerical_split.py` | +258 / −0 | 测试/验收 | 数值分裂：测试驱动、结果校验与身份/设备保护。 |
| A | `tests/npu_port/check_precise_scoring.py` | +178 / −0 | 测试/验收 | 精确评分：测试驱动、结果校验与身份/设备保护。 |
| A | `tests/npu_port/check_row_partition.py` | +178 / −0 | 测试/验收 | 行分区：测试驱动、结果校验与身份/设备保护。 |
| A | `tests/npu_port/check_split_finalization.py` | +170 / −0 | 测试/验收 | 分裂完成：测试驱动、结果校验与身份/设备保护。 |
| A | `tests/npu_port/check_split_statistics.py` | +191 / −0 | 测试/验收 | 分裂统计：测试驱动、结果校验与身份/设备保护。 |
| A | `tests/npu_port/cuda_column_sampler_host_test.cc` | +399 / −0 | 测试/验收 | 固定 CUDA 参考列采样：主机参考、生命周期或契约回归（不是 CUDA 设备执行证明）。 |
| A | `tests/npu_port/cuda_contract.json` | +198 / −0 | 测试/验收 | CUDA 源码语义契约：参考契约/固定期望值。 |
| A | `tests/npu_port/cuda_row_sampler_device_test.cc` | +215 / −0 | 测试/验收 | 固定 CUDA 参考行采样：回归、设备探针或验收支持。 |
| A | `tests/npu_port/cuda_row_sampler_host_test.cc` | +252 / −0 | 测试/验收 | 固定 CUDA 参考行采样：主机参考、生命周期或契约回归（不是 CUDA 设备执行证明）。 |
| A | `tests/npu_port/cuda_semantics_test.cc` | +302 / −0 | 测试/验收 | CUDA 源码语义契约：回归、设备探针或验收支持。 |
| A | `tests/npu_port/dynamic_quantizer_cases.h` | +84 / −0 | 测试/验收 | 动态量化：测试用例和边界输入。 |
| A | `tests/npu_port/dynamic_quantizer_host_test.cc` | +86 / −0 | 测试/验收 | 动态量化：主机参考、生命周期或契约回归（不是 CUDA 设备执行证明）。 |
| A | `tests/npu_port/dynamic_quantizer_test.cc` | +107 / −0 | 测试/验收 | 动态量化：回归、设备探针或验收支持。 |
| A | `tests/npu_port/fixed_quantizer_host_test.cc` | +57 / −0 | 测试/验收 | 固定量化：主机参考、生命周期或契约回归（不是 CUDA 设备执行证明）。 |
| A | `tests/npu_port/fixed_quantizer_test.cc` | +148 / −0 | 测试/验收 | 固定量化：回归、设备探针或验收支持。 |
| A | `tests/npu_port/histogram_cases.h` | +89 / −0 | 测试/验收 | 直方图：测试用例和边界输入。 |
| A | `tests/npu_port/integer_histogram_host_test.cc` | +53 / −0 | 测试/验收 | 整数直方图：主机参考、生命周期或契约回归（不是 CUDA 设备执行证明）。 |
| A | `tests/npu_port/integer_histogram_test.cc` | +329 / −0 | 测试/验收 | 整数直方图：回归、设备探针或验收支持。 |
| A | `tests/npu_port/native_prediction_kernel/CMakeLists.txt` | +17 / −0 | 测试/验收 | 原生设备预测：测试内核独立构建配置。 |
| A | `tests/npu_port/native_prediction_kernel/native_prediction_test_kernels.asc` | +2 / −0 | 测试/验收 | 原生设备预测：测试专用设备内核，不能计入生产库。 |
| A | `tests/npu_port/native_prediction_regression.py` | +199 / −0 | 测试/验收 | 原生设备预测：回归、设备探针或验收支持。 |
| A | `tests/npu_port/native_scale_cases.py` | +148 / −0 | 测试/验收 | 原生量化边界：测试用例和边界输入。 |
| A | `tests/npu_port/numerical_split_cases.h` | +89 / −0 | 测试/验收 | 数值分裂：测试用例和边界输入。 |
| A | `tests/npu_port/numerical_split_host_test.cc` | +6 / −0 | 测试/验收 | 数值分裂：主机参考、生命周期或契约回归（不是 CUDA 设备执行证明）。 |
| A | `tests/npu_port/numerical_split_test.cc` | +261 / −0 | 测试/验收 | 数值分裂：回归、设备探针或验收支持。 |
| A | `tests/npu_port/precise_arithmetic.asc` | +33 / −0 | 测试/验收 | 精确算术：测试专用设备内核，不能计入生产库。 |
| A | `tests/npu_port/precise_arithmetic_abi.h` | +11 / −0 | 测试/验收 | 精确算术：回归、设备探针或验收支持。 |
| A | `tests/npu_port/precise_scoring_cases.h` | +146 / −0 | 测试/验收 | 精确评分：测试用例和边界输入。 |
| A | `tests/npu_port/precise_scoring_host_test.cc` | +39 / −0 | 测试/验收 | 精确评分：主机参考、生命周期或契约回归（不是 CUDA 设备执行证明）。 |
| A | `tests/npu_port/precise_scoring_test.cc` | +174 / −0 | 测试/验收 | 精确评分：回归、设备探针或验收支持。 |
| A | `tests/npu_port/precise_test_kernels.asc` | +10 / −0 | 测试/验收 | NPU 组件：测试专用设备内核，不能计入生产库。 |
| A | `tests/npu_port/precision_kernel/CMakeLists.txt` | +18 / −0 | 测试/验收 | 精度：测试内核独立构建配置。 |
| A | `tests/npu_port/quantizer_cases.h` | +99 / −0 | 测试/验收 | 量化：测试用例和边界输入。 |
| A | `tests/npu_port/row_partition_cases.h` | +76 / −0 | 测试/验收 | 行分区：测试用例和边界输入。 |
| A | `tests/npu_port/row_partition_host_test.cc` | +42 / −0 | 测试/验收 | 行分区：主机参考、生命周期或契约回归（不是 CUDA 设备执行证明）。 |
| A | `tests/npu_port/row_partition_test.cc` | +242 / −0 | 测试/验收 | 行分区：回归、设备探针或验收支持。 |
| A | `tests/npu_port/split_finalization_cases.h` | +109 / −0 | 测试/验收 | 分裂完成：测试用例和边界输入。 |
| A | `tests/npu_port/split_finalization_host_test.cc` | +22 / −0 | 测试/验收 | 分裂完成：主机参考、生命周期或契约回归（不是 CUDA 设备执行证明）。 |
| A | `tests/npu_port/split_finalization_test.cc` | +134 / −0 | 测试/验收 | 分裂完成：回归、设备探针或验收支持。 |
| A | `tests/npu_port/split_statistics_cases.h` | +66 / −0 | 测试/验收 | 分裂统计：测试用例和边界输入。 |
| A | `tests/npu_port/split_statistics_host_test.cc` | +29 / −0 | 测试/验收 | 分裂统计：主机参考、生命周期或契约回归（不是 CUDA 设备执行证明）。 |
| A | `tests/npu_port/split_statistics_test.cc` | +183 / −0 | 测试/验收 | 分裂统计：回归、设备探针或验收支持。 |
| A | `tests/npu_port/test_check_cuda_contract_host.py` | +94 / −0 | 测试/验收 | CUDA 源码语义契约：主机参考、生命周期或契约回归（不是 CUDA 设备执行证明）。 |
| A | `tests/npu_port/test_check_integer_histogram_host.py` | +56 / −0 | 测试/验收 | 整数直方图：主机参考、生命周期或契约回归（不是 CUDA 设备执行证明）。 |
| A | `tests/npu_port/test_native_scale_cases_host.py` | +103 / −0 | 测试/验收 | 原生量化边界：主机参考、生命周期或契约回归（不是 CUDA 设备执行证明）。 |

### 6.3 qlib

根目录：`/workspace/qlib`。

#### 归属：直接迁移

##### 框架侧 — 生产代码/构建/配置（8 文件）

| 状态 | 文件 | 增/删行 | 模块 | 修改内容/职责 |
| --- | --- | --- | --- | --- |
| M | `examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml` | +2 / −0 | 框架/部署 | 显式设置 device:npu 和 verbose:2。 |
| M | `examples/benchmarks/XGBoost/workflow_config_xgboost_Alpha158.yaml` | +2 / −0 | 框架/部署 | 显式设置 device:npu:0、tree_method:hist；保留数据范围和原训练参数。 |
| M | `examples/benchmarks/XGBoost/workflow_config_xgboost_Alpha360.yaml` | +2 / −0 | 框架/部署 | 显式设置 device:npu:0、tree_method:hist；保留数据范围和原训练参数。 |
| M | `qlib/contrib/model/gbdt.py` | +11 / −9 | 框架/部署 | 给原 lgb.train 增加 TimeInspector 计时，不修改 early stopping、数据或训练参数。 |
| M | `qlib/contrib/model/xgboost.py` | +56 / −0 | 框架/部署 | 检查 USE_NPU；首轮和训练完成检查 npu:0/grow_npu_hist；打印选择/实际训练日志，拒绝静默 CPU 回退。 |
| ? | `examples/benchmarks/XGBoost/npu_port/direct_qrun.py` | +275 / −0 | 框架/部署 | 默认 qrun 的 XGBModel NPU 工作流分流、私有环境、库身份校验、NPU 占用检查与文件锁；非 XGB/显式 CPU/CUDA 走原入口。 |
| ? | `examples/benchmarks/XGBoost/npu_port/native_backend_expected.json` | +21 / −0 | 框架/部署 | 当前验收 NPU 包路径、两份动态库路径/哈希等固定身份。 |
| ? | `examples/benchmarks/XGBoost/npu_port/python-bootstrap/sitecustomize.py` | +27 / −0 | 框架/部署 | XGBoost 子进程中固定 libpath，验证动态库身份，失败退出而非回退。 |

##### 框架侧 — 测试/验收（2 文件）

| 状态 | 文件 | 增/删行 | 模块 | 修改内容/职责 |
| --- | --- | --- | --- | --- |
| ? | `examples/benchmarks/XGBoost/npu_port/test_direct_qrun.py` | +193 / −0 | 测试/验收 | qrun 分流、CPU/CUDA/非 XGB 透传、锁和环境相关主机回归。 |
| ? | `examples/benchmarks/XGBoost/npu_port/test_npu_logging.py` | +156 / −0 | 测试/验收 | NPU 构建能力、实际 updater、训练阶段日志及拒绝回退回归。 |

#### 归属：既有公共设备适配（单列）

##### 框架侧 — 生产代码/构建/配置（3 文件）

| 状态 | 文件 | 增/删行 | 模块 | 修改内容/职责 |
| --- | --- | --- | --- | --- |
| M | `qlib/config.py` | +10 / −0 | 框架/部署 | 此前公共 accelerator 默认配置及初始化合并；非本次树模型专属修改。 |
| A | `qlib/contrib/model/accelerator_config.py` | +32 / −0 | 框架/部署 | 此前公共设备参数合并助手；不能理解为替 LightGBM/XGBoost 自动选择 NPU。 |
| M | `qlib/model/trainer.py` | +3 / −0 | 框架/部署 | 此前公共模型设备参数补齐；非树模型专属，可能给模型 kwargs 补充 GPU。 |

### 6.4 qlib-xgb-npu

根目录：`/workspace/qlib-xgb-npu`。

#### 归属：隔离副本的既有公共/LightGBM变更（不重复归因）

##### 框架侧 — 生产代码/构建/配置（2 文件）

| 状态 | 文件 | 增/删行 | 模块 | 修改内容/职责 |
| --- | --- | --- | --- | --- |
| M | `examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml` | +2 / −0 | 框架/部署 | 显式设置 device:npu 和 verbose:2。 |
| M | `qlib/contrib/model/gbdt.py` | +11 / −9 | 框架/部署 | 给原 lgb.train 增加 TimeInspector 计时，不修改 early stopping、数据或训练参数。 |

#### 归属：既有公共设备适配（单列）

##### 框架侧 — 生产代码/构建/配置（3 文件）

| 状态 | 文件 | 增/删行 | 模块 | 修改内容/职责 |
| --- | --- | --- | --- | --- |
| M | `qlib/config.py` | +10 / −0 | 框架/部署 | 此前公共 accelerator 默认配置及初始化合并；非本次树模型专属修改。 |
| A | `qlib/contrib/model/accelerator_config.py` | +32 / −0 | 框架/部署 | 此前公共设备参数合并助手；不能理解为替 LightGBM/XGBoost 自动选择 NPU。 |
| M | `qlib/model/trainer.py` | +3 / −0 | 框架/部署 | 此前公共模型设备参数补齐；非树模型专属，可能给模型 kwargs 补充 GPU。 |

#### 归属：直接迁移

##### 框架侧 — 测试/验收（1 文件）

| 状态 | 文件 | 增/删行 | 模块 | 修改内容/职责 |
| --- | --- | --- | --- | --- |
| ? | `examples/benchmarks/XGBoost/npu_port/test_final_acceptance.py` | +709 / −0 | 测试/验收 | 完整验收驱动的主机侧契约/预检回归。 |

##### 框架侧 — 验收工具（2 文件）

| 状态 | 文件 | 增/删行 | 模块 | 修改内容/职责 |
| --- | --- | --- | --- | --- |
| ? | `examples/benchmarks/XGBoost/npu_port/run_cpu_baseline.py` | +147 / −0 | 验收工具 | 隔离 Qlib 的 CPU 基线采集驱动。 |
| ? | `examples/benchmarks/XGBoost/npu_port/run_final_acceptance.py` | +828 / −0 | 验收工具 | 完整 YAML 计时、独立 profiling、身份及保存加载验收驱动，含设备/资源保护。 |

## 7. 仓库外系统入口、部署产物与辅助文件

### 7.1 系统入口（真实源代码改动）

| 绝对路径 | 分类 | 原始备份 | 职责 |
| --- | --- | --- | --- |
| /usr/local/python3.12.13/bin/qrun | 框架侧 | /workspace/xgb-npu-runs/direct-qrun-backup-20261019-073449/files/usr/local/python3.12.13/bin/qrun | 默认入口增加 XGBModel NPU 分流；保留非 XGB/显式 CPU/CUDA 原入口。 |

### 7.2 动态库及安装元数据（单列，不计入源码文件数）

| 绝对路径 | 性质 | 字节数 | 当前SHA256 |
| --- | --- | --- | --- |
| /workspace/LightGBM/build-npu/out/lib_lightgbm.so | 构建/安装产物（不计作源码变更） | 10850640 | f786a2ee8af100e54d647f7ee7681d9ea977e50ef632b8528c2993e150b66a8b |
| /usr/local/python3.12.13/lib/python3.12/site-packages/lightgbm/lib/lib_lightgbm.so | 构建/安装产物（不计作源码变更） | 10850640 | f786a2ee8af100e54d647f7ee7681d9ea977e50ef632b8528c2993e150b66a8b |
| /workspace/xgboost-npu/lib-npu/libxgboost.so.3.4.1 | 构建/安装产物（不计作源码变更） | 17791240 | f73e319bcd41ccdcb9fd96297c601e4b317823e3ea02dd29f502216338e4128e |
| /workspace/xgboost-npu/lib-npu/libxgboost_npu_kernels.so | 构建/安装产物（不计作源码变更） | 1444576 | c232aa7ce483abf033f734e815d588d688ab712066334f8913043b7944be55fd |
| /workspace/xgb-npu-venv/bin/qrun | 构建/安装产物（不计作源码变更） | 227 | 2edfa05c3fa6e4311f540ce653166dd85459cea8258c8e4497adfafae7cb149c |
| /usr/local/python3.12.13/lib/python3.12/site-packages/lightgbm-4.7.0.dist-info/RECORD | 部署时同步更新的包文件校验元数据（非生产算法源码） | 1737 | a2d12298091c606406f308af4099acc1073f12e166bc8b1a2c1b7e7d8750e56c |

LightGBM 构建输出与默认安装库的哈希应一致；仓库根目录可能残留旧库，不作为当前部署库。XGBoost 以 native_backend_expected.json 校验两份库。RECORD 为安装元数据，本报告只记录当前身份，不声称它具有独立原始 Git 基线。

### 7.3 外部部署/回滚、验收及早期实验源码

| 绝对路径 | 分类 | 用途 |
| --- | --- | --- |
| /workspace/lgbm_npu_port/tests/test_bincount.cpp | 算子侧 | 早期算子精度/性能实验源文件；与主库 tests/npu_port 可能有副本，不重复计算生产改动。 |
| /workspace/lgbm_npu_port/tests/test_chunked.cpp | 算子侧 | 早期算子精度/性能实验源文件；与主库 tests/npu_port 可能有副本，不重复计算生产改动。 |
| /workspace/lgbm_npu_port/tests/test_dtype_perf.cpp | 算子侧 | 早期算子精度/性能实验源文件；与主库 tests/npu_port 可能有副本，不重复计算生产改动。 |
| /workspace/lgbm_npu_port/tests/test_dtype_scale.cpp | 算子侧 | 早期算子精度/性能实验源文件；与主库 tests/npu_port 可能有副本，不重复计算生产改动。 |
| /workspace/lgbm_npu_port/tests/test_error_metric.cpp | 算子侧 | 早期算子精度/性能实验源文件；与主库 tests/npu_port 可能有副本，不重复计算生产改动。 |
| /workspace/lgbm_npu_port/tests/test_exact_fast.cpp | 算子侧 | 早期算子精度/性能实验源文件；与主库 tests/npu_port 可能有副本，不重复计算生产改动。 |
| /workspace/lgbm_npu_port/tests/test_fp16_hypothesis.cpp | 算子侧 | 早期算子精度/性能实验源文件；与主库 tests/npu_port 可能有副本，不重复计算生产改动。 |
| /workspace/lgbm_npu_port/tests/test_scale_hist.cpp | 算子侧 | 早期算子精度/性能实验源文件；与主库 tests/npu_port 可能有副本，不重复计算生产改动。 |
| /workspace/xgb-npu-runs/cuda-alignment-comparison-20261019/collect.py | 框架侧 | 完整工作流采集、性能比较或既有产物核验工具；不计作训练算子。 |
| /workspace/xgb-npu-runs/direct-qrun-backup-20261019-073449/rollback.py | 框架侧 | 安装、部署或回滚工具；不计作训练库生产源文件。 |
| /workspace/xgb-npu-runs/direct-qrun-verification-20261019-v1/run.py | 框架侧 | 完整工作流采集、性能比较或既有产物核验工具；不计作训练算子。 |
| /workspace/xgb-npu-runs/direct-qrun-verification-20261019-v1/verify_artifacts.py | 框架侧 | 完整工作流采集、性能比较或既有产物核验工具；不计作训练算子。 |
| /workspace/xgb-npu-runs/explicit-npu-qrun-verification-20261019-v1/run.py | 框架侧 | 完整工作流采集、性能比较或既有产物核验工具；不计作训练算子。 |
| /workspace/xgb-npu-runs/explicit-npu-qrun-verification-20261019-v1/verify_artifacts.py | 框架侧 | 完整工作流采集、性能比较或既有产物核验工具；不计作训练算子。 |
| /workspace/z30084427/bench/cpu_qrun_20261016T052748Z/compare.py | 框架侧 | 完整工作流采集、性能比较或既有产物核验工具；不计作训练算子。 |
| /workspace/z30084427/bench/prof_train.py | 框架侧 | 完整工作流采集、性能比较或既有产物核验工具；不计作训练算子。 |
| /workspace/z30084427/lightgbm-direct-qrun-backup-20261019T081850Z/audit-existing-qrun.py | 框架侧 | 完整工作流采集、性能比较或既有产物核验工具；不计作训练算子。 |
| /workspace/z30084427/lightgbm-direct-qrun-backup-20261019T081850Z/deploy.py | 框架侧 | 安装、部署或回滚工具；不计作训练库生产源文件。 |
| /workspace/z30084427/lightgbm-direct-qrun-backup-20261019T081850Z/verify-full-qrun.py | 框架侧 | 完整工作流采集、性能比较或既有产物核验工具；不计作训练算子。 |
| /workspace/z30084427/lightgbm-npu-acceptance-20261019/default-qrun/validate-artifacts.py | 框架侧 | 完整工作流采集、性能比较或既有产物核验工具；不计作训练算子。 |
| /workspace/z30084427/lightgbm-npu-acceptance-20261019/profile-review/analyze-qlib-artifacts.py | 框架侧 | 完整工作流采集、性能比较或既有产物核验工具；不计作训练算子。 |
| /workspace/z30084427/lightgbm-npu-backup-20261017T170758Z/deploy-validated-library.py | 框架侧 | 安装、部署或回滚工具；不计作训练库生产源文件。 |
| /workspace/z30084427/lightgbm-npu-backup-20261017T170758Z/restore-old-version.py | 框架侧 | 安装、部署或回滚工具；不计作训练库生产源文件。 |
| /workspace/z30084427/lightgbm-npu-backup-20261017T170758Z/validate-default-install.py | 框架侧 | 完整工作流采集、性能比较或既有产物核验工具；不计作训练算子。 |
| /workspace/z30084427/lightgbm-round-comparison-20261019/collect-comparison.py | 框架侧 | 完整工作流采集、性能比较或既有产物核验工具；不计作训练算子。 |
| /workspace/z30084427/lightgbm-round-comparison-20261019/qrun-entry.py | 框架侧 | 完整工作流采集、性能比较或既有产物核验工具；不计作训练算子。 |

## 8. 排除但已核对的其他 Qlib 变更（逐文件）

这些是相对原始 Qlib 的真实其他变更，但与本报告两种树模型迁移无直接归因；明确列出，避免漏项与错误归因。三个公共设备适配文件已在第 6 节单列。

| 仓库 | 状态 | 绝对路径 | 排除理由 |
| --- | --- | --- | --- |
| qlib | A | /workspace/qlib/examples/qlib_AscendC_GRU.sh | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib | A | /workspace/qlib/examples/qrun_npu.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib | A | /workspace/qlib/examples/replace_gru_npu.sh | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib | A | /workspace/qlib/qlib/111.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib | M | /workspace/qlib/qlib/contrib/model/pytorch_adarnn.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib | M | /workspace/qlib/qlib/contrib/model/pytorch_add.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib | M | /workspace/qlib/qlib/contrib/model/pytorch_alstm.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib | M | /workspace/qlib/qlib/contrib/model/pytorch_alstm_ts.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib | M | /workspace/qlib/qlib/contrib/model/pytorch_gats.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib | M | /workspace/qlib/qlib/contrib/model/pytorch_gats_ts.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib | M | /workspace/qlib/qlib/contrib/model/pytorch_general_nn.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib | M | /workspace/qlib/qlib/contrib/model/pytorch_gru.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib | M | /workspace/qlib/qlib/contrib/model/pytorch_gru_ts.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib | M | /workspace/qlib/qlib/contrib/model/pytorch_hist.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib | M | /workspace/qlib/qlib/contrib/model/pytorch_igmtf.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib | M | /workspace/qlib/qlib/contrib/model/pytorch_krnn.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib | M | /workspace/qlib/qlib/contrib/model/pytorch_localformer.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib | M | /workspace/qlib/qlib/contrib/model/pytorch_localformer_ts.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib | M | /workspace/qlib/qlib/contrib/model/pytorch_lstm.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib | M | /workspace/qlib/qlib/contrib/model/pytorch_lstm_ts.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib | M | /workspace/qlib/qlib/contrib/model/pytorch_nn.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib | M | /workspace/qlib/qlib/contrib/model/pytorch_sandwich.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib | M | /workspace/qlib/qlib/contrib/model/pytorch_sfm.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib | M | /workspace/qlib/qlib/contrib/model/pytorch_tabnet.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib | M | /workspace/qlib/qlib/contrib/model/pytorch_tcn.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib | M | /workspace/qlib/qlib/contrib/model/pytorch_tcn_ts.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib | M | /workspace/qlib/qlib/contrib/model/pytorch_tcts.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib | M | /workspace/qlib/qlib/contrib/model/pytorch_tra.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib | M | /workspace/qlib/qlib/contrib/model/pytorch_transformer.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib | M | /workspace/qlib/qlib/contrib/model/pytorch_transformer_ts.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib | A | /workspace/qlib/qlib/contrib/model/torch_device.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib | M | /workspace/qlib/qlib/data/dataset/__init__.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib | M | /workspace/qlib/tests/security/test_workflow_artifact_trust.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib | ? | /workspace/qlib/qlib/contrib/model/infer_utils.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib | ? | /workspace/qlib/qlib/contrib/model/npu_rnn.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib-xgb-npu | A | /workspace/qlib-xgb-npu/examples/qlib_AscendC_GRU.sh | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib-xgb-npu | A | /workspace/qlib-xgb-npu/examples/qrun_npu.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib-xgb-npu | A | /workspace/qlib-xgb-npu/examples/replace_gru_npu.sh | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib-xgb-npu | A | /workspace/qlib-xgb-npu/qlib/111.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib-xgb-npu | M | /workspace/qlib-xgb-npu/qlib/contrib/model/pytorch_adarnn.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib-xgb-npu | M | /workspace/qlib-xgb-npu/qlib/contrib/model/pytorch_add.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib-xgb-npu | M | /workspace/qlib-xgb-npu/qlib/contrib/model/pytorch_alstm.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib-xgb-npu | M | /workspace/qlib-xgb-npu/qlib/contrib/model/pytorch_alstm_ts.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib-xgb-npu | M | /workspace/qlib-xgb-npu/qlib/contrib/model/pytorch_gats.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib-xgb-npu | M | /workspace/qlib-xgb-npu/qlib/contrib/model/pytorch_gats_ts.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib-xgb-npu | M | /workspace/qlib-xgb-npu/qlib/contrib/model/pytorch_general_nn.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib-xgb-npu | M | /workspace/qlib-xgb-npu/qlib/contrib/model/pytorch_gru.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib-xgb-npu | M | /workspace/qlib-xgb-npu/qlib/contrib/model/pytorch_gru_ts.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib-xgb-npu | M | /workspace/qlib-xgb-npu/qlib/contrib/model/pytorch_hist.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib-xgb-npu | M | /workspace/qlib-xgb-npu/qlib/contrib/model/pytorch_igmtf.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib-xgb-npu | M | /workspace/qlib-xgb-npu/qlib/contrib/model/pytorch_krnn.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib-xgb-npu | M | /workspace/qlib-xgb-npu/qlib/contrib/model/pytorch_localformer.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib-xgb-npu | M | /workspace/qlib-xgb-npu/qlib/contrib/model/pytorch_localformer_ts.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib-xgb-npu | M | /workspace/qlib-xgb-npu/qlib/contrib/model/pytorch_lstm.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib-xgb-npu | M | /workspace/qlib-xgb-npu/qlib/contrib/model/pytorch_lstm_ts.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib-xgb-npu | M | /workspace/qlib-xgb-npu/qlib/contrib/model/pytorch_nn.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib-xgb-npu | M | /workspace/qlib-xgb-npu/qlib/contrib/model/pytorch_sandwich.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib-xgb-npu | M | /workspace/qlib-xgb-npu/qlib/contrib/model/pytorch_sfm.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib-xgb-npu | M | /workspace/qlib-xgb-npu/qlib/contrib/model/pytorch_tabnet.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib-xgb-npu | M | /workspace/qlib-xgb-npu/qlib/contrib/model/pytorch_tcn.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib-xgb-npu | M | /workspace/qlib-xgb-npu/qlib/contrib/model/pytorch_tcn_ts.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib-xgb-npu | M | /workspace/qlib-xgb-npu/qlib/contrib/model/pytorch_tcts.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib-xgb-npu | M | /workspace/qlib-xgb-npu/qlib/contrib/model/pytorch_tra.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib-xgb-npu | M | /workspace/qlib-xgb-npu/qlib/contrib/model/pytorch_transformer.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib-xgb-npu | M | /workspace/qlib-xgb-npu/qlib/contrib/model/pytorch_transformer_ts.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib-xgb-npu | A | /workspace/qlib-xgb-npu/qlib/contrib/model/torch_device.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |
| qlib-xgb-npu | M | /workspace/qlib-xgb-npu/tests/security/test_workflow_artifact_trust.py | 原始 Qlib 后已有的其他模型/安全/数据集等变更，不归因于本次 LightGBM/XGBoost 迁移。 |

## 9. 既有完整验收与尚未达成项

以下为已归档的 2026-10-19 完整 Alpha158 验收，不宣称是此后所有手动运行中的最后一次；本报告未重新训练。两者均为158特征、261207条测试预测，原数据区间和 early stopping 保留。

| 项目 | LightGBM | XGBoost |
| --- | --- | --- |
| 普通 qrun 墙钟 | 224.497631 秒 | 最新归档直接入口 343.920998 秒；此前隔离入口 336.637397 秒 |
| 实际评估轮数 | 73 | 73 |
| 最佳迭代（从1计数） | 23 | 23（属性从0计数为22） |
| 保存树数 | 23（最佳模型） | 73（原 Qlib 使用全部已训练树） |
| IC / Rank IC | 0.0470302 / 0.0486943 | 0.0608911 / 0.0515282 |
| 对齐结论 | 对 CPU 参考 21处树字段差异，预测最大差约8.33e-17；production_acceptance=false | 对已验收 NPU 参考树和预测一致；best_score元数据差2.22e-16，文件字节不同；非CUDA等价 |

XGBoost 用户 A100 的 89.157秒/66棵树目标尚未达到；不同树数/模型和单次耗时不可当作严格同工作量加速比。LightGBM 分类路径、量化/单机能力和生产验收限制须保留。完整 NPU qrun、保存加载成功、采样组件参考一致，不等于训练全链路 CUDA 等价。

具体证据路径：
- `/workspace/z30084427/lightgbm-direct-qrun-backup-20261019T081850Z/default-full-qrun/acceptance.json`
- `/workspace/xgb-npu-runs/explicit-npu-qrun-verification-20261019-v1/execution.json`
- `/workspace/xgb-npu-runs/explicit-npu-qrun-verification-20261019-v1/artifact-verification.json`
- `/workspace/xgb-npu-runs/cuda-alignment-comparison-20261019/alignment-audit.json`

## 10. 交付物与校验

- `inventory.csv` / `inventory.json`：第6节完整机器可读清单，含分类、角色、原始/当前哈希和差分行数。
- `LightGBM-vs-original.patch` / `xgboost-npu-vs-original.patch`：原始库到当前完整迁移净差分（包含测试）。
- `qlib-vs-original.patch` / `qlib-xgb-npu-vs-original.patch`：第6节纳入文件差分，公共及副本归属需按清单区分。
- `system-qrun-entry.patch`：系统入口对备份原始入口差分；不是以 Qlib 根目录直接应用的补丁。
- `external-support.csv/json`、`deployment.json`：外部辅助源码、部署文件和产物身份。
- `excluded-unrelated-qlib.csv/json`：全部排除的其他 Qlib 变更及理由。
- `summary.json` / `verification.json`：基线、统计、覆盖检查及四份仓库补丁反向检查结果。
- `accepted-run-facts.json`：此前验收数字和证据路径副本。
- `generate_report.py`：本报告可复核生成器；不属于迁移源码，不计入第6节。

覆盖检查要求：各仓库原始 diff + 限定源码路径未跟踪文件 = 已纳入清单 + 有理由排除清单；缺失/额外均为空。四份仓库补丁已对当前树做 git apply --reverse --check（只检查，未实际应用），所有纳入源码哈希已复查。系统入口补丁只用于对照，不声称通过同一仓库 git apply 检查。

本次操作仅生成报告和差分；未修改源代码、未编译、未启动训练、未访问 NPU、未提交 Git。报告不等同于完整二进制/依赖可搬迁归档，也不会把原有 CUDA 等价任务或生产验收任务标记完成。
