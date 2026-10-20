import { Stack, Row, Grid, H1, H2, Text, Table, Callout, Card, CardHeader, CardBody, useHostTheme, useCanvasAction, Button } from "cursor/canvas";

const base = "/workspace/z30084427/lightgbm-round-comparison-20261019";
const xb = "/workspace/xgb-npu-runs/cuda-alignment-comparison-20261019";
const lgbProfile = base + "/profile-after-fused/raw/PROF_000001_20261019070826589_00671324CABJARMH/mindstudio_profiler_output/op_statistic_20261019071225.csv";
type Op = { name: string; cores: string; count: number; seconds: number };
const lgbOps: Op[] = [
  {name: "LGBMNPUFusedHistogramKernel_interface", cores: "AI_CPU", count: 4700, seconds: 138.268880011},
  {name: "LGBMNPUFusedSplitKernel_interface", cores: "AI_CPU", count: 9327, seconds: 6.742905722},
  {name: "ScatterElements", cores: "AI_CPU", count: 6441, seconds: 2.260723557},
  {name: "Sub", cores: "AI_CPU / AI_VECTOR_CORE", count: 21851, seconds: 1.466243036},
  {name: "Cumsum", cores: "AI_CPU / AI_VECTOR_CORE / MIX_AIV", count: 10856, seconds: 1.063297499},
  {name: "ArgMax", cores: "AI_CPU", count: 6441, seconds: 0.558038113},
  {name: "BroadcastTo", cores: "AI_CPU", count: 6441, seconds: 0.331898858},
  {name: "GatherV2", cores: "AI_VECTOR_CORE", count: 10856, seconds: 0.063360843},
  {name: "Cast", cores: "AI_VECTOR_CORE", count: 12809, seconds: 0.032720041},
  {name: "ScatterElementsV2", cores: "MIX_AIV", count: 6368, seconds: 0.029202680},
  {name: "Equal", cores: "AI_VECTOR_CORE", count: 12495, seconds: 0.025705698},
  {name: "LogicalOr", cores: "AI_VECTOR_CORE", count: 10491, seconds: 0.019789614},
  {name: "SelectV2", cores: "AI_VECTOR_CORE", count: 6368, seconds: 0.016606132},
  {name: "Add", cores: "AI_VECTOR_CORE", count: 6368, seconds: 0.016376908},
  {name: "LessEqual", cores: "AI_VECTOR_CORE", count: 6368, seconds: 0.015090481},
  {name: "TensorMove", cores: "AI_VECTOR_CORE", count: 577, seconds: 0.006401213},
  {name: "LogicalNot", cores: "AI_VECTOR_CORE", count: 1236, seconds: 0.003331965},
  {name: "LogicalAnd", cores: "AI_VECTOR_CORE", count: 1236, seconds: 0.002778868},
];
const xgbOps: Op[] = [
  {name: "IntegerHistogramBuildKernel", cores: "MIX_AIC", count: 8381, seconds: 144.801699047},
  {name: "ScatterElementsV2", cores: "MIX_AIV", count: 38847, seconds: 62.593062593},
  {name: "BatchedSplitScoreKernel", cores: "MIX_AIC", count: 16689, seconds: 17.534224666},
  {name: "BatchedSplitSelectKernel", cores: "MIX_AIC", count: 16689, seconds: 10.619962964},
  {name: "BatchedSplitPrefixKernel", cores: "MIX_AIC", count: 16689, seconds: 6.806689384},
  {name: "BatchedSplitFinishKernel", cores: "MIX_AIC", count: 16689, seconds: 1.003575079},
  {name: "NativePredictKernel", cores: "MIX_AIC", count: 147, seconds: 0.692384559},
  {name: "Cumsum", cores: "AI_VECTOR_CORE / MIX_AIV", count: 60786, seconds: 0.474055592},
  {name: "GatherV2", cores: "AI_VECTOR_CORE", count: 62242, seconds: 0.332968826},
  {name: "ReduceSum", cores: "MIX_AIV / AI_VECTOR_CORE", count: 74938, seconds: 0.307243074},
  {name: "AxpyV2", cores: "AI_VECTOR_CORE", count: 95332, seconds: 0.304228387},
  {name: "Cast", cores: "AI_VECTOR_CORE", count: 86890, seconds: 0.282680610},
  {name: "TensorMove", cores: "AI_VECTOR_CORE", count: 11257, seconds: 0.178272053},
  {name: "SelectV2", cores: "AI_VECTOR_CORE", count: 63560, seconds: 0.152728333},
  {name: "Greater", cores: "AI_VECTOR_CORE", count: 33394, seconds: 0.111089306},
  {name: "Less", cores: "AI_VECTOR_CORE", count: 48481, seconds: 0.109355749},
  {name: "Add", cores: "AI_VECTOR_CORE", count: 38561, seconds: 0.107979233},
  {name: "LogicalNot", cores: "AI_VECTOR_CORE", count: 28611, seconds: 0.057955438},
  {name: "NativeSampleRowsKernel", cores: "MIX_AIC", count: 73, seconds: 0.047182899},
  {name: "NativeBinRowsKernel", cores: "MIX_AIC", count: 1, seconds: 0.047006550},
  {name: "LogicalOr", cores: "AI_VECTOR_CORE", count: 26138, seconds: 0.042799279},
  {name: "Equal", cores: "AI_VECTOR_CORE", count: 16328, seconds: 0.033659827},
  {name: "LogicalAnd", cores: "AI_VECTOR_CORE", count: 14035, seconds: 0.023508826},
  {name: "RightShift", cores: "AI_VECTOR_CORE", count: 2190, seconds: 0.019943822},
  {name: "BitwiseAnd", cores: "AI_VECTOR_CORE", count: 1752, seconds: 0.010563932},
  {name: "NativeSquareErrorKernel", cores: "MIX_AIC", count: 73, seconds: 0.009888308},
  {name: "ClipByValueV2", cores: "AI_VECTOR_CORE", count: 584, seconds: 0.007471266},
  {name: "Mul", cores: "AI_VECTOR_CORE", count: 292, seconds: 0.002435684},
];
const metricRows = [
  ["端到端 qrun（秒）", "220.740", "336.637"],
  ["已单独记录的训练耗时（秒）", "175.935", "该结果未单独记录，不由墙钟推算"],
  ["IC（预测与收益相关系数）", "0.047030", "0.060891"],
  ["Rank IC（秩相关系数）", "0.048694", "0.051528"],
  ["ICIR / Rank ICIR（原始日志值）", "0.381641 / 0.405690", "0.435843 / 0.411568"],
  ["扣费后超额年化收益（%）", "11.0624", "7.1286"],
  ["扣费后超额信息比率", "1.305132", "0.805570"],
  ["扣费后超额最大回撤（%）", "−8.5821", "−10.3409"],
  ["未扣费超额年化收益（%）", "14.7334", "10.5224"],
  ["未扣费超额信息比率", "1.737630", "1.188975"],
  ["未扣费超额最大回撤（%）", "−7.8346", "−9.5045"],
  ["保存模型树数 / 最佳迭代", "23 / 23（从 1 开始）", "73 / 22（从 0 开始）"],
];
const migrationRows = [
  ["框架集成", "NPUTreeLearner 与 device=npu 路由；保留 Qlib 原始非量化 Alpha158 配置", "device=npu:0、tree_method=hist、grow_npu_hist 与原生预测路由；接入独立 Python 环境"],
  ["数据准备与分箱", "数据准备/初始分箱由主机组织，分箱列、梯度与叶行驻留设备", "CPU SketchOnDMatrix 生成切分点；NativeBinRowsKernel 在 NPU 编码分箱并缓存"],
  ["目标函数与采样", "梯度/Hessian 由主机生成并上传，保留 boosting 编排和采样控制", "平方误差梯度、行采样和梯度量化在设备；初始截距与小特征集列采样仍在主机"],
  ["直方图构建", "自定义融合 AICPU 算子：FP32 梯度/Hessian 转 FP64 累加，融合原 gather/cast/scatter 链", "自定义整数直方图内核，梯度量化和直方图统计在设备执行，复用不可变分箱缓存"],
  ["数值分裂搜索", "融合 AICPU 阈值扫描、增益计算和赢家选择，回传选中结果；类别路径存在主机兼容逻辑", "设备批处理前缀、精确评分、候选选择与结果收尾；主机负责树生长调度及选中节点元数据"],
  ["叶行划分与更新", "设备 INT32 前缀和、scatter 划分驻留叶行，训练分数更新在设备；主机维护兼容布局", "设备 gather / prefix / scatter 维护行成员和左右子节点；复用缓冲区与批量调度"],
  ["模型预测", "训练分数更新已用设备算子；一般模型推理仍在主机逐树遍历，不能称通用推理已迁移", "NativePredictKernel 执行普通 DMatrix 标量树预测，接入预测缓存及标准模型保存/加载"],
  ["验证闭环", "指定 220.740 秒运行通过 74 项产物检查；融合开关对照全量预测差为 0、树结构一致", "原生训练/预测/保存加载和完整 qrun 通过；UBJ/JSON/raw 重载预测全量一致"],
];
const lgbRoles: Record<string, string> = {
  LGBMNPUFusedHistogramKernel_interface: "融合特征取值、FP64 梯度/Hessian 直方图累加",
  LGBMNPUFusedSplitKernel_interface: "数值阈值搜索、增益计算、最优分裂选择",
};
const xgbRoles: Record<string, string> = {
  IntegerHistogramBuildKernel: "量化梯度/Hessian 的整数直方图累加",
  BatchedSplitPrefixKernel: "分裂候选前缀统计",
  BatchedSplitScoreKernel: "分裂候选精确评分",
  BatchedSplitSelectKernel: "最优候选选择",
  BatchedSplitFinishKernel: "选中分裂摘要与结果收尾",
  NativePredictKernel: "原生树模型批量预测",
  NativeSampleRowsKernel: "均匀行采样及梯度掩码",
  NativeBinRowsKernel: "按已有切分点进行分箱编码",
  NativeSquareErrorKernel: "平方误差梯度/Hessian 计算",
};
const sources = [
  ["LightGBM 指定 3m40.740 计时", base + "/previous-timing-comparison.json"],
  ["LightGBM 指定运行指标及 74 项检查", base + "/previous-histogram-artifacts.json"],
  ["LightGBM 融合对照预测一致性", base + "/previous-fused-parity.json"],
  ["LightGBM 独立采样算子原表", lgbProfile],
  ["LightGBM 采样加载库身份", base + "/profile-after-fused/loaded-environment.json"],
  ["LightGBM 采样运行元数据", base + "/profile-after-fused/time.json"],
  ["XGBoost 全量指标与版本比较", xb + "/comparison.json"],
  ["XGBoost 原始算子对照表", xb + "/op_comparison.csv"],
  ["XGBoost CUDA 对齐审计与剩余项", xb + "/alignment-audit.json"],
  ["LightGBM 设备与主机边界源码", "/workspace/LightGBM/src/treelearner/npu_tree_learner.cpp"],
  ["XGBoost 原生树更新源码", "/workspace/xgboost-npu/src/tree/updater_npu_hist.cc"],
  ["XGBoost 预测能力边界源码", "/workspace/xgboost-npu/src/predictor/npu_predictor.cc"],
];
function operatorRows(ops: Op[]) {
  const total = ops.reduce((s, op) => s + op.seconds, 0);
  return ops.map(op => [op.name, op.cores, op.count.toLocaleString("en-US"), op.seconds.toFixed(6), (op.seconds * 100 / total).toFixed(3) + "%", (op.seconds * 1e6 / op.count).toFixed(3)]);
}
function kernelRows(ops: Op[], roles: Record<string, string>) {
  return ops.filter(op => roles[op.name]).map(op => [op.name, roles[op.name], op.cores, op.count.toLocaleString("en-US"), op.seconds.toFixed(6)]);
}

export default function QrunMigrationSummary() {
  const theme = useHostTheme();
  const dispatch = useCanvasAction();
  return <Stack gap={24} style={{padding: 24, maxWidth: 1440, margin: "0 auto", color: theme.text.primary, background: theme.bg.editor}}>
    <Stack gap={8}>
      <Text size="small" tone="secondary">工作汇报 · 截至 2026-10-19 · 已有运行证据汇总 · 本次未启动设备任务</Text>
      <H1>LightGBM / XGBoost 昇腾 NPU 迁移进展</H1>
      <Text weight="semibold">两模型已在完整 Alpha158 工作负载打通训练、预测、信号分析、回测与产物验证。当前阶段成果是关键训练链路迁移和端到端功能闭环；主机保留部分逻辑，尚非全功能、全设备实现。</Text>
    </Stack>
    <Grid columns="repeat(auto-fit, minmax(300px, 1fr))" gap={18}>
      <Card><CardHeader>LightGBM · 指定汇报基准</CardHeader><CardBody><Stack gap={8}>
        <H2>3m40.740 / 220.740 秒</H2>
        <Text>采用 timing-histogram 运行；训练耗时 175.935 秒；74/74 项产物检查通过。开启融合直方图与融合分裂，最终保留 23 棵树。</Text>
        <Text tone="secondary">IC 0.047030 · 扣费后超额年化 11.0624% · 信息比率 1.305132</Text>
      </Stack></CardBody></Card>
      <Card><CardHeader>XGBoost · CUDA 逻辑对齐 v1</CardHeader><CardBody><Stack gap={8}>
        <H2>5m36.637 / 336.637 秒</H2>
        <Text>采用 final-alpha158-npu-cuda-alignment-20261019-v1；qrun 与原生 NPU 后端验收通过。最终 73 棵树，预测按既有 Qlib 逻辑使用全部树。</Text>
        <Text tone="secondary">IC 0.060891 · 扣费后超额年化 7.1286% · 信息比率 0.805570</Text>
      </Stack></CardBody></Card>
    </Grid>
    <Callout tone="info">汇报口径已修正：LightGBM 主结果仅用 220.740 秒那次数据及其 175.935 秒训练耗时。其他运行的 243.706 秒总耗时、198.209 秒训练耗时均不作为本报告主结果。qrun 是 Qlib 的端到端流程，包含数据准备、训练、预测、信号分析及回测。</Callout>

    <Stack gap={8}>
      <H2>01 · 工作负载与主要结果</H2>
      <Text>Alpha158 / 158 特征 / CSI300 股票池 / SH000300 基准；训练 2008-01-01—2014-12-31，验证 2015-01-01—2016-12-31。测试配置 2017-01-01—2020-08-01，实际预测覆盖 2017-01-03—2020-07-31，共 871 个交易日、261,207 条预测；未裁剪测试范围。</Text>
      <Table headers={["指标（单位 / 定义）", "LightGBM · 3m40.740 那次", "XGBoost · 10-19 v1"]} rows={metricRows} columnAlign={["left", "right", "right"]} striped />
      <Text size="small" tone="secondary">来源：LightGBM previous-histogram-artifacts.json / execution、qrun_log、required_mlflow_metrics；XGBoost comparison.json / runs.after。收益与回撤均为超额口径、百分比按原值乘 100；ICIR 是原始日志值，不另行年化。不同模型与预测树数语义不支持直接推导模型间加速比。</Text>
    </Stack>

    <Stack gap={8}>
      <H2>02 · 迁移完成情况与 CPU / NPU 边界</H2>
      <Table headers={["模块", "LightGBM", "XGBoost"]} rows={migrationRows} striped />
      <Text size="small" tone="secondary">来源：两模型原生树更新、运行时、预测源码及保存的验收产物。这里的“设备”含 NPU 板上 AICPU 和加速核心；不等于全部在向量核心执行。源码存在某能力不等于已通过全部配置的端到端验收。</Text>
    </Stack>

    <Stack gap={10}>
      <H2>03 · 算子规模与热点结论</H2>
      <Table headers={["独立性能采样统计", "LightGBM · 同融合库", "XGBoost · 10-19 v1"]} rows={[
        ["合并 Core Type 后的 OP Type 数", "18 类", "28 类"],
        ["原表“算子 × 核心”记录数", "21 条", "30 条"],
        ["设备算子调用总次数", "141,229", "779,649"],
        ["设备算子累计任务耗时（秒）", "150.923351", "246.714615"],
        ["采样中出现的自定义内核类型", "2 类融合 AICPU 内核", "9 类自定义内核"],
        ["首要设备热点", "融合直方图 91.615%", "整数直方图 58.692% + ScatterElementsV2 25.371%"],
      ]} columnAlign={["left", "right", "right"]} striped />
      <Callout tone="warning">算子耗时来自独立 msprof 性能采样，均不是上述非采样 qrun 的阶段拆分。占比的分母是该采样的设备算子累计时长，不能用 150.923 / 220.740 或 246.715 / 336.637 当作设备利用率，也不能叠加主机 API 时长。LightGBM 的 AI_CPU 表示设备侧 AICPU，不是主机 CPU，更不是向量核心。</Callout>
      <Text>LightGBM 采样墙钟为 260.786 秒，clean_device=true；实际加载库 SHA256 与 220.740 秒基准相同，融合开关一致，但不是同次运行且 Qlib 加载路径不同。两份重复导出的算子统计 CSV 经 SHA256 比对字节相同，只计一份；旧库 341.929 秒算子统计未混入。</Text>
      <Text>LightGBM 的设备算子时间主要落在 AICPU：150.589480 秒；向量核心与 MIX_AIV 合计 0.333871 秒。XGBoost 的 MIX_AIC 为 181.562613 秒、MIX_AIV 为 62.967935 秒、AI_VECTOR_CORE 为 2.184067 秒。以上为 profiler 原始分类，不把混合核心标签解释为某个物理核心独占执行。</Text>
    </Stack>

    <Stack gap={8}>
      <H2>04 · 自定义算子与职责</H2>
      <Text weight="semibold">LightGBM：融合直方图与融合数值分裂</Text>
      <Table headers={["自定义算子（精确名称）", "职责", "采样执行单元", "调用次数", "累计耗时（秒）"]} rows={kernelRows(lgbOps, lgbRoles)} columnAlign={["left", "left", "left", "right", "right"]} />
      <Text weight="semibold">XGBoost：分箱、采样、目标函数、直方图、批量分裂、预测</Text>
      <Table headers={["自定义算子（精确名称）", "职责", "采样执行单元", "调用次数", "累计耗时（秒）"]} rows={kernelRows(xgbOps, xgbRoles)} columnAlign={["left", "left", "left", "right", "right"]} />
      <Text size="small" tone="secondary">来源：2026-10-19 独立 msprof op_statistic 与对应源码；仅统计采样实际出现的自定义 OP Type，不代表源码里全部内核数量。XGBoost 的 ScatterElementsV2 为通用算子，不纳入自定义内核数量。</Text>
    </Stack>

    <Grid columns="repeat(auto-fit, minmax(320px, 1fr))" gap={24}>
      <Stack gap={8}>
        <H2>05 · 已完成优化与验证</H2>
        <Text weight="semibold">LightGBM</Text>
        <Text>同库、原始工作负载单次开关对照：仅融合分裂 311.957 秒 → 融合直方图与分裂 220.740 秒，减少 91.218 秒（29.24%）。全测试集 261,207 条预测最大绝对差 0，树结构一致，最佳迭代均为 23。结果描述为本次对照观察，不泛化为相对 CPU / CUDA 的正式加速比。</Text>
        <Text weight="semibold">XGBoost</Text>
        <Text>完成分箱及分配缓存复用、批量分裂驱动、驻留行成员与预测缓存；本批对齐行/列采样、随机数推进、预测累加顺序及 depthwise 最大 1024 节点批大小。原生训练、预测、标准模型保存加载与完整 qrun 验证形成闭环。</Text>
        <Text>10-17 v4 → 10-19 v1 的单次墙钟 342.667 → 336.637 秒，减少 1.76%，但树数 74 → 73；IC 上升 0.001750，扣费后超额年化下降 1.9897 个百分点。跨版本预测相关约 0.8960，不声称数值结果不变或收益无回退。</Text>
      </Stack>
      <Stack gap={8}>
        <H2>06 · 当前边界与待完成项</H2>
        <Text weight="semibold">LightGBM</Text>
        <Text>本次验收是非量化 Alpha158。公开 NPU 量化训练入口仍拒绝；单调约束、线性树、CEGB、强制分裂及分布式训练未作为已支持能力。类别分裂、refit 等存在主机兼容路径，通用推理仍在主机；生产验收标记仍为 false。</Text>
        <Text>新采集流程在发现两个相同算子导出文件时断言中断，导致完整新旧对照未完成；本报告核对哈希后只读取一份已有 CSV，没有修改采集程序或重启实验。</Text>
        <Text weight="semibold">XGBoost</Text>
        <Text>范围限于单卡、内存内、数值特征、标量平方误差、gbtree；类别/分布式/外存/多目标/加权特征采样，以及原地预测、叶索引与 SHAP 贡献等不在当前原生支持范围。</Text>
        <Text>切分点仍用 CPU SketchOnDMatrix，尚未迁移为 CUDA DeviceSketch 等价算法；初始截距归约、量化尺度边界与编译舍入仍有未对齐或未验证项。尚无冻结相同输入下的 CUDA 逐轮模型/预测等价证明。</Text>
      </Stack>
    </Grid>

    <Stack gap={8}>
      <H2>07 · 下一阶段工作建议</H2>
      <Text>优先级 1：围绕已验证热点优化。LightGBM 优先优化融合直方图的设备 AICPU 循环；向量化/并行化须保持 FP64 累加与预测一致性。XGBoost 优先优化整数直方图与行成员相关 scatter，二者占采样设备算子时间约 84.06%。</Text>
      <Text>优先级 2：固定输入、参数、训练轮数与运行环境，在独占设备上重复计时，补充分位数和方差；LightGBM 对重复导出文件去重后补齐完整前后算子比较。</Text>
      <Text>优先级 3：XGBoost 补齐切分点、初始截距、量化边界与 CUDA 同输入逐轮对照；两模型分别扩展功能回归、打包部署和生产验收，避免把基准通过写成全功能迁移完成。</Text>
    </Stack>

    <Stack gap={8}>
      <H2>附录 A · LightGBM 全部采样算子</H2>
      <Table headers={["OP Type", "Core Type（合并显示）", "调用次数", "累计（秒）", "设备算子时间占比", "平均（微秒/次）"]} rows={operatorRows(lgbOps)} columnAlign={["left", "left", "right", "right", "right", "right"]} striped />
      <Text size="small" tone="secondary">来源：op_statistic_20261019071225.csv，同库独立采样，2026-10-19。按 OP Type 合并不同 Core Type，平均耗时按总微秒 / 总次数加权计算；所有占比分母为 150.923351239 秒。保留原始 CSV 以便追溯不同核心记录。</Text>
    </Stack>
    <Stack gap={8}>
      <H2>附录 B · XGBoost 全部采样算子</H2>
      <Table headers={["OP Type", "Core Type（合并显示）", "调用次数", "累计（秒）", "设备算子时间占比", "平均（微秒/次）"]} rows={operatorRows(xgbOps)} columnAlign={["left", "left", "right", "right", "right", "right"]} striped />
      <Text size="small" tone="secondary">来源：op_comparison.csv 的 after 列，原始 op_statistic_20261019053119.csv，2026-10-19 独立采样。按 OP Type 合并不同 Core Type，平均耗时按总微秒 / 总次数计算；所有占比分母为 246.714615286 秒。</Text>
    </Stack>
    <Stack gap={8}>
      <H2>原始证据与源码</H2>
      <Text size="small" tone="secondary">静态工作汇报快照，不随后台新实验自动更新。点击打开对应文件。</Text>
      <Row wrap gap={8}>{sources.map(([label, path]) => <Button key={path} variant="secondary" onClick={() => dispatch({type: "openFile", path})}>{label}</Button>)}</Row>
    </Stack>
  </Stack>;
}