# 架构

## 任务分解

模型被切分到两块加速器上。生产配置下，AMD 层级持有 0..15 层（前缀）、PLE 常量层与 lm_head；
NVIDIA 层级持有 16..47 层（尾部）以及尾部各层的 QSA 索引器。每个解码步为：

1. AMD 前缀：常驻前缀状态针对新 token 前进一步；残差经网络发出。
2. Spark 尾部：尾部各层执行（稠密 FP8 投影、路由 NVFP4 专家、超连接、GDN 循环、适用处的
   QSA 稀疏注意力）；残差返回。
3. AMD 融合输出头：输出 HC、lm_head、argmax；准备下一前缀窗口。

因此每个 token 的关键路径上 AMD 层级出现两次。切分点可配置（QF_LAYER_BEGIN、QF_AMD_PREFIX）。
见 figures/heterogeneous_pipeline.svg。

## 常驻状态

两个层级都保持模型常驻：权重、KV 缓存、GDN 循环状态、卷积环、QSA 索引池与专家槽位映射。
请求不是冷加载。这正是再填充（无重放注入）、续写与分叉有意义的前提；见
docs/PREFILL_REFILL.md。

## 随机覆盖层

同一套常驻状态支撑随机通道：一个已灌注的父状态被复制到两个层级各 M 个行槽中（分叉），然后
作为场推进。各行共享父状态的不可变历史，仅通过各自的分支局部可变状态与独立 RNG 流发散。
见 docs/STOCHASTIC_ENGINE.md。

## 精度

稠密权重为 FP8（E4M3，每 64 块一个尺度）；路由专家在生产 Spark 路径为 NVFP4，在 AMD 生产
层级为 INT8（分叉拓扑中两套 arena 并存）；PLE 常量与 lm_head 为更高精度。精度变更为早期冻结
所关闭，不属于本窗口的工作。
