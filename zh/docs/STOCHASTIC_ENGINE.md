# 随机计算引擎

## 概念

带常驻因果状态的自回归模型是一个昂贵的受条件转移算子。引擎把一个常驻受条件状态作为一片
GPU 常驻随机场的初始条件：M 个独立分支由此分叉，逐步推进、剪枝、压缩、观测、归约——无需把
每条轨迹导出到 CPU。流水线：

    常驻受条件状态 -> GPU 分叉 -> 逐分支计数器 RNG -> 场推进
    -> 生存/压缩 -> 观测量求值 -> 归约 -> 小结果返回主机

“Kolmogorov”与“Dumitrescu”是项目对两个互补原语的命名：Kolmogorov 推进受条件随机场；
Dumitrescu 求值与归约该场。刻意保持生成与归约可分，即便后续内核做了融合。

## 分支状态模型

每分支：KV 追加行、GDN 循环状态、卷积环、索引池、位置、当前 token、对数权重、观测量累加器、
存活标志。共享：父状态的不可变历史（子分支只追加；父行从不改写——由分叉独立性回执证明）。
RNG 状态完全不存储：流是计数器式的，以 (seed, branch, step, lane) 寻址。分支描述符 16 B；
100 万分支的状态创建 0.08 ms，当前规模无需写时复制。

## 组件

### RNG（stoch/include/philox.cuh）

Philox4x32-10，按项目冻结的 RNG 规范实现：key=(slot,round)；
counter=(position,branch,purpose,index)；uniform=(word0>>8)*2^-24。十轮，除最后一轮外每轮
之后做 Weyl 键增量。已知答案测试：counter=key=0 -> 6627e8d5 e169c58d bc57ac4c 9b00dbd8
（主机与设备）。另测：确定性、2,048 个单元间的流独立性、计数器快照/恢复、100 万均匀样本的
均值/方差。回执：receipts/RNG_RECEIPT_20260907.md。

### Dumitrescu 归约（stoch/reduce/dumi.cuh）

单遍加权统计：N、Σw、Σw²、Σw f_k、Σw f_k²、max log-w、log-sum-exp(log w)、
ESS=(Σw)²/Σw²，每分支 K 个观测量。fp64 累加；每块写一个部分和；主机以双精度合并部分和
（确定性，无跨块原子操作）。零权哨兵 lw <= -1e300。14 个用例对照 CPU fp64 参考验证（奇数
规模、微小/巨大/零权重、多观测量）。回执：receipts/DUMI_RECEIPT_20260907.md。

### Kolmogorov 场推进（stoch/field/kolmo_field.cu）

合成场：受条件父状态 x0 分叉为 M 个 Ornstein-Uhlenbeck 动力学分支 x <- a*x + sigma*xi，
噪声为计数器 RNG 的 Box-Muller。每步：推进，然后击杀硬币（p=0.01）以权重补偿 1/(1-p)
（无偏俄罗斯轮盘）剪枝，随后排他扫描压缩幸存者。观测量逐步累加，最终归约一次。三个闭式门：
E[X_T] = a^T x0；E[X_T²] = (a^T x0)² + sigma²(1-a^{2T})/(1-a²)；E[(1/T) Σ x_t²]。三者均在
期望蒙特卡洛误差内吻合；生存率 52.5% 与击杀硬币预测 0.99^64 = 52.6% 一致。回执：
receipts/KOLMO_FIELD_RECEIPT_20260907.md。

### 模型支撑场（MC-C）

同一机制作用于常驻模型：一个已灌注的 382-token 父状态（8/40 切分）分叉为 M=8 行。第 0 行
为贪婪对照；第 1-7 行按温度 1 的 Gumbel-max 采样转移（服务端 k_sample_rows，确定性
splitmix64，以 (seed,row,v) 为键）。16 步后，读回每行状态签名（sigR = Spark 各层
GDN+卷积状态之和；sigK = 最近 4 个写入位置 K/V 之和，经增量开关 QF_M8_FORK_SIG），并在
7 个采样行上做无权归约：场均值 sigR +228 vs 贪婪行 sigR -704，样本标准差 1076，ESS 7；末端
token 分布 318 x3、264、19、8260、53235、156566。场以 21.8 兄弟 token/s 执行（367.5 ms/步）；
全词表 Gumbel 循环约 110 ms/步，是首要优化目标。重复运行按位复现了 token 轨迹（签名值在
记录的 M 行算术带内波动）。回执：receipts/MCC_MODEL_FIELD_RECEIPT_20260907.md。

sigR 的解释有限：它是分支循环状态的确定性泛函，用作轨迹观测量；不主张更宽泛的统计解释。
M>8 未验证（见 docs/LIMITATIONS.md）。

## 吞吐

合成场，T=64 步，GB10，seed 42：M=1,024 时 2.6e7 轨迹步/秒；16,384 时 3.8e8；262,144 时
1.52e9；1,048,576 时 1.60e9。约 16K 分支以下受发射限制；约 262K 以上饱和。单位：轨迹步/秒
（M 个分支各前进一步）；不是完成轨迹/s，也不是 token 吞吐。回执：
receipts/THROUGHPUT_LAW_RECEIPT_20260907.md。

## 渲染

渲染指可视化受条件随机过程生成的统计结构。场的末端状态对（X_T, X_{T-1}）在 GPU 上分箱为
512x512 密度网格（每轴 -3..3）并写为 PPM。参数：M=1,048,576；T=64；a=0.50、sigma=1.00；
x0=1.00；Philox seed 42；p_kill=0.01 带权重补偿；推进用时 44.7 ms；约 1.5e9 轨迹步/秒；
ESS 550,847。工件：figures/kolmo_field_density.ppm（及无损 PNG）。它演示的是通用链条——
随机样本 -> 场演化 -> 观测量 -> 归约/渲染——而非图形管线。
