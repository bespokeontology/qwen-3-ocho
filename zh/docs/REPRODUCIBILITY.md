# 可复现性

## 权威

规范冻结包 FREEZE_QWEN_OVERNIGHT_20260907（内含 SHA256SUMS，三台机器镜像）是本树每个数字的
权威。公开回执为脱敏副本（主机/路径已替换）；原始回执在包内。见 freeze/PROVENANCE.md 与
freeze/MANIFEST.md。

## 校验

    ./scripts/verify_freeze.sh

按公开树 SHA256SUMS 校验公开工件，并打印各组件的来源类别。私有包清单逐字保留在
freeze/package_SHA256SUMS，用于交叉核验规范包本身。

## 独立复现（任意 CUDA GPU ≥ sm_70）

    ./scripts/reproduce_spark_tests.sh         # RNG + Dumitrescu 单元测试
    ./scripts/reproduce_mc.sh                  # MC-A / MC-A2 / MC-A3 / MC-B
    ./scripts/reproduce_stochastic_field.sh    # 合成场 + 密度工件

要求：CUDA ≥ 12（nvcc）；这些目标除 CUDA 工具链外无外部依赖。保留构建面向 sm_121a；脚本
遵循 CUDA_HOME 与 QF_ARCH。每个脚本头注明了期望输出，必须落在回执容差之内。

## 设备复现（双机，冻结二进制）

依赖模型的实验与解码回执需要双机设备与冻结二进制，它们不在本树中。确切环境与配方引用在
回执（脱敏）与规范冻结包中。关键配置：生产切分 QF_LAYER_BEGIN=16、QF_AMD_PREFIX=16、
QF_MAX_CONTEXT=16384、QF_EXPERT_MODE=full、时钟锁定；分叉拓扑 QF_LAYER_BEGIN=8、
QF_AMD_PREFIX=8、QF_M1_INT8_HEAD=0。模型检查点标识：RadixArk-Qwen3.8-Flash-Next-NVFP4
（目录布局以冻结包为准）。冻结生产 AMD 二进制为 824e5c50…；其精确源码树尚未找回
（PENDING-PROVENANCE），因此目前无法从源码重建 AMD 侧。

## 数字到回执的映射（数值审计）

| 声明 | 回执 |
|---|---|
| RNG KAT + 确定性 | RNG_RECEIPT_20260907.md |
| 归约 CPU 参考验证 | DUMI_RECEIPT_20260907.md |
| MC-A / MC-A2 / MC-B 数字 | MC_GATES_RECEIPT_20260907.md |
| 分层 188 倍误差缩减 | VARIANCE_REDUCTION_RECEIPT_20260907.md |
| 场解析门 + 生存率 | KOLMO_FIELD_RECEIPT_20260907.md |
| 吞吐定律数字 | THROUGHPUT_LAW_RECEIPT_20260907.md |
| MC-C 场统计 | MCC_MODEL_FIELD_RECEIPT_20260907.md |
| B1 漂移/同一性门 | B1_CAPTURE_FIX_RECEIPT_20260907.md |
| B2 单元 + 300/300 + 尾部计时 | B2_RADIX_RECEIPT_20260907.md |
| B3 单元门 + 来源阻塞 | B3_AMD_TWIN_RECEIPT_20260907.md |
| 预填充 5.21 s / 1,543 tok/s | FREEZE_QWEN_PREFILL_1543（更早冻结） |
| 再填充速率 | QWEN_REFILL_RECEIPT（更早冻结） |

## 单位

样本/s、轨迹步/s、完成轨迹/s 与兄弟 token/s 是不同的量，各处均分别标注。ms/token 与 tok/s
只在同一配置内互为倒数；未经标注不得把不同启动/配置的数字混入同一比较。
