# 来源与权威

本公开树是规范夜间冻结包 FREEZE_QWEN_OVERNIGHT_20260907（内含 SHA256SUMS 与 tarball
qwen_overnight_20260907_canonical.tgz）的脱敏派生品，并附中文译本。本树与包不一致时，以包为准。

## 权威层级

1. 规范夜间冻结包 + SHA256SUMS。
2. 包内冻结回执与工程报告。
3. 生产 Spark 源码：m8-amd-routed @ 420bca8（不在本树中；见下）。
4. Lane B：m1-30tok-0907 — B1 12652f0、B2 f76ebeb。
5. Lane A：stoch-engine-0907 — 01e5d28、148f740、b4bb1d2。
6. 冻结生产 AMD 二进制：824e5c50…（SHA256SUMS 在包内）。
7. AMD 源码镜像：对该二进制 **不** 具权威性。

## 本树使用的来源标签

| 标签 | 含义 |
|---|---|
| AUTHORITATIVE（权威） | 已提交至冻结分支并存在于规范包中 |
| HASH-VERIFIED（哈希核验） | 与规范包条目逐字节一致（对照 package_SHA256SUMS） |
| PATCH-DERIVED（补丁派生） | 从冻结分支提交中提取；仅作为该仓库的一部分才有意义 |
| SANITIZED COPY（脱敏副本） | 由规范回执派生，主机地址、家目录与模型路径已替换 |
| EXPERIMENTAL（实验性） | 已单元验证、无生产集成回执的机制 |
| PENDING-PROVENANCE（来源待定） | 源码未找回；不主张可复现性 |

## 各组件分类

| 路径 | 标签 | 来源 |
|---|---|---|
| stoch/（include、rng、reduce、field、gates、build.sh、BRANCH_DESCRIPTOR.md） | AUTHORITATIVE | stoch-engine-0907 提交 01e5d28、148f740、b4bb1d2（包内 lane_a_stoch.patch） |
| figures/kolmo_field_density.ppm | HASH-VERIFIED | 由保留二进制（stoch/build/kolmo_field，sha be27bb4d）按回执参数生成；已提交于 stoch-engine-0907 |
| figures/kolmo_field_density.png | HASH-VERIFIED | 上述文件的无损转换（PIL） |
| src/cuda/qf_qsa_index.cu、src/cuda/qf.cu | PATCH-DERIVED | m1-30tok-0907 在 B1+B2（12652f0、f76ebeb）之后的状态 |
| src/main.cu | PATCH-DERIVED | stoch-engine-0907 在 b4bb1d2（QF_M8_FORK_SIG 开关）之后的状态 |
| patches/lane_a/、patches/lane_b/ | HASH-VERIFIED | 规范包（lane_a_stoch.patch、lane_b_b1_b2.patch） |
| patches/amd_pending/ | EXPERIMENTAL / UNIT-VERIFIED / PENDING-INTEGRATION | 规范包 tests/ |
| receipts/ | SANITIZED COPY | 规范包 receipts/（原始回执是权威） |
| freeze/package_SHA256SUMS | HASH-VERIFIED | 规范包逐字复制 |
| AMD 生产源码（824e5c50…） | PENDING-PROVENANCE | 未找回；任何镜像均不具权威性 |

## 对回执执行的脱敏

公开回执将私有 AMD 主机地址替换为 <amd-host>，其余私有局域网地址替换为 <mac-host>，家目录
替换为 ~，模型目录替换为 ~/models/<checkpoint>。除此之外，任何数字、计时、token 或命令均
未改动。原始回执保留在规范包中。

## 关于语言

本文档与 docs/ 为中文译本；源码、补丁与回执（receipts/）保留英文原样。回执是证据记录：
翻译证据会引入误差，故以原文为唯一可引用的形式；关键数字已在中文文档中逐条复述并标注
出处。

## 可复现边界

- 独立随机内核（RNG、归约、MC 门、合成场）可在任意 CUDA GPU ≥ sm_70 上从 stoch/ 构建运行。
- Lane B 补丁仅适用于生产仓库（m8-amd-routed @ 420bca8），该仓库不在本树中。
- 依赖模型的实验（MC-C、解码回执）需要双机设备与冻结二进制；冻结生产 AMD 二进制的源码
  状态为 PENDING-PROVENANCE。
