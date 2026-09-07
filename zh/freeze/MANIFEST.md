# 公开树清单

候选发布版 0.1.0-rc1，派生自规范冻结包 FREEZE_QWEN_OVERNIGHT_20260907。哈希校验：
`./scripts/verify_freeze.sh`（公开树 SHA256SUMS）与 `sha256sum -c freeze/package_SHA256SUMS`
（对照私有包，需先还原其目录布局）。

| 公开路径 | 规范包对应物 | 类别 |
|---|---|---|
| README.md、CHANGELOG.md、CITATION.cff、LICENSE | 由包内文档派生 | 发布文本（中文译本） |
| docs/*.md | 由包内文档与回执派生 | 发布文本（中文译本） |
| figures/*.svg | 仅由回执表格派生 | 生成图（中文标注） |
| figures/kolmo_field_density.ppm | stoch-engine-0907:stoch/kolmo_field_density.ppm | 保留工件 |
| figures/kolmo_field_density.png | 无损转换 | 保留工件 |
| stoch/ | lane_a_stoch.patch（源码） | 引擎源码（保留英文） |
| src/cuda/qf_qsa_index.cu、src/cuda/qf.cu | lane_b_b1_b2.patch（目标文件） | 补丁派生源码 |
| src/main.cu | lane_a_stoch.patch（目标文件） | 补丁派生源码 |
| patches/lane_a/lane_a_stoch.patch | 包内 lane_a_stoch.patch | 补丁 |
| patches/lane_b/lane_b_b1_b2.patch | 包内 lane_b_b1_b2.patch | 补丁 |
| patches/amd_pending/ | 包内 tests/ | AMD 待办工作 |
| receipts/*.md | 包内 receipts/（脱敏副本） | 回执（证据记录，保留英文） |
| scripts/*.sh | 新建；复现公开门 | 工具（脚本头注释中文化） |
| freeze/package_SHA256SUMS | 包内 SHA256SUMS | 来源证据 |

本树中引用的任何数字都必须能映射到回执；审计表见 docs/REPRODUCIBILITY.md。
