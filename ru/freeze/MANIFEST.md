# Манифест публичного дерева

Кандидат в релиз 0.1.0-rc1, производный от канонического фриз-пакета
FREEZE_QWEN_OVERNIGHT_20260907. Проверка хешей: `./scripts/verify_freeze.sh` (SHA256SUMS
публичного дерева) и `sha256sum -c freeze/package_SHA256SUMS` (по приватному пакету, после
восстановления его раскладки).

| публичный путь | канонический аналог | класс |
|---|---|---|
| README.md, CHANGELOG.md, CITATION.cff, LICENSE | производные от документов пакета | публикационный текст (русский перевод) |
| docs/*.md | производные от документов и чеков пакета | публикационный текст (русский перевод) |
| figures/*.svg | производные только от таблиц чеков | сгенерированные рисунки (русские подписи) |
| figures/kolmo_field_density.ppm | stoch-engine-0907:stoch/kolmo_field_density.ppm | сохранённый артефакт |
| figures/kolmo_field_density.png | lossless-конверсия | сохранённый артефакт |
| stoch/ | lane_a_stoch.patch (исходники) | исходники движка (на английском) |
| src/cuda/qf_qsa_index.cu, src/cuda/qf.cu | lane_b_b1_b2.patch (целевые файлы) | исходники, производные от патчей |
| src/main.cu | lane_a_stoch.patch (целевой файл) | исходник, производный от патча |
| patches/lane_a/lane_a_stoch.patch | lane_a_stoch.patch пакета | патч |
| patches/lane_b/lane_b_b1_b2.patch | lane_b_b1_b2.patch пакета | патч |
| patches/amd_pending/ | tests/ пакета | отложенная работа AMD |
| receipts/*.md | receipts/ пакета (санитизированные) | чеки (свидетельства; на английском) |
| scripts/*.sh | новые; воспроизводят публичные гейты | инструменты (комментарии переведены) |
| freeze/package_SHA256SUMS | SHA256SUMS пакета | свидетельство происхождения |

Любое число, цитируемое в этом дереве, обязано отображаться на чек; таблица аудита — в
docs/REPRODUCIBILITY.md.
