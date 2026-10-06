# scripts/ -  V3 增量实验脚本

本目录是 2026-09-30《审稿结论先行》之后、为支撑重构稿新写的分析代码，
按论文的分析顺序分阶段编排。每个脚本的自述摘自它自己的文档字符串，避免张冠李戴。

**运行前设置两个环境变量**（本机绝对路径已从脚本中移除）：

| 变量 | 含义 |
| --- | --- |
| `CODONBENCH_EXP_ROOT` | 实验数据根目录：队列、冻结折、嵌入缓存都挂在它下面（默认占位 `/path/to/your/codonbench-exp`） |
| `CODONBENCH_SERVER_HOME` | 服务器 home：仅用于 conda / 预装依赖路径（默认占位 `/path/to/your/server-home`） |

## Prerequisites

主流程前必须先通过的守卫性检查。

| 脚本 | 自述 | 脚本内出现的结果文件 |
| --- | --- | --- |
| `WB-E0-必过测试.py` | WB \| E0 必过测试（最小够用版） | — |

## Stage 1 - 队列与冻结折

定义评价队列，把 240 个留一基因折冻结下来；此后所有结果都在同一套折上比较。

| 脚本 | 自述 | 脚本内出现的结果文件 |
| --- | --- | --- |
| `WB-e0-cohort-declare.py` | WB-e0-cohort-declare.py (2026-10-03) | — |
| `WB-e0-fold-unit-audit.py` | WB-e0-fold-unit-audit.py  (2026-10-03) | — |
| `WB-e0-freeze-folds.py` | WB-e0-freeze-folds.py  (2026-10-03) | — |
| `WB-e0-freeze-pack.py` | WB-e0-freeze-pack.py (2026-10-03) | — |
| `WB-e0-frozen-v2.py` | WB-e0-frozen-v2.py (2026-10-03) | — |
| `WB-e0-model-registry.py` | WB-e0-model-registry.py (2026-10-03) | — |
| `WB-e0-rebuild-foldid.py` | WB-e0-rebuild-foldid.py  (2026-10-03) | — |
| `WB-e0-run-all.py` | WB \| E0 全量执行（补做 10-02 欠下的那一步） | — |

## Stage 2 - 冻结折上的表征比较

随机划分 vs 逐基因留一，以及各冻结表征的表现。

| 脚本 | 自述 | 脚本内出现的结果文件 |
| --- | --- | --- |
| `WB-e1-baselines-240folds.py` | WB-e1-baselines-240folds.py  (2026-10-03) | — |
| `WB-e1-extract.py` | WB \| E1 严格版 —— 第 1 步：提取 reference-only 与 alternate 嵌入 | `e1_seq_meta.json` |
| `WB-e1-extract2.py` | WB \| E1 严格版 —— 提嵌入（修正版） | `e1v2_extract_report.json` |
| `WB-e1-multimodel-fix.py` | WB \| E1 严格版 —— 多模型扩展 修复版（EXP-007b） | `e1_multimodel_results.json` |
| `WB-e1-multimodel.py` | WB \| E1 严格版 —— 多模型扩展（EXP-007） | `e1_multimodel_results.json` |
| `WB-e1-seed-uncertainty.py` | WB-e1-seed-uncertainty.py (2026-10-03) | — |

## Stage 3 - 位置与序列基线

只读序列标量的查表基线（不进嵌入），把外显子位置这条通道单独称出来。

| 脚本 | 自述 | 脚本内出现的结果文件 |
| --- | --- | --- |
| `WB-e1-b3-ablation.py` | WB-e1-b3-ablation.py (2026-10-03) | — |
| `WB-e1-b4-splice.py` | WB-e1-b4-splice.py (2026-10-03) | — |
| `WB-e1-b4-vs-b3.py` | WB-e1-b4-vs-b3.py (2026-10-03) | — |
| `WB-e1-strict-phaseA.py` | WB \| E1 严格版 —— Phase A：数据自检（纯 CPU，不加载模型） | `e1_strict_phaseA_report.json` |
| `WB-e1-strict-phaseA2.py` | WB \| E1 严格版 —— Phase A2：序列构造与自检（纯 CPU） | `e1s_meta.json` |
| `WB-e1-strict-phaseB.py` | WB \| E1 严格版 —— Phase B：提嵌入 + 跑臂（GPU） | `e1_strict_results.json` |
| `WB-e1-strict-phaseC.py` | WB \| E1 严格版 —— Phase C：配对检验 | `e1_strict_paired.json` |

## Stage 4 - 响应向量与四来源分解

构造等价对照变异（位置匹配 → 再匹配碱基化学类别），把可预测性分解成四来源。

| 脚本 | 自述 | 脚本内出现的结果文件 |
| --- | --- | --- |
| `WB-e1-decompose-0138.py` | 分解结论三的 +0.138：这个差值由哪些成分构成（纯 CPU，不再占用 GPU）。 | — |
| `WB-e1-decompose-0138b.py` | 分解 +0.138 的第二式：增量分解（increment），替代残差化。 | — |

## Stage 5 - 复核

对上一阶段中间产物的独立复核。

| 脚本 | 自述 | 脚本内出现的结果文件 |
| --- | --- | --- |
| `WB-e1-audit-emb.py` | WB \| E1 严格版 —— 现有嵌入缓存退化审计 | — |
| `WB-e1-esm-aa-vs-nt.py` | WB-e1-esm-aa-vs-nt.py (2026-10-03) | — |

## 基因家族聚类

按基因家族聚类，用于跨家族留出。

| 脚本 | 自述 | 脚本内出现的结果文件 |
| --- | --- | --- |
| `WB-fetch-refgene.py` | WB-fetch-refgene.py —— 下载 UCSC refGene（hg19 + hg38），解析出每个转录本的外显子结构。 | — |
| `WB-g3-build-clusters.py` | WB \| G3 —— 第 2 步：由 HGNC 家族构造基因簇，并检查规模 | — |
| `WB-g3-fetch-hgnc.py` | WB \| G3 真实同源/家族划分 —— 第 1 步：取 HGNC 基因家族注释 | — |
| `WB-g3-rerun.py` | WB \| G3（未见同源簇）—— 真正做出来 | `g3_gene_clusters.json` |

## 补充分析

基因置换检验、阴性对照、跨任务重跑。

| 脚本 | 自述 | 脚本内出现的结果文件 |
| --- | --- | --- |
| `WB-e2-rerun.py` | WB \| E2 重做（按 WB-实验设计书 v1） | `e2_rerun_results.json` |
| `WB-e3-rerun.py` | WB \| E3 重做（扩容）：支持臂 vs 排除臂（按 WB-实验设计书 v1） | `e3_rerun_results.json` |
| `WB-e4a-fixup.py` | WB-e4a-fixup.py (2026-10-03) | — |
| `WB-e4a-gene-perm.py` | WB-e4a-gene-perm.py  (2026-10-03) | — |
| `WB-e4a-negative.py` | WB-e4a-negative.py (2026-10-03) | — |

## 复审状态分析

ClinVar 复审状态分层。

| 脚本 | 自述 | 脚本内出现的结果文件 |
| --- | --- | --- |
| `WB-exp010-reviewstatus.py` | EXP-010｜标签地基检验：按 ClinVar review status 分层重跑响应向量主分析 | `exp010_reviewstatus_results.json` |
| `WB-probe-reviewstatus.py` | EXP-010 前置探测（纯 CPU，不占 GPU） | — |

## 诊断与补跑

历史数值一致性诊断与补跑，用于确认锚点对得上。

| 脚本 | 自述 | 脚本内出现的结果文件 |
| --- | --- | --- |
| `WB-b3-audit.py` | B3 可信度审计（2026-10-02） | — |
| `WB-b5-rerun.py` | WB \| B-5 重跑：直接 cLM-pLM 配对差值（按 WB-实验设计书 v1） | `b5_rerun_results.json` |
| `WB-diag-p0o.py` | WB-诊断-P0-o：CodonBERT 主结果锚点不一致，定位根因并判定是否需要重跑。 | `e1_strict_results.json` |
| `WB-gap-fill.py` | WB 缺口补齐（2026-10-02）—— 按「新论文设计GPT.md」第五章与第七章补齐我这版没做的部分。 | — |
| `WB-mispath-multimodel.py` | WB \| MisPath 错义通道 · 多模型响应向量（EXP-009） | — |
| `WB-mispath-positive.py` | WB \| MisPath 阳性对照（EXP-008） | — |
| `WB-verify-log.py` | WB-verify-log.py —— 实验日志数字对账 | `e2_rerun_WB.json` |

## 启动器

后台启动器（setsid + nohup，避免 SSH 断连丢输出）。

| 脚本 | 自述 | 脚本内出现的结果文件 |
| --- | --- | --- |
| `run_diag.sh` | cd "${CODONBENCH_EXP_ROOT:-/path/to/your/codonbench-exp}" | — |
| `run_e0all.sh` | cd "${CODONBENCH_EXP_ROOT:-/path/to/your/codonbench-exp}" | — |
| `run_e0t8.sh` | cd "${CODONBENCH_EXP_ROOT:-/path/to/your/codonbench-exp}" | — |
| `run_e1base240.sh` | cd "${CODONBENCH_EXP_ROOT:-/path/to/your/codonbench-exp}" | — |
| `run_e4a.sh` | cd "${CODONBENCH_EXP_ROOT:-/path/to/your/codonbench-exp}" | — |
| `run_exp009.sh` | source "${CODONBENCH_SERVER_HOME:-/path/to/your/server-home}/anaconda3/etc/profile.d/conda.sh" | — |
| `run_exp010.sh` | cd "${CODONBENCH_EXP_ROOT:-/path/to/your/codonbench-exp}" | — |

## Other

| 脚本 | 自述 | 脚本内出现的结果文件 |
| --- | --- | --- |
| `WB-e0-cdsbert-tokdiag.py` | WB-e0-cdsbert-tokdiag.py —— cdsBERT / cdsBERT-plus 分词诊断 | — |
| `WB-e0-s4-diag.py` | WB-e0-s4-diag.py —— 诊断 S4 里那个极端不平衡： | — |
| `WB-e0-t1-diag.py` | WB-e0-t1-diag.py (2026-10-03) | — |
| `WB-e0-t1-diag2.py` | WB-e0-t1-diag2.py (2026-10-03) | — |
| `WB-e0-t1-diag3.py` | WB-e0-t1-diag3.py (2026-10-03) | — |
| `WB-e0-t1-hgvs-arbitrate.py` | WB-e0-t1-hgvs-arbitrate.py (2026-10-03) | — |
| `WB-e0-t8-seqhash.py` | WB \| E0-T8 序列哈希（纯 CPU） | — |
| `WB-e0-t8b-align.py` | WB \| E0-T8b 逐条对齐比对（纯 CPU） | — |

