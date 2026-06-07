# 实验记录（毕业论文用）

> 优先级：B/C/D 最重要，E 次之，A 最后（A 可随时读 paper1.pdf）。

## B. 我们的实验配置（与论文的差异）⭐
- **模型：Qwen（阿里百炼，OpenAI 兼容端点 `https://dashscope.aliyuncs.com/compatible-mode/v1`）**
  - LLM：`qwen-plus`；VLM：`qwen-vl-max`；Embedding：`text-embedding-v3`（dim **1024**）
  - 解析：MinerU 3.2.1（source=modelscope）
  - ⚠️ 因模型 ≠ 论文（GPT-4o-mini），**必须自跑 Qwen baseline，不能直接套论文数字**
- 硬件：AutoDL RTX 4090 24GB；torch **2.7.1+cu118**（通用驱动，换机器不犯病）
- 数据：开发子集 **25 篇 / 158 问 / 51% 多模态**（`data/DocBench_subset`）；全量 229 篇（最终）
- 检索：`mode="mix"`，`response_type="One Sentence"`，`vlm_enhanced=False`

## C. 我们的方法（改进思路）⭐
- **问题**：RAG-Anything 融合两图靠"实体名精确字符串匹配"（`merge_nodes_and_edges`），名字差一点就不合并 → 图碎片化。
- **L1 名称规范化（万无一失）**：合并前 strip+合并空白 / Unicode NFC / 去结尾 `.,;:`（保护 C++/C#）。开关 `ENABLE_CANONICALIZATION`。
- **L2 同义边 + 结构约束**：建图后，对 `cos(eᵢ,eⱼ) > τ` 且 邻居 `Jaccard(N_i,N_j) > θ` 的实体对**加同义边（不合并，HippoRAG 式）**。开关 `ENABLE_SYNONYM_EDGES`。
- **消融**：2×2（baseline / +L1 / +L2 / full），全部从 feat 分支跑、靠环境变量开关切换。
- **创新点**：① 填补 RAG-Anything 缺同义边机制（论文引了 HippoRAG 但没用其机制）；② 比 HippoRAG 多"邻居 Jaccard 结构约束"，排除"营收/营业成本"这类 embedding 像但语义不同的假阳性。
- 全程 **training-free**（不训练任何模型）。

## D. 每次跑要记录的指标 ⭐
评测脚本 `reproduce/llm_answer_evaluator.py` 用 LLM 判分，支持 **6 个维度**：
`accuracy`(0/1，主指标，对齐论文) / `relevance` / `completeness` / `faithfulness` / `clarity` / `consistency`（各 0–1）+ `overall_score`。
- **准确率**：总体 + 分领域（学术/金融/政府/法律/新闻）+ 分类型（text-only/multimodal/meta-data/unanswerable）
- **质量维度**（可选）：尤其关注 **faithfulness**（L2 改善跨模态 grounding，应提升）
- **机制证据**（比准确率更直接）：图节点数、边数、同义边数（L2）、被合并实体数（L1）
- **成本（¥）、耗时**
- 对比：baseline vs +L1 vs +L2 vs full

## E. 文件 / 位置
- 仓库：`github.com/Keep-Passionate/RAG-Anything-thesis`，分支 `feat/graph-fusion`（`main`=纯净 baseline，不参与消融）
- L1 代码：`raganything/graph_fusion/`（canonicalizer + config）；钩子在 `raganything/modalprocessors.py`（`_create_entity_and_chunk`、`_process_chunk_for_extraction`）
- Qwen 化：`reproduce/index.py`、`reproduce/query.py`（模型名读环境变量 LLM_MODEL/VISION_MODEL/EMBEDDING_MODEL/EMBEDDING_DIM）
- 跑基准（官方脚本，单文件，需 bash 循环调度）：
  - `index.py <pdf> --working_dir rag_storage/<id>` 建索引
  - `query.py <pdf> --working_dir rag_storage/<id>` 答题 → 存 `<docdir>/qa_results_mix_mm.json`
  - `llm_answer_evaluator.py` 判分
- 数据：服务器 `/root/autodl-tmp/DocBench_subset`；本地 `data/DocBench_subset`

## A. 论文（RAG-Anything）实验配置（可随时读 paper1.pdf）
- 数据集：DocBench（229 文档 / 均 66 页 / 1102 问 / 5 领域 / 4 类型）、MMLongBench（135 / 均 47.5 页 / 1082 问 / 7 类型）
- baseline：GPT-4o-mini；embedding text-embedding-3-large(3072)；rerank bge-reranker-v2-m3；解析 MinerU
- token：实体+关系 20000，chunk 12000；答案限一句话；GPT-4o-mini 判分
- 论文结果：DocBench RAGAnything **63.4%**（overall）；MMLongBench **42.8%**
- 消融（DocBench）：Chunk-only 60.0，w/o Reranker 62.4，full 63.4（rerank 仅 +1%）

## 实验结果记录（待填）
| 实验 | 总体% | 学术 | 金融 | 政府 | 法律 | 新闻 | faithful. | 图节点 | 同义边 | 成本¥ | 耗时 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline(子集) | | | | | | | | | - | | |
| +L1(子集) | | | | | | | | | - | | |
| +L2(子集) | | | | | | | | | | | |
| full(子集) | | | | | | | | | | | |
| baseline(全量) | | | | | | | | | - | | |
| full(全量) | | | | | | | | | | | |
