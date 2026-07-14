# Development Process

这个文档记录项目从零搭建到当前可展示版本的过程，方便复盘设计思路，也方便面试时说明项目是怎样一步一步做出来的。

## 1. 明确项目目标

项目目标不是简单做一个聊天机器人，而是做一个面向 COMP9444 课程资料的 RAG 助手。

核心需求：

- 能读取本地 PDF 课件；
- 能基于课件内容回答问题；
- 回答必须带引用；
- 支持中文提问和英文课件之间的检索差异；
- 支持持续添加新课件；
- 支持本地模型和 API 模型两种部署方式；
- 代码结构要足够清晰，适合作为简历项目讲解。

## 2. 选择第一阶段技术栈

第一阶段优先选择稳定、成本低、容易解释的方案：

- PDF 解析：PyMuPDF；
- 向量数据库：Chroma；
- Embedding：Ollama `nomic-embed-text`；
- Chat：DeepSeek API；
- CLI：Typer；
- Demo UI：Streamlit；
- 测试：pytest。

这样可以把模型能力、检索逻辑和应用代码拆开，后续也方便替换模型提供方。

## 3. 搭建基础 RAG 流程

最初实现的主流程是：

```text
PDF -> 文本提取 -> 分块 -> embedding -> Chroma -> 检索 -> LLM 生成回答
```

这一阶段先完成最小可用版本：

- 从 PDF 中提取每一页文本；
- 按固定大小切 chunk；
- 对 chunk 生成 embedding；
- 写入 Chroma；
- 用户提问时检索 top-k chunk；
- 用检索结果生成回答。

## 4. 加入增量索引

为了支持“课程资料还会更新”，项目加入了文件 hash 机制。

每次索引时：

1. 读取 `manifest.yaml`。
2. 计算每个 PDF 的 SHA256。
3. 如果文件没有变化，则跳过。
4. 如果文件新增或修改，则重新解析和索引。
5. 把索引状态写入 `data/vectorstore/index_state.json`。

这样用户后续添加新课件时，不需要重建全部索引。

## 5. 处理中文问题和英文课件的检索差异

课程资料是英文，但用户可能用中文提问。

如果直接用中文问题做 embedding 检索，容易匹配不到英文课件中的关键内容。因此项目加入 query rewrite：

- 英文问题直接检索；
- 中文问题先改写成多个英文检索词；
- 对每个英文检索词分别检索；
- 合并、去重、排序后再交给大模型回答。

例如：

```text
用户问题：请问什么是交叉熵，用英文回答
检索词：
- cross entropy loss
- cross entropy
- cross entropy derivation
- cross entropy KL divergence
```

## 6. 加入回答语言控制

项目要求：

- 用户用什么语言问，默认就用什么语言答；
- 如果用户明确指定回答语言，则按指定语言回答。

所以最终生成回答时，不只使用改写后的英文检索词，而是保留原始用户问题，让大模型看到完整意图。

## 7. 加入查询日志

为了方便调试和复盘，项目加入 JSONL 查询日志。

日志记录：

- 用户问题；
- 实际使用的检索词；
- 检索到的 chunk；
- 来源文件和页码；
- distance；
- 回答结果；
- 模型配置。

这对排查“为什么没搜到”“为什么答错了”很有帮助。

## 8. 加入检索评估

为了让项目更像工程项目，而不是只靠人工测试，加入了轻量 retrieval evaluation。

评估集放在：

```text
data/eval/retrieval_cases.yaml
```

运行：

```bash
python -m src.app eval-retrieval --top-k 6
```

评估结果会显示：

- 是否命中预期文件；
- 是否命中预期页码；
- source recall；
- page recall。

## 9. 加入 Streamlit Demo

CLI 能证明功能，但面试展示时不够直观。因此增加了一个简单网页：

- 左侧展示索引状态和模型配置；
- 主区域输入问题；
- 展示回答；
- 展示检索词；
- 展示引用表格；
- 展示检索原文。

## 10. 当前阶段结果

当前项目已经完成：

- PDF ingestion；
- chunking；
- embedding；
- Chroma indexing；
- incremental indexing；
- bilingual query rewrite；
- multi-query retrieval；
- grounded generation；
- citations；
- query logs；
- CLI；
- Streamlit demo；
- retrieval evaluation；
- unit tests。

目前可以作为一个第一阶段可展示版本。

## 11. 加入 BM25 混合检索和 RRF

纯向量检索适合同义表达和语义匹配，但对代码、缩写和专有术语的精确字面匹配不稳定。因此第二阶段增加 BM25 关键词检索，并保留 Chroma 向量检索作为另一路召回。

两路结果的原始分数不可直接比较：Chroma 返回向量距离，BM25 返回词项相关性分数。项目使用 Reciprocal Rank Fusion（RRF）按各自排名进行融合，避免手工归一化不同分数尺度。

当前支持三种可配置模式：

- `vector`：multi-query 向量检索；
- `bm25`：multi-query 关键词检索；
- `hybrid`：两路召回后使用 RRF 融合。

在现有 5 条轻量验证用例上，三种模式的 `source_recall@6` 和 `page_recall@6` 都是 `1.00`。这只证明新链路没有造成已有用例回退，不能证明 hybrid 已经优于纯向量。下一步需要扩展固定评估集，加入精确术语、代码 API 和语义改写等能够区分检索策略的案例。

## 12. 扩展评估并加入 Cross-encoder Reranker

评估集从 5 条扩展到 20 条，覆盖精确术语、PyTorch API、概率公式、优化方法和卷积尺寸计算。中文用例保存固定的英文 retrieval queries，确保消融实验不会因为 LLM query rewrite 的随机性而波动。

除 source/page recall 外，评估增加：

- MRR：正确页面首次出现得越早，分数越高；
- nDCG：衡量整个 top-k 中相关页面的排序质量。

Hybrid 先召回最多 20 个候选，再使用 `cross-encoder/ms-marco-MiniLM-L6-v2` 联合编码 query 和 chunk 并重新打分。模型加载或推理失败时，系统回退到 RRF 排名。

20 条固定用例的结果：

| 模式 | Source Recall@6 | Page Recall@6 | MRR@6 | nDCG@6 | 平均检索延迟 |
|---|---:|---:|---:|---:|---:|
| Vector | 1.00 | 1.00 | 0.960 | 0.870 | 53 ms |
| BM25 | 1.00 | 1.00 | 0.863 | 0.796 | 16 ms |
| Hybrid + RRF | 1.00 | 1.00 | 0.938 | 0.866 | 55 ms |
| Hybrid + Reranker | 1.00 | 1.00 | 0.975 | 0.900 | 530 ms |

这组结果显示等权 RRF 没有直接超过纯向量检索，而 cross-encoder 提高了最终排序质量。reranker 同时增加本地推理延迟，因此四种模式继续保留，便于按质量和延迟要求切换。延迟为本机 CPU 对 20 条用例的一次运行结果，只用于项目内相对比较。

## 13. 扩展课件并引入 Hard Negatives

知识库新增 Image Processing、Reinforcement Learning、Deep Reinforcement Learning 和 Recurrent Networks 四份课件。增量索引只处理新增文件：

```text
Indexed files: 4
skipped files: 9
chunks added: 140
```

索引规模从 9 份课件、216 个 chunk 增长到 13 份课件、356 个 chunk。评估集同步扩展到 42 条，并增加明确的 source/page relevance judgments、Hit@k、P95 延迟和 JSON 报告导出。

新增文档引入了更多相似主题。例如 supervised learning、KL divergence 和 recurrent networks 会同时出现在多份课件中，形成比原始小集合更真实的 hard negatives。最终结果：

| 模式 | Source Recall@6 | Page Recall@6 | MRR@6 | nDCG@6 | 平均检索延迟 |
|---|---:|---:|---:|---:|---:|
| Vector | 1.00 | 1.00 | 0.939 | 0.816 | 46 ms |
| BM25 | 1.00 | 0.98 | 0.800 | 0.702 | 16 ms |
| Hybrid + RRF | 1.00 | 1.00 | 0.923 | 0.799 | 59 ms |
| Hybrid + Reranker | 1.00 | 1.00 | 0.964 | 0.861 | 305 ms（含冷启动） |

Reranker 的稳态 P95 为 183 ms。一次失败分析还发现英文查询中的课程编号会让所有课件标题页获得虚假相关性，因此英文问题也加入检索噪声清洗；BM25 tokenizer 同时增加轻量复数归一化，将考核信息相关页从 BM25 第 74 名提升到第 2 名。
