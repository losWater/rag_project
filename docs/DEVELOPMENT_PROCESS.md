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

