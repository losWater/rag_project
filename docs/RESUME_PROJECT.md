# Resume Project

## 1. 项目名字

可扩展课程资料 RAG 智能问答系统

## 2. 项目内容

设计并实现一个面向课程资料的可扩展 RAG 智能问答系统，并以 UNSW COMP9444 神经网络课程资料作为首个落地场景。系统支持对本地 PDF 课件进行解析、切分、向量化索引和语义检索，并结合大语言模型基于检索到的资料内容生成带引用的回答。

项目通过配置化资料清单、模型 provider 抽象和增量索引机制，使系统可以扩展到其他课程、文档集合或知识库场景。

## 3. 实现功能

使用 PyMuPDF 解析 PDF 文档，按页保留来源、页码、标题等元数据，并通过固定长度 chunking 策略切分文本；使用 Ollama `nomic-embed-text` 生成向量表示，结合 Chroma 构建本地向量数据库；接入 DeepSeek API 作为生成模型，实现基于检索上下文的 grounded answer generation。

针对中文提问和英文资料之间的语义差异，实现 query rewrite 和 multi-query retrieval；结合 Chroma 向量检索与 BM25 关键词检索，通过 RRF 融合两路排名，再使用 cross-encoder 对候选片段精排。

系统通过 `manifest.yaml` 管理资料清单，通过 provider 抽象预留本地模型和 API 模型切换能力，并基于 SHA256 实现增量索引，支持持续添加或更新资料；同时提供 Typer CLI、Streamlit 可视化 Demo、JSONL 查询日志和轻量级检索评估模块。

## 4. 最终效果

以 COMP9444 课程资料作为验证集，系统已完成 13 份 PDF 课件索引，共构建 356 个可检索文本片段；支持中文/英文提问、指定回答语言、引用来源展示和新资料增量更新。

单元测试 24 项全部通过。在 42 条固定检索用例上，rerank 模式达到 `source_recall@6 = 1.00`、`page_recall@6 = 1.00`、`MRR@6 = 0.964`、`nDCG@6 = 0.861`；对比纯向量模式的 MRR 0.939 和 nDCG 0.816，提高了相关页面的排序质量，同时记录了本地推理延迟成本。

在示例问题“请问什么是交叉熵，用英文回答”中，系统可自动生成 `cross entropy loss`、`cross entropy` 等英文检索词，并准确检索到对应课件页面后生成英文回答。
