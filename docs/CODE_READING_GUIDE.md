# Code Reading Guide

这个文档是给开发者读代码用的。建议不要从前端或模型接口开始读，而是按 RAG 数据流从入口读到核心逻辑。

## 1. 先看项目配置

先读：

```text
configs/rag.yaml
manifest.yaml
.env.example
```

重点理解：

- PDF 放在哪里；
- 向量库放在哪里；
- chunk size 和 overlap 是多少；
- 默认 top-k 是多少；
- chat provider 和 embedding provider 如何配置；
- 哪些 PDF 会被索引。

然后读：

```text
src/config.py
```

理解项目如何把 YAML 和 `.env` 转成运行时配置。

## 2. 再看数据进入系统的过程

按顺序读：

```text
src/ingest.py
src/chunk.py
src/index.py
```

重点理解：

- `ingest.py` 如何从 PDF 中按页提取文本；
- `chunk.py` 如何清洗文本并切分 chunk；
- `index.py` 如何生成 embedding 并写入 Chroma；
- `index.py` 如何用 SHA256 判断文件是否需要重新索引。

读完这一步，你应该能解释：

```text
一个 PDF 是如何变成向量库里的若干条记录的。
```

## 3. 然后看用户提问后的检索过程

按顺序读：

```text
src/query_rewrite.py
src/retrieve.py
```

重点理解：

- 中文问题为什么需要转成英文检索词；
- glossary 兜底逻辑怎么工作；
- LLM query planner 在什么情况下被调用；
- 多个 retrieval query 如何分别检索；
- 检索结果如何合并、去重、重新排序。

读完这一步，你应该能解释：

```text
为什么“请问什么是交叉熵，用英文回答”能搜到英文课件里的 cross entropy。
```

## 4. 再看回答生成

读：

```text
src/generate.py
```

重点理解：

- prompt 如何要求模型只基于检索上下文回答；
- 为什么要保留原始用户问题；
- 回答语言如何控制；
- 当资料不足时，为什么要提醒用户可能需要添加新资料。

## 5. 再看模型 provider 抽象

读：

```text
src/providers.py
```

重点理解：

- 为什么代码里定义 `EmbeddingClient` 和 `ChatClient`；
- Ollama embedding 是怎么调用的；
- DeepSeek 为什么可以用 OpenAI-compatible client；
- 以后如何替换成 OpenAI embedding、本地 chat model 或其他 API。

## 6. 再看完整流水线

读：

```text
src/pipeline.py
```

这是 CLI 和 Streamlit 共用的业务流程。

核心顺序是：

```text
load config
create embedding client
create chat client
rewrite query
retrieve contexts
generate answer
write log
return response
```

## 7. 最后看入口层

CLI 入口：

```text
src/app.py
```

网页入口：

```text
streamlit_app.py
```

这两个文件主要负责接收用户输入、调用 pipeline、展示结果。真正的 RAG 逻辑不应该写在入口层。

## 8. 测试和评估

单元测试：

```text
tests/
```

检索评估：

```text
src/evaluate.py
data/eval/retrieval_cases.yaml
```

建议读代码时最后看测试，因为测试能帮你确认每个模块的预期行为。

## 9. 推荐阅读顺序总结

```text
configs/rag.yaml
manifest.yaml
src/config.py
src/ingest.py
src/chunk.py
src/index.py
src/query_rewrite.py
src/retrieve.py
src/generate.py
src/providers.py
src/pipeline.py
src/app.py
streamlit_app.py
src/logging_utils.py
src/evaluate.py
tests/
```

## 10. 面试时的讲解顺序

如果要向面试官讲这个项目，可以按这个顺序：

1. 先说项目要解决什么问题。
2. 再说为什么普通 LLM 不够，需要 RAG。
3. 解释 PDF 如何进入向量库。
4. 解释中文问题如何匹配英文课件。
5. 解释 multi-query retrieval 为什么比单查询稳定。
6. 解释回答为什么带引用。
7. 解释如何支持新课件增量添加。
8. 最后说测试、日志和后续改进。

