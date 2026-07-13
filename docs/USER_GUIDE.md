# User Guide

这个文档面向项目使用者，说明如何在本地运行 COMP9444 RAG Assistant、添加新课件、提问并查看结果。

## 1. 准备环境

进入项目目录：

```bash
cd rag_project
```

创建并启用虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
```

安装依赖：

```bash
python -m pip install -r requirements.txt
```

## 2. 配置模型

复制 `.env.example`，创建本地 `.env` 文件，并填入自己的 API key。

默认测试配置是：

```text
Chat: DeepSeek API
Embedding: Ollama nomic-embed-text
```

启动 Ollama 后，确认 embedding 模型可用：

```bash
ollama pull nomic-embed-text
ollama list
```

## 3. 添加课程资料

把 PDF 课件放到：

```text
data/raw/comp9444/
```

然后在 `manifest.yaml` 中添加对应记录，包括文件路径、标题、文档类型和标签。

示例：

```yaml
- path: data/raw/comp9444/4a_Convolution.pdf
  title: Week 4a Convolutional Networks
  document_type: lecture
  tags: [convolutional_networks, cnn, conceptual_lecture]
  index: true
```

## 4. 建立或更新索引

运行：

```bash
python -m src.app index
```

系统会根据文件 hash 判断哪些 PDF 是新增或修改过的。旧文件不会重复索引。

查看索引状态：

```bash
python -m src.app status
```

## 5. 命令行提问

英文问题：

```bash
python -m src.app ask "What are convolutional networks?" --top-k 6
```

中文问题，并指定英文回答：

```bash
python -m src.app ask "请问什么是交叉熵，用英文回答" --top-k 6
```

系统会：

1. 判断是否需要把问题改写成英文检索词。
2. 使用 Chroma 向量检索和 BM25 关键词检索召回相关片段。
3. 使用 RRF 融合两路排名并去重。
4. 把检索结果交给大模型生成回答。
5. 返回回答、引用来源和检索来源。

默认使用 `hybrid` 模式。也可以临时覆盖模式进行对比：

```bash
python -m src.app ask "What is ReLU?" --retrieval-mode vector
python -m src.app ask "What is ReLU?" --retrieval-mode bm25
python -m src.app ask "What is ReLU?" --retrieval-mode hybrid
```

## 6. 使用网页 Demo

启动 Streamlit：

```bash
streamlit run streamlit_app.py
```

页面中可以：

- 输入问题；
- 调整检索片段数量；
- 查看生成的 retrieval queries；
- 查看回答；
- 查看引用来源；
- 展开查看检索到的原文片段；
- 选择是否写入查询日志。

## 7. 查询日志

默认会把每次查询记录到：

```text
logs/queries/YYYY-MM-DD.jsonl
```

日志包含：

- 原始问题；
- 检索词；
- 检索到的片段；
- 距离分数；
- 检索来源和 RRF 融合分数；
- 回答内容；
- 使用的模型配置名。

日志不会保存 API key。

## 8. 常见问题

如果新课件没有被检索到：

1. 确认 PDF 已放入 `data/raw/comp9444/`。
2. 确认 `manifest.yaml` 中有对应记录。
3. 重新运行 `python -m src.app index`。
4. 在页面中刷新，或重新提问。

运行检索消融评估：

```bash
python -m src.app eval-retrieval --top-k 6 --retrieval-mode vector
python -m src.app eval-retrieval --top-k 6 --retrieval-mode bm25
python -m src.app eval-retrieval --top-k 6 --retrieval-mode hybrid
```

如果模型连接失败：

1. 确认 `.env` 已填写。
2. 确认 Ollama 正在运行。
3. 确认 DeepSeek API key 有效。
