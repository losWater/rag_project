# RAG 知识库问答系统 — 项目计划书（面试导向）

> 本文档是交给 Codex 的执行作战手册。
> **项目定位：** 从零搭建一个可复现、可评估、能演示的 RAG（Retrieval-Augmented Generation）知识库问答系统。
> **目标不是做一个普通聊天机器人，而是把 RAG 的关键工程环节讲清楚、跑通、量化。**
> 技术主线：文档解析 + chunking + embedding + 向量检索 + rerank + grounded generation + RAG 评估。
> 执行周期：约 2 周（可视情况伸缩）。

新增核心能力：
- **增量入库**：课程资料持续更新时，可以随时加入新 PDF / notes，只重建新增或变更文件的索引。
- **资料使用规则与检索路由**：允许用户定义一类资料的使用规则，例如“代码课件适合回答实现类问题”。系统先判断用户问题命中了哪些规则，再用 metadata tags 调整检索范围或排序权重，而不是把说明硬绑定到单个 PDF。

---

## 0. 给 Codex 的总体指令（先读这一段）

1. **这是一个面试项目，不是堆功能。** 重点是让我能讲清楚 RAG 为什么有用、如何减少幻觉、检索链路怎么设计、如何评估效果。
2. **不要一次性写完。** 按阶段推进，每阶段结束停下来汇报、列风险、等我确认再继续。
3. **先通后好。** 先用少量文档跑通“上传文档 → 建索引 → 检索 → 带引用回答”的最小链路，再优化 chunking、检索、rerank 和评估。
4. **每个关键环节都要解释为什么。** 包括 chunk size、overlap、embedding 模型、向量库、top-k、rerank、引用生成、评估指标。
5. **API Key、数据路径、模型路径都由我确认。** 不硬编码，用 `.env`，并确认 `.env`、本地向量库、大文件、缓存目录都进 `.gitignore`。
6. 遇到设计决策分叉，先列选项和取舍，问我，再动手。

---

## 1. 项目定位

**任务：** 构建一个面向指定资料库的问答系统。用户输入问题，系统从本地知识库检索相关文档片段，再让 LLM 基于检索内容回答，并给出引用来源。

**默认场景：** “COMP9444 课程资料智能问答助手”。

默认资料库：
- UNSW COMP9444 课程资料、handbook、assignment spec、FAQ；
- 课程 lecture notes / tutorial notes；
- assignment specification、rubric、FAQ、论坛中可合法使用的公开说明；
- 个人整理的课程复习笔记和项目记录。

**为什么选这个任务（面试要能讲清楚）：**
- RAG 是企业 LLM 应用里最常见的落地形态之一，比单纯 prompt demo 更工程化。
- 它能接入私有知识，不需要重新训练模型，知识更新成本低。
- 它能通过引用和检索结果降低幻觉，回答可追溯。
- 它可以量化评估：retrieval recall、answer faithfulness、context precision、引用命中率等。

**明确不做的事（避免范围膨胀）：**
- 不做通用搜索引擎。
- 不做大规模分布式向量数据库。
- 不做模型微调。
- 不做复杂 Agent 编排，除非基础 RAG 已经完成。
- 不公开发布受版权保护的课程资料；仓库只提交脱敏样例、数据格式说明和可复现脚本。

---

## 2. 目标架构

核心链路：

```text
文档资料
→ 读取资料使用规则 / 人工注释
→ 文档解析 / 清洗
→ 文本切块 chunking
→ embedding 向量化
→ 向量库索引 / 增量更新
→ 用户问题向量化
→ Top-k 检索
→ 可选 rerank
→ 构造 prompt
→ LLM 生成带引用答案
→ 记录日志与评估
```

回答模式：
- `grounded`: 检索到足够课程资料，答案必须基于 retrieved contexts，并给出 citations。
- `limited_fallback`: 没有检索到足够课程资料，但问题属于通用、确定、低风险知识；允许基于 LLM 通用知识简短回答，开头必须声明不是课件检索结果。
- `clarify`: 没有检索到足够课程资料，且问题模糊、有歧义、课程特定或高风险；请求用户补充问题或提供相关资料。

系统模块：

```text
rag_project/
├── data/
│   ├── raw/comp9444/        # COMP9444 原始文档，不提交 git
│   ├── rules/comp9444/      # 用户定义的资料使用规则，不提交隐私内容
│   ├── processed/           # 解析后的文本，不提交 git
│   └── eval/                # 小规模评估集，可提交脱敏样例
├── src/
│   ├── ingest.py            # 文档加载、解析、清洗
│   ├── chunk.py             # chunking 策略
│   ├── index.py             # embedding + 向量库写入
│   ├── sync.py              # 文件 hash、增量入库、变更检测
│   ├── router.py            # 规则匹配、语义路由、检索偏置
│   ├── providers.py         # chat / embedding provider 工厂，统一 API 与本地模型接口
│   ├── retrieve.py          # 检索、metadata filter、rerank
│   ├── generate.py          # prompt 构造 + LLM 调用
│   ├── app.py               # CLI 或 Gradio/Streamlit demo
│   └── evaluate.py          # RAG 评估
├── configs/
│   └── rag.yaml             # chunk size、top-k、模型等配置
├── manifest.yaml            # 文档清单、文件类型、tags、是否参与索引
├── retrieval_rules.yaml     # 资料使用规则、触发词/正则/语义描述、偏置 tags
├── tests/
├── README.md
└── .env.example
```

---

## 3. 技术选型

**推荐默认方案（稳、好讲、容易跑通）：**
- 语言：Python。
- RAG 框架：LangChain 或 LlamaIndex，第一阶段二选一。
- 向量库：Chroma（本地持久化，零成本，适合面试项目）。
- Embedding：阶段一默认 Ollama + `nomic-embed-text`；同时预留 OpenAI-compatible embedding API。
- LLM：阶段一默认 DeepSeek API + `deepseek-v4-flash`；同时预留 Ollama 本地 chat model 和其他 OpenAI-compatible API。
- Provider 抽象：从第一阶段开始用 `chat.provider` / `embedding.provider` 配置驱动，不在业务代码里硬编码模型厂商。
- Rerank：先不加；基础链路跑通后再接 cross-encoder 或 API reranker。
- Demo：CLI 优先，时间充足再做 Gradio/Streamlit。
- 评估：自写评估脚本 + 可选 Ragas。

**为什么不是直接微调：**
- RAG 适合知识频繁更新、需要引用来源、数据量不够大或不想训练模型的场景。
- 微调更适合学习输出格式、风格或固定任务行为；不适合把大量事实知识“硬塞”进模型参数。

---

## 4. 分阶段执行计划

### 阶段一：场景确定 + 最小 RAG 跑通（约 2–3 天）

**任务：**
- [ ] 确定 COMP9444 资料范围：handbook、assignment spec、lecture notes、FAQ、个人笔记。
- [ ] 设计 `manifest.yaml`：记录每个资料文件的类型、主题、tags、是否参与索引。
- [ ] 设计 `retrieval_rules.yaml`：记录资料使用规则、触发词/正则/语义描述、偏置 tags。
- [ ] 搭建基础环境，配置 `.env.example` 和 `.gitignore`。
- [ ] 实现 provider 配置骨架：支持 `deepseek_api` chat、`ollama` embedding，并预留 `openai_api` / `ollama` chat / embedding 接口。
- [ ] 实现最小链路：读取少量文档 → 切块 → embedding → Chroma 入库 → 相似度检索 → LLM 回答。
- [ ] 实现最小增量入库：新增 PDF 后重新运行索引命令，只处理新增文件。
- [ ] 输出回答时带上 source metadata，如文件名、页码、chunk id。
- [ ] 产出《最小 RAG 说明》：数据用了什么、chunk 怎么切、top-k 是多少、能回答哪些问题。

**验收标准：** 输入一个资料库相关问题，系统能检索到相关 chunk，并生成带引用的答案；新增一个 PDF 后能增量入库。**此阶段结束停下等我确认。**

---

### 阶段二：检索质量优化（约 3–4 天）— 项目的技术核心

**任务：**
- [ ] 对比不同 chunk 策略：固定长度、按标题/段落切、不同 chunk size 和 overlap。
- [ ] 增加 metadata：source、page、section、document_type，支持按来源过滤。
- [ ] 支持资料使用规则：读取 `retrieval_rules.yaml`，先判断用户问题是否命中某类规则。
- [ ] 第一版规则匹配：关键词 + 正则；命中后使用 metadata tags 做检索过滤或排序加权。
- [ ] 第二版规则匹配：加入语义路由，用用户问题 embedding 和规则描述 embedding 做相似度匹配，避免只依赖关键词。
- [ ] 支持从用户对话整理资料使用规则：先由 Codex 整理成规则条目，再写入 `retrieval_rules.yaml`。
- [ ] 针对特殊资料增加 tags，例如 `code_example`、`assignment_spec`、`conceptual_lecture`、`exam_review`。
- [ ] 实现检索日志：每次问题记录 query、top-k chunks、相似度分数、最终答案。
- [ ] 加入 rerank，对初筛 top-k 结果重新排序。
- [ ] 做检索消融对比：naive vector search vs metadata filter vs rerank。

**要让我理解的点：**
- chunk 太大会稀释语义，太小会丢上下文。
- top-k 不是越大越好，太大会把噪声塞进 prompt。
- rerank 的作用是用更精细的模型判断 query 和 chunk 的相关性。
- 资料使用规则不是替代原文，而是帮助系统理解“这类资料应该在什么问题里优先使用”。
- 关键词/正则适合第一版快速落地，但要预留语义路由，处理用户没有说出明确关键词但意图相同的问题。

**验收标准：** 有一张检索策略对比表，能说明哪种策略更适合当前资料库；资料使用规则能影响检索 metadata、排序和回答策略。

---

### 阶段三：回答可靠性与引用（约 2–3 天）

**任务：**
- [ ] 设计 grounded prompt：要求模型只基于检索上下文回答，不知道就说不知道。
- [ ] 设计 fallback policy：当检索不到足够资料时，区分“可有限回答的通用知识问题”和“需要澄清的模糊问题”。
- [ ] 对非资料库答案强制加声明：例如“以下回答不是从 COMP9444 课件资料中检索得到，而是基于通用知识的有限回答。”
- [ ] 对通用且确定的问题允许有限回答，例如“交叉熵的公式是什么”；回答必须简短、标明非课件来源，并建议用户补充资料以获得课程口径。
- [ ] 对模糊、有歧义、课程特定或无法确定的问题谨慎拒答 / 请求澄清，例如“我没有在资料库中找到对应内容，你可以提供更具体的问题或相关课件吗？”
- [ ] 输出结构化结果：answer、citations、confidence、retrieved_contexts、answer_mode。
- [ ] 实现引用校验：答案里引用的 source 必须来自检索结果。
- [ ] 增加拒答机制：当检索相似度低、上下文不足且不适合有限回答时，不编造答案。
- [ ] 加入简单的 conversation memory，但每轮回答仍以当前检索结果为主，避免历史对话污染事实。

**验收标准：** 系统能区分“资料库内问题”“资料库外但可有限回答的问题”“资料库外且需要澄清的问题”；所有非资料库答案都带明确声明。

---

### 阶段四：评估与量化（约 3–4 天）— 简历的关键

**任务：**
- [ ] 构造 30–50 条评估集：question、gold_answer、gold_source。
- [ ] 评估检索：
  - Recall@k：正确来源是否出现在 top-k；
  - MRR：正确来源排得是否靠前；
  - Context precision：检索结果里噪声比例。
- [ ] 评估生成：
  - Answer correctness：答案是否接近标准答案；
  - Faithfulness：答案是否被上下文支持；
  - Citation accuracy：引用是否正确。
- [ ] 对比实验：
  - 不使用 RAG，直接问 LLM；
  - naive RAG；
  - RAG + rerank；
  - RAG + rerank + 拒答机制。
- [ ] 产出结果表，可直接写进简历。

**验收标准：** 有一组可复现的量化指标，能证明 RAG 比直接 prompt 更稳。

---

### 阶段五：收尾与交付（约 1–2 天）

**任务：**
- [ ] 写 README：项目目标、架构图、RAG 流程、技术选型、如何运行、评估结果。
- [ ] 做 demo：CLI 必做；有余力做 Gradio/Streamlit 页面。
- [ ] 写简历描述句：
  - “构建基于 LangChain/LlamaIndex + Chroma 的 RAG 知识库问答系统，实现文档解析、向量检索、重排、引用生成与自动评估，在自建评估集上将 Recall@5 从 X 提升到 Y，引用准确率达到 Z。”
- [ ] 整理「面试问答自查清单」：
  - RAG 和微调的区别；
  - embedding 和向量检索原理；
  - chunk size / overlap 怎么选；
  - top-k 和 rerank 的作用；
  - 如何减少 hallucination；
  - 如何设计 RAG 评估；
  - 为什么要做引用和拒答。

---

## 5. 风险与注意事项

- **不要只做 UI。** 面试官真正会问的是检索链路、评估、失败案例，而不是页面长什么样。
- **数据源要小而干净。** 初期 10–30 个文档足够，先把链路和评估做好。
- **课程资料注意版权。** 私有课程文件可以本地使用，但不要提交到公开仓库；README 只说明数据来源类型和目录放置方式。
- **评估集比文档数量更重要。** 没有评估，就只能说“看起来能用”，简历说服力弱。
- **引用必须可信。** citations 不能让模型自由编，必须来自检索 metadata。
- **拒答机制要做。** 这是区分玩具 RAG 和工程 RAG 的关键。
- **有限回答必须标注来源边界。** 如果答案不是基于课程资料检索得到，必须在开头说明“非课件资料答案”，并控制回答范围。
- **课程特定问题不能靠常识补。** 涉及 assignment 要求、评分标准、课程安排、老师口径时，检索不到就请求用户提供资料或澄清。
- **知识库要可更新。** 新课件加入后应支持增量索引，避免每次全量重建。
- **人工规则要可追溯。** 资料使用规则必须落成 `retrieval_rules.yaml`，不能只存在聊天上下文里。
- **不要只靠正则。** 关键词/正则用于第一版快速落地，但要保留语义规则匹配入口，避免漏掉表达不同但意图相同的问题。
- **成本控制。** 开发期用小数据、小 top-k、低价模型；日志里记录 token 使用量。
- **Git 卫生。** 不提交原始大文件、向量库、缓存、API key。

---

## 6. 交接 checklist（我交给 Codex 时会补充）

- [x] 资料库最终用：UNSW COMP9444 课程资料
- [ ] RAG 框架：______（LangChain / LlamaIndex / 自写轻量版）
- [ ] LLM 提供方：______（OpenAI / Anthropic / DeepSeek / 本地模型 / 其他）
- [ ] Embedding 模型：______
- [ ] 是否需要 Web demo：______（默认 CLI，时间充足再做）
- [ ] 资料使用规则格式：______（默认 `retrieval_rules.yaml`，规则命中后偏置 `manifest.yaml` 中的 tags）
