# Agentic RAG — 基于 LangGraph 的智能检索增强知识库服务

一个生产级 Agentic RAG 服务：**计划(plan) → 调度(schedule) → 向量工具路由 → 并行检索 → 双重评估 → 聚合生成**。它不只是一个"检索再生成"的普通 RAG，而是把用户问题先改写、再分解成可依赖的任务 DAG，通过 LangGraph 并行调度执行，并在检索与生成两个环节都加入了质量评估和重试循环。

- **后端**：FastAPI + LangGraph 编排，LangChain / OpenAI 负责模型调用，LlamaIndex 负责文件解析切分
- **检索**：Milvus 稠密向量 + MySQL BM25 稀疏检索，经倒数排名融合(RRF)混合打分，再由文档评分器(Document Grader)把关
- **存储**：MySQL 存文档/任务/分块元数据，Milvus 存稠密向量，Redis 存匿名会话状态
- **前端**：Vue 3 + Vite 单页应用，提供问答工作台、资料库、运行记录三个视图，支持 SSE 流式输出

## 特性

- **查询改写 + 意图识别**：单次 LLM 调用同时完成查询改写、检索子查询生成与意图分类，减少模型往返
- **任务计划 DAG**：把复杂问题拆分为最小任务集，任务间可声明依赖，天然支持并行/串行混合编排
- **并行任务调度**：基于 LangGraph 的 `Send` 机制将每个就绪任务作为独立分支派发，互不共享状态、并行执行，`finalize` 节点统一聚合结果
- **向量工具路由**：将每个计划任务向量化，与所有已启用工具的能力描述做语义相似度匹配，只有通过置信度与硬约束校验的工具才会被执行
- **混合检索 + 双重评估**：
  - 文档评分器(Document Grader)：按检索相关度阈值确定性判定证据是否充分，不足则自动转网络搜索补充
  - 回答评分器(Answer Grader)：校验回答是否有证据支撑且切题，不达标时把该任务送回重检(有重试上限)
- **兜底能力**：工具路由失败时默认回退到 RAG；仅当用户显式选中了不支持的工具时才返回"能力未开通"
- **可观测性**：每个节点主动上抛 trace 事件，前端通过 SSE 实时展示检索、评分、生成等环节的轨迹
- **流式问答与文件知识库**：支持 SSE 流式输出，支持 pdf/docx/txt/md 等文件上传后在后台异步切分索引

## 工作流程

```mermaid
flowchart TD
    START([START]) --> rewrite["rewrite<br/>改写 + 意图识别"]
    rewrite --> plan["plan<br/>任务 DAG"]
    plan --> schedule["schedule<br/>计算就绪任务集"]
    schedule --> dispatch{dispatch}
    dispatch -- 有就绪任务 --> et["execute_task 任务子图 xN<br/>(并行 Send)"]
    et --> schedule
    dispatch -- 无任务 --> finalize["finalize<br/>聚合答案与引用"]
    finalize --> END([END])

    subgraph task["任务子图（每个任务独立实例，并行运行）"]
        route["route<br/>向量工具路由"] --> br{分支}
        br -- rag-agent --> retrieve["retrieve<br/>混合检索"]
        br -- web-search-agent --> ws["web_search<br/>Tavily"]
        br -- 不支持工具 --> unsup["complete_unsupported"]
        retrieve --> dg["document_grade<br/>证据评估"]
        dg -- 证据不足 --> ws
        dg -- 证据充分 --> gen["generate<br/>生成回答"]
        ws --> gen
        gen --> ag["answer_grade<br/>回答评分"]
        ag -- 不达标且未超重试 --> retrieve
        ag -- 通过/超限 --> ct["complete_task"]
        unsup --> e1([任务结束])
        ct --> e1
    end
```

整体上：**外层主图**负责改写 → 规划 → 并行调度 → 收尾；**任务子图**负责每个单任务的路由 → 检索/搜索 → 证据评估 → 生成 → 回答评分 → 完成。

**外层主图（改写 → 规划 → 并行调度 → 收尾）**

| 节点 | 职责 |
|:---|:---|
| `rewrite` | 改写为独立问句、生成检索子查询、识别意图（一次调用） |
| `plan` | 生成任务 DAG，初始化任务结果与重试计数 |
| `schedule` / `dispatch` | 计算就绪任务集，用 `Send` 并行派发；无就绪任务则送 `finalize` |
| `finalize` | 聚合各任务的答案、引用与警告 |

**任务子图（每个任务一个独立实例，并行运行）**

```
route → retrieve → document_grade → generate → answer_grade → complete_task
     ↘ web_search ↗                  ↗(证据不足自动转网络搜索)
```

| 节点 | 职责 | 分支 |
|:---|:---|:---|
| `route` | 向量工具路由：选择 `rag-agent` / `web-search-agent`，或标记不支持 | 命中 web 搜索走 `web_search`；选到不支持工具走 `complete_unsupported`；否则默认走 RAG `retrieve` |
| `retrieve` | 混合检索（Milvus 稠密 + BM25 稀疏 + RRF），返回引用与最高相关度 | — |
| `document_grade` | 按相关度阈值确定性判定证据充分性 | 充分 → `generate`；不足 → `web_search` 补充 |
| `web_search` | Tavily 网络搜索，结果并入引用列表 | — |
| `generate` | 基于引用生成带引文的回答 | — |
| `answer_grade` | 校验回答是否有证据支撑、是否切题 | 不达标且重试未超限 → 送回 `retrieve`；否则 → `complete_task` |
| `complete_unsupported` | 选中的工具不可用时返回"能力未开通" | — |

## 技术栈

| 层 | 技术 |
|:---|:---|
| 运行时 | Python 3.11/3.12 |
| Web 框架 | FastAPI, Uvicorn, pydantic v2 |
| 工作流编排 | LangGraph |
| 模型调用 | LangChain, langchain-openai（OpenAI Chat / Embedding） |
| 文档解析 | llama-index-core, llama-index-readers-file（SentenceSplitter 切分） |
| 稠密检索 | PyMilvus(≥2.5), Milvus |
| 稀疏检索 | rank-bm25 (BM25Okapi) + MySQL |
| 缓存/会话 | Redis |
| 元数据存储 | SQLAlchemy 2.0 + PyMySQL |
| 网络搜索 | Tavily API |
| 前端 | Vue 3, Vite 6, TypeScript, lucide-vue-next |
| 测试/质量 | pytest, pytest-asyncio, ruff |

## 目录结构

```
├── app/                    # FastAPI 后端
│   ├── main.py             # 应用组装：中间件、路由、静态资源、全局异常
│   ├── api/routes.py       # HTTP 路由（健康检查、知识库、问答）
│   ├── core/               # 配置、统一响应信封、错误处理
│   ├── domain/models.py    # 领域模型（计划任务、工具规格、引用、请求响应）
│   ├── services/
│   │   ├── workflow.py     # ★ LangGraph 工作流（主图 + 任务子图）
│   │   ├── tool_router.py  # 向量工具路由（default_tool_specs）
│   │   ├── retrieval.py    # 混合检索 + RRF
│   │   ├── llm.py          # 改写/意图/规划/生成/评分
│   │   ├── embedding.py    # OpenAI / Hash 向量化
│   │   ├── knowledge.py    # 文件上传/解析/切分/索引
│   │   ├── web_search.py   # Tavily 搜索
│   │   ├── session.py      # Redis 会话
│   │   └── repository*.py  # MySQL 持久层
│   └── static/             # 内嵌静态首页（无前端构建产物时兜底）
├── frontend/               # Vue 3 前端
├── tests/                  # pytest 测试
├── data/uploads/           # 上传文件落盘目录（git ignored）
├── pyproject.toml
└── .env.example            # 环境变量示例（需自行创建，见下方“配置”）
```

## 快速开始

### 1. 后端

要求 Python **3.11 或 3.12**。

```bash
# 创建虚拟环境
py -3.12 -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

# 安装依赖（含 dev）
pip install -e ".[dev]"

# 配置环境变量
# 复制示例并填写 MySQL / Redis / Milvus / OpenAI / Tavily 配置（见“配置”）
# 若 .env.example 尚未创建，请按下方“配置”一节自行创建 `.env` 文件

# 启动 API（单实例，避免同端口多实例路由错乱）
.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

打开 http://127.0.0.1:8000/docs 查看 Swagger 文档。

> 注意：生产环境请求需要已配置的外部服务；模拟适配器(Fake adapters)仅供自动化测试使用。

### 2. 前端

```bash
cd frontend
npm install
npm run dev        # 打开 http://127.0.0.1:5173
```

开发时 Vite 会把 `/api` 请求代理到 FastAPI（端口 8000）。执行 `npm run build`（含 `vue-tsc` 类型检查）后，FastAPI 会把 `frontend/dist/` 的构建产物作为根页面提供。

## 配置（.env）

参考 `.gitignore`，项目会忽略 `.env` 并保留 `!.env.example`。请创建 `.env` 文件，可参考以下变量：

```dotenv
# 应用
APP_ENV=development
API_V1_PREFIX=/api/v1
UPLOAD_DIR=./data/uploads

# 外部服务
DATABASE_URL=mysql+pymysql://user:pass@127.0.0.1:3306/agentic_rag?charset=utf8mb4
REDIS_URL=redis://127.0.0.1:6379/0
MILVUS_URI=http://127.0.0.1:19530
MILVUS_TOKEN=
MILVUS_DATABASE=default
MILVUS_KNOWLEDGE_COLLECTION=knowledge_chunks
MILVUS_TOOL_COLLECTION=tool_registry

# OpenAI（可用阿里云百炼等兼容网关，通过 OPENAI_BASE_URL 指定）
OPENAI_API_KEY=sk-xxxx
OPENAI_BASE_URL=
OPENAI_CHAT_MODEL=gpt-4.1-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Tavily
TAVILY_API_KEY=tvly-xxxx

# 行为调参（可选）
MAX_UPLOAD_SIZE_MB=20
RETRIEVAL_TOP_K=8
ROUTER_THRESHOLD=0.62          # 工具路由置信度阈值
ROUTER_AMBIGUITY_DELTA=0.05    # 与次高分的最小差距，避免歧义
RETRIEVAL_GROUNDED_THRESHOLD=0.35
MAX_AGENT_RETRIES=2            # 回答评分未通过的最大重试次数
SESSION_TTL_SECONDS=3600
```

## 默认注册的工具

由 `app/services/tool_router.py` 的 `default_tool_specs()` 提供，路由时按语义相似度匹配：

| tool_id | 名称 | 类型 | 默认状态 | 用途 |
|:---|:---|:---|:---|:---|
| `rag-agent` | Knowledge Base RAG Agent | rag | 启用 | 检索内部知识库，回答带引用的提问 |
| `web-search-agent` | Web Search Agent | tools | 启用 | 搜索公开网络来源 |
| `text2sql-agent` | Text2SQL Agent | tools | 禁用 | 将分析类问题转成只读 SQL（需显式开启才可被路由） |

路由判定：工具必须启用、类型匹配且输入覆盖，再按向量余弦相似度排序；最高分低于阈值或与次高分差距过小时不路由。

## API 一览

统一响应信封：`{ "success": bool, "data": ..., "error": {...}, "timestamp": ... }`。所有接口前缀为 `API_V1_PREFIX`（默认 `/api/v1`）。

| 方法 | 路径 | 说明 |
|:---|:---|:---|
| `GET` | `/health/live` | 存活探针 |
| `GET` | `/health/ready` | 就绪探针，返回缺失的外部依赖清单 |
| `POST` | `/knowledge/files` | 上传文件（pdf/docx/txt/md），后台异步索引，返回 202 |
| `GET` | `/knowledge/files?page=&size=` | 分页列出已上传文件 |
| `DELETE` | `/knowledge/files/{document_id}` | 删除文档及其索引 |
| `POST` | `/knowledge/files/{document_id}/reindex` | 触发重新索引 |
| `GET` | `/tasks/{task_id}` | 查询索引任务状态 |
| `POST` | `/chat/query` | 非流式问答，返回答案与引用 |
| `POST` | `/chat/query/stream` | SSE 流式问答（metadata / trace / citation / grader / token / done 事件） |

**问答请求体示例**

```json
{ "query": "介绍一下大模型 RAG 的演进", "session_id": "可选" }
```

## 数据存储

- **MySQL（元数据）**
  - `knowledge_documents`：文档（id / filename / path / status / created_at）
  - `knowledge_jobs`：索引任务（id / type / document_id / status / error / created_at）
  - `knowledge_chunks`：分块文本（id / document_id / content / page / created_at），同时供 BM25 稀疏检索打分
- **Milvus（稠密向量）**，collection `knowledge_chunks`，schema 字段：`id` / `vector`(COSINE) / `document_id` / `chunk_id` / `document_name` / `content` / `page`
- **Redis（会话）**：key 前缀 `agentic-rag:session:`，带 TTL

文件上传后在后台任务中：LlamaIndex 解析文件 → `SentenceSplitter` 切分（chunk_size=800，overlap=120）→ 向量化 → 写入 Milvus 并替换 MySQL 分块。

## 测试

```bash
.venv\Scripts\python.exe -m pytest
```

项目采用 pytest + pytest-asyncio（`asyncio_mode="auto"`），`testpaths=["tests"]`。目前覆盖统一响应信封契约与向量工具路由（阈值拒绝、禁用工具不被选中）。生产路径依赖外部服务，测试中用 Fake/Hash 适配器替换。可参考根目录 `_smoke_parallel.py` 验证多任务并行执行。

## 许可

> 项目当前未包含 `LICENSE` 文件。若计划开源到 GitHub，建议先选择许可证（如 [MIT](https://opensource.org/licenses/MIT) / Apache-2.0）并创建对应的 `LICENSE` 文件。

## 说明

本项目仍处于早期开发阶段（v0.1.0）。检索、评分的阈值请根据你的知识库与实际使用的模型在 `.env` 中调优。