# Agentic RAG

基于 FastAPI 实现的 Agentic RAG 知识库服务。它使用 LangGraph 进行编排，LangChain/OpenAI 处理模型调用，LlamaIndex 负责文件摄取，Milvus + BM25 实现混合检索，Redis 保存匿名会话状态，MySQL 存储文档与任务元数据。

## 流程

`改写(重写查询) -> 意图识别 -> 计划 DAG -> 任务调度 -> 向量工具路由 -> RAG 检索 -> 文档评分 -> 网络搜索 -> 生成回答 -> 回答评分`

路由器会嵌入(向量化)每个计划任务，并将其与所有已启用工具的能力描述进行比对。只有通过语义相似度匹配，并经过输入、能力与策略校验之后，工具才能被执行。

## 运行

1. 使用 Python 3.11 或 3.12 创建虚拟环境。
2. 安装依赖：`py -3.12 -m pip install -e ".[dev]"`
3. 将 `.env.example` 复制为 `.env`，并配置 MySQL、Redis、Milvus、OpenAI 和 Tavily。
4. 启动 API：`py -3.12 -m uvicorn app.main:app --reload`
5. 打开 `http://127.0.0.1:8000/docs`。

生产环境的请求需要已配置好的外部服务。模拟适配器(Fake adapters)仅在自动化测试中可用。

文件上传接口会在后台任务中解析并切分文档。MySQL 存储文档/任务/分块元数据，Milvus 存储稠密向量，BM25 对 MySQL 中的分块文本进行打分；检索器使用倒数排名融合(reciprocal rank fusion)融合稠密与稀疏排名结果，再交由文档评分器(Document Grader)评估。

## Vue 前端

Vue 3 前端位于 `frontend/`。开发时，Vite 会把 `/api` 请求代理到 FastAPI。

1. 先启动 FastAPI（端口 `8000`）。
2. 打开第二个终端，进入 `frontend/`。
3. 运行 `npm install`，再运行 `npm run dev`。
4. 打开 `http://127.0.0.1:5173`。

运行 `npm run build` 后，FastAPI 会将 `frontend/dist/` 的构建产物作为根页面提供。
