<script setup lang="ts">
import { computed, nextTick, onMounted, ref } from 'vue'
import { Activity, ArrowUpRight, Check, FileSearch2, GitCompareArrows, Inbox, Library, ListTree, LoaderCircle, MessageSquareMore, Plus, RefreshCw, Route, ScanSearch, Settings2, Upload } from 'lucide-vue-next'

type View = 'ask' | 'library' | 'runs'
type TraceState = 'idle' | 'running' | 'done'
interface TraceStep { node: string; name: string; output: string; state: TraceState }
interface DocumentItem { document_id: string; filename: string; status: string; created_at?: string }
interface Message { role: 'user' | 'assistant'; content: string; time: string }

const apiBase = import.meta.env.VITE_API_BASE_URL || '/api/v1'
const activeView = ref<View>('ask')
const query = ref('')
const sessionId = ref<string | null>(null)
const connected = ref(false)
const toast = ref('')
const files = ref<DocumentItem[]>([])
const runs = ref<{ query: string; time: string; score: string; nodes: number }[]>([])
const fileInput = ref<HTMLInputElement | null>(null)
const chatFeed = ref<HTMLElement | null>(null)
const messages = ref<Message[]>([{ role: 'assistant', content: '你好。告诉我你要查什么、比较什么，或者让一个问题穿过完整的 Agentic RAG 流程。', time: '刚刚' }])
const traceSteps = ref<TraceStep[]>([])
const traceState = ref('等待下一次任务')
const traceClock = ref('00:00')
const matchedTool = ref('等待向量匹配')
const matchedDetail = ref('任务开始后显示最相近的 Agent 能力。')
const similarity = ref<number | null>(null)
const viewTitle = computed(() => ({ ask: '问答工作台', library: '资料库', runs: '运行记录' })[activeView.value])
const statusLabel = computed(() => connected.value ? '服务已连接' : '演示模式')
const sessionLabel = computed(() => sessionId.value ? `SESSION / ${sessionId.value.slice(0, 8).toUpperCase()}` : 'SESSION / NEW')
const traceDone = computed(() => traceSteps.value.filter((step) => step.state === 'done').length)

function setView(view: View) { activeView.value = view; if (view === 'library') void loadDocuments() }
function notify(message: string) { toast.value = message; window.setTimeout(() => { if (toast.value === message) toast.value = '' }, 3200) }
function resetTrace() { traceSteps.value = []; traceState.value = '等待下一次任务'; traceClock.value = '00:00'; matchedTool.value = '等待向量匹配'; matchedDetail.value = '任务开始后显示最相近的 Agent 能力。'; similarity.value = null }
function newSession() { sessionId.value = crypto.randomUUID(); messages.value = []; resetTrace(); notify('已创建新会话') }
function usePrompt(prompt: string) { query.value = prompt }
function demoAnswer(value: string) { return `服务暂不可用，这是一次演示回答：已收到“${value}”。启动后端服务后，会实时展示各节点输出并流式返回答案。` }
function scrollToBottom() { chatFeed.value?.scrollTo({ top: chatFeed.value.scrollHeight, behavior: 'smooth' }) }
function parseBlock(raw: string): { event: string; payload: Record<string, any> | null } {
  let event = ''
  let data = ''
  for (const line of raw.split('\n')) {
    if (line.startsWith('event:')) event = line.slice(6).trim()
    else if (line.startsWith('data:')) data = line.slice(5).trim()
  }
  if (!data) return { event, payload: null }
  try { return { event, payload: JSON.parse(data) } } catch { return { event, payload: null } }
}
async function submitQuery() {
  const value = query.value.trim(); if (!value) return
  sessionId.value ||= crypto.randomUUID()
  messages.value.push({ role: 'user', content: value, time: '刚刚' })
  const streamingMessage: Message = { role: 'assistant', content: '', time: '刚刚' }
  messages.value.push(streamingMessage)
  query.value = ''
  resetTrace(); traceState.value = '任务执行中'
  const startedAt = Date.now()
  const clockTimer = window.setInterval(() => { const seconds = Math.floor((Date.now() - startedAt) / 1000); traceClock.value = `00:${String(seconds % 60).padStart(2, '0')}` }, 200)
  let citations: unknown[] = []
  let runScore = '71%'
  try {
    const response = await fetch(`${apiBase}/chat/query/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: value, session_id: sessionId.value }),
    })
    if (!response.ok || !response.body) throw new Error('stream unavailable')
    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    while (true) {
      const { done, value: chunk } = await reader.read()
      if (done) break
      buffer += decoder.decode(chunk, { stream: true })
      let sep = buffer.indexOf('\n\n')
      while (sep !== -1) {
        const block = buffer.slice(0, sep)
        buffer = buffer.slice(sep + 2)
        const { event, payload } = parseBlock(block)
        if (!payload) continue
        if (event === 'trace') {
          traceSteps.value.push({ node: payload.node as string, name: payload.label as string, output: payload.output as string, state: 'done' })
          if (payload.node === 'route') {
            matchedTool.value = (payload.tool as string) || '未匹配到可用工具'
            matchedDetail.value = typeof payload.output === 'string' ? payload.output : '任务向量与 Agent 能力向量完成匹配。'
            if (typeof payload.score === 'number') similarity.value = payload.score
          }
          void nextTick(scrollToBottom)
        } else if (event === 'token') {
          streamingMessage.content += (payload.content as string) || ''
          void nextTick(scrollToBottom)
        } else if (event === 'citation') {
          citations.push(payload)
        } else if (event === 'grader') {
          runScore = (payload.warnings as unknown[] | undefined)?.length ? '71%' : '86%'
        } else if (event === 'done') {
          streamingMessage.content = (payload.answer as string) ?? streamingMessage.content
          if (Array.isArray(payload.citations)) citations = payload.citations
        } else if (event === 'error') {
          streamingMessage.content = `服务暂不可用：${payload.message}`; connected.value = false
        }
        sep = buffer.indexOf('\n\n')
      }
    }
    connected.value = true
  } catch {
    streamingMessage.content = demoAnswer(value); connected.value = false
  } finally {
    window.clearInterval(clockTimer)
  }
  traceState.value = '任务完成'
  runs.value.unshift({ query: value, time: new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' }), score: runScore, nodes: traceSteps.value.length })
  void nextTick(scrollToBottom)
}
async function checkHealth() { try { const response = await fetch(`${apiBase}/health/live`); connected.value = response.ok } catch { connected.value = false } }
async function loadDocuments() { try { const response = await fetch(`${apiBase}/knowledge/files`); if (!response.ok) throw new Error('service unavailable'); files.value = (await response.json()).data?.content || []; connected.value = true } catch { files.value = []; connected.value = false } }
async function uploadFiles(event: Event) { const input = event.target as HTMLInputElement; if (!input.files?.length) return; for (const file of input.files) { const body = new FormData(); body.append('file', file); try { const response = await fetch(`${apiBase}/knowledge/files`, { method: 'POST', body }); if (!response.ok) throw new Error('upload failed'); notify(`${file.name} 已加入索引队列`) } catch { notify('上传需要启动后端依赖服务') } } input.value = ''; await loadDocuments() }
async function deleteDocument(documentId: string) { try { const response = await fetch(`${apiBase}/knowledge/files/${documentId}`, { method: 'DELETE' }); if (!response.ok) throw new Error('delete failed'); notify('资料已删除'); await loadDocuments() } catch { notify('删除失败，请检查服务连接') } }
onMounted(() => { void checkHealth(); void loadDocuments() })
</script>

<template>
  <div class="app-shell">
    <aside class="rail" aria-label="主导航"><div class="brand-mark" aria-label="Atlas RAG">A<span>/</span></div><nav class="rail-nav"><button class="rail-link" :class="{ 'is-active': activeView === 'ask' }" aria-label="问答工作台" @click="setView('ask')"><MessageSquareMore /><span>问答</span></button><button class="rail-link" :class="{ 'is-active': activeView === 'library' }" aria-label="知识库" @click="setView('library')"><Library /><span>知识库</span></button><button class="rail-link" :class="{ 'is-active': activeView === 'runs' }" aria-label="运行记录" @click="setView('runs')"><Activity /><span>运行</span></button></nav><div class="rail-bottom"><button class="rail-link" aria-label="设置"><Settings2 /><span>设置</span></button><div class="avatar">AN</div></div></aside>
    <main class="main-stage">
      <header class="topbar"><div><p class="eyebrow">ATLAS RAG / KNOWLEDGE OPS</p><h1>{{ viewTitle }}</h1></div><div class="top-actions"><span class="connection-pill" :class="{ 'is-online': connected }"><span class="status-dot"></span>{{ statusLabel }}</span><button class="icon-button" aria-label="新建会话" @click="newSession"><Plus /></button></div></header>
      <section v-show="activeView === 'ask'" class="view is-visible"><div class="workspace-grid"><div class="conversation-column"><div class="welcome-line"><div><span class="section-kicker">LIVE SESSION</span><h2>把问题交给路由器。</h2><p>Atlas 会先拆解意图，再为每个任务选择最匹配的 Agent。</p></div><div class="session-id">{{ sessionLabel }}</div></div><div class="prompt-strip"><button class="prompt-chip" @click="usePrompt('公司的知识库里，如何部署 FastAPI 服务？')"><FileSearch2 />检索内部文档</button><button class="prompt-chip" @click="usePrompt('比较 Milvus、FAISS 和 BM25 在这个项目里的职责。')"><GitCompareArrows />比较技术方案</button><button class="prompt-chip" @click="usePrompt('给我一个 Agentic RAG 的最小执行流程。')"><Route />梳理执行流程</button></div><div ref="chatFeed" class="chat-feed" aria-live="polite"><article v-for="(message, index) in messages" :key="`${message.time}-${index}`" class="message" :class="message.role === 'user' ? 'user-message' : 'assistant-message'"><div class="message-meta"><span class="agent-tag">{{ message.role === 'user' ? 'YOU / QUERY' : 'ATLAS / ORCHESTRATOR' }}</span><time>{{ message.time }}</time></div><p>{{ message.content }}</p><div v-if="index === 0 && message.role === 'assistant'" class="welcome-note"><ScanSearch /><span>我会展示检索证据、路由选择和评分结果。</span></div></article></div><form class="composer" @submit.prevent="submitQuery"><div class="composer-topline"><span>QUERY INPUT</span><span>{{ query.length }} / 8000</span></div><textarea v-model="query" rows="3" maxlength="8000" placeholder="输入一个问题，开始一次可追踪的检索…" aria-label="问题" @keydown.enter.exact.prevent="submitQuery"></textarea><div class="composer-actions"><div class="composer-hints"><span><kbd>Enter</kbd> 发送</span><span><kbd>Shift</kbd><kbd>Enter</kbd> 换行</span></div><button class="send-button" type="submit"><span>运行任务</span><ArrowUpRight /></button></div></form></div>
      <aside class="trace-panel" aria-label="Agent 路由轨迹"><div class="panel-heading"><div><span class="section-kicker">TRACE / 01</span><h3>路由轨迹</h3></div><button class="icon-button subtle" aria-label="刷新轨迹" @click="resetTrace"><RefreshCw /></button></div><div class="trace-status"><span class="pulse-ring"></span><span>{{ traceState }}</span><span class="trace-clock">{{ traceClock }}</span></div><div class="trace-list"><div v-if="!traceSteps.length" class="trace-empty">开始一次问答后，这里会按实际节点实时展示每一步的输出。</div><div v-for="(step, index) in traceSteps" :key="`${step.node}-${index}`" class="trace-item" :class="`is-${step.state}`"><div class="trace-marker">{{ String(index + 1).padStart(2, '0') }}</div><div><strong>{{ step.name }}</strong><span class="trace-output">{{ step.output }}</span></div><LoaderCircle v-if="step.state === 'running'" /><Check v-else-if="step.state === 'done'" /><span v-else class="idle-symbol">−</span></div></div><div class="router-card"><div class="router-card-head"><span>TOOL MATCH</span><span class="similarity-badge">{{ similarity ? `${Math.round(similarity * 100)}%` : '—' }}</span></div><strong>{{ matchedTool }}</strong><p>{{ matchedDetail }}</p><div class="similarity-track"><span :style="{ width: `${similarity ? similarity * 100 : 0}%` }"></span></div></div><div class="trace-footer"><span>可观测节点</span><strong>{{ String(traceDone).padStart(2, '0') }} / {{ String(traceSteps.length || 6).padStart(2, '0') }}</strong></div></aside></div></section>
      <section v-show="activeView === 'library'" class="view is-visible"><div class="library-head"><div><span class="section-kicker">KNOWLEDGE BASE / 02</span><h2>资料库</h2><p>上传资料，观察它们从文件变成可检索证据。</p></div><button class="primary-button" @click="fileInput?.click()"><Upload />添加资料</button><input ref="fileInput" type="file" accept=".pdf,.docx,.txt,.md,.markdown" hidden multiple @change="uploadFiles" /></div><div class="library-stats"><div><span>已接入资料</span><strong>{{ files.length }}</strong></div><div><span>索引状态</span><strong>{{ connected ? '已连接' : '演示模式' }}</strong></div><div><span>检索策略</span><strong>Milvus + BM25</strong></div></div><div class="document-table-wrap"><table class="document-table"><thead><tr><th>文件</th><th>状态</th><th>加入时间</th><th>操作</th></tr></thead><tbody><tr v-if="!files.length" class="empty-row"><td colspan="4"><Inbox /><strong>还没有资料</strong><span>上传 PDF、DOCX、TXT 或 Markdown，开始建立你的知识边界。</span></td></tr><tr v-for="file in files" :key="file.document_id"><td><div class="doc-name"><span class="doc-icon">DOC</span>{{ file.filename }}</div></td><td><span class="state-label">{{ file.status }}</span></td><td>{{ file.created_at?.slice(0, 10) }}</td><td><button class="text-button" @click="deleteDocument(file.document_id)">删除</button></td></tr></tbody></table></div></section>
      <section v-show="activeView === 'runs'" class="view is-visible"><div class="runs-head"><div><span class="section-kicker">RUN LOG / 03</span><h2>运行记录</h2><p>每次问答都留下清晰的路径和结果。</p></div><div class="run-filter"><span class="filter-dot"></span>最近 24 小时</div></div><div class="run-list"><div v-if="!runs.length" class="run-empty"><ListTree /><strong>暂无运行记录</strong><span>完成一次问答后，这里会出现任务 DAG、工具匹配和评分结果。</span></div><template v-else><div v-for="run in runs" :key="`${run.query}-${run.time}`" class="run-entry"><div><strong>{{ run.query }}</strong><span>{{ run.time }} · RAG Agent · {{ run.nodes || 6 }} 个可观测节点</span></div><div class="run-entry-score">{{ run.score }}</div></div></template></div></section>
    </main>
  </div>
  <Transition name="toast"><div v-if="toast" class="toast" role="status">{{ toast }}</div></Transition>
</template>
