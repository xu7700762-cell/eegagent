'use strict';

const $ = id => document.getElementById(id);
const domainNames = {shared: '共享知识库', vrms: 'VR 晕动症', fatigue: '疲劳', emotion: '情感与心理负荷'};
const defaultAgents = [
  {id: 'vrms', name: 'VRMSAgent', label: 'VR 晕动症'},
  {id: 'fatigue', name: 'FatigueAgent', label: '疲劳', model_status: 'tools_only'},
  {id: 'emotion', name: 'EmotionAgent', label: '情绪', model_status: 'tools_only'}
];
const stateNames = {idle: '就绪', running: '协作中', failed: '运行失败'};
const agentStatusNames = {waiting: '待命', pending: '待命', running: '分析中', completed: '已完成', done: '已完成', failed: '失败', unavailable: '不可用', skipped: '未参与'};
const eventNames = {user_message: '用户提问', domains_selected: '本轮协作领域', plan: 'Supervisor 制定计划', agent_status: 'Agent 状态', tool_result: '本地工具结果', retrieval: '分层 RAG 检索', agent_report: '领域报告完成', synthesis_review: 'Supervisor 逐项核验', synthesis: 'Supervisor 综合回答', error: '运行提示'};
const profileNames = {deepseek: 'DeepSeek', cpa_gpt: 'CPA GPT'};
const pageTitles = {conversation: '协作对话', evaluation: '实验评价', toolkits: '领域工具包', knowledge: '分层知识库'};
const pageFromHash = () => Object.hasOwn(pageTitles, location.hash.slice(1)) ? location.hash.slice(1) : 'conversation';
const initialPage = pageFromHash();
const defaultSeed = 2026;
let catalog = null, vrmsCatalog = null, session = null, sessionId = null, lastSequence = -1;
let stream = null, retryTimer = null, currentTurn = null, submitting = false, running = false, recovering = false;
let initializing = true;
let apiConfigured = null, cloudSettings = null, latestGeneration = null, cloudSwitching = false;
let evaluation = null, evaluationId = null, evaluationTimer = null, evaluationSubmitting = false;
const evaluationTerminal = new Set(['completed', 'failed', 'stopped', 'cancelled', 'interrupted']);
let tools = [], citations = [], activity = [], agentReports = [], latestReport = null;
let synthesisReview = null;
let recordings = [], uploading = false;
const messageKeys = new Set();
const agentStates = new Map();

// Presentation aliases leave stored identifiers, checkpoints and measurements intact.
function displayText(value) { return String(value); }
function displayEvidence(value) {
  if (Array.isArray(value)) return value.map(displayEvidence);
  if (value && typeof value === 'object' && value.tool === 'vrms.raw_model' && value.gpt_requested) {
    value = {...value, measurements: {...value.measurements, segments: list(value.measurements?.segments).map(row => {
      const view = {...row}; delete view.numeric_fusion; delete view.mil_class; delete view.mil_raw_high_probability;
      if (view.cloud_judgment) {
        view.cloud_judgment = {...view.cloud_judgment};
        delete view.cloud_judgment.rounds; delete view.cloud_judgment.tool_outputs;
      }
      return view;
    })}};
  }
  if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value)
    .filter(([key]) => !/seed/i.test(key))
    .map(([key, item]) => [displayText(key), displayEvidence(item)]));
  return typeof value === 'string' ? displayText(value) : value;
}
function displayReport(value) { return String(value || ""); }
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = displayText(text);
  return node;
}
function icon(name) {
  const node = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  node.setAttribute('class', 'icon');
  node.setAttribute('aria-hidden', 'true');
  const use = document.createElementNS('http://www.w3.org/2000/svg', 'use');
  use.setAttribute('href', '#i-' + name); node.append(use); return node;
}
function asText(value) {
  if (value === null || value === undefined) return '';
  return typeof value === 'string' ? displayText(value) : JSON.stringify(displayEvidence(value), null, 2);
}
function list(value) {
  if (Array.isArray(value)) return value;
  if (value && typeof value === 'object') return Object.entries(value).map(([id, item]) => typeof item === 'object' && item !== null ? {id, ...item} : {id, text: item});
  return [];
}
function selectedDomains(name = 'domain') {
  return Array.from(document.querySelectorAll('input[name="' + name + '"]:checked')).map(input => input.value);
}
function questionDomains(text = $('question').value) {
  const patterns = {vrms: /vrms|眩晕|晕动症|晕动|晕车|cybersickness/i, fatigue: /fatigue|疲劳/i, emotion: /emotion|心理负荷|工作负荷|情绪|情感|mental\s*workload/i};
  const only = text.match(/(?:只|仅)(?:让|由|调用|使用|运行|分析|问|回答|需要)?\s*(?:VRMS(?:Agent)?|Fatigue(?:Agent)?|Emotion(?:Agent)?|眩晕|晕动症|疲劳|心理负荷|工作负荷|情绪|情感)[^，,。；;\n！？!?]*/i);
  const mentioned = Object.keys(patterns).filter(id => patterns[id].test(only ? only[0] : text));
  if (only) return mentioned;
  if (/(?:三|3)\s*(?:个|位)?\s*agents?\b|(?:全部|所有)\s*(?:领域\s*)?agents?\b|(?:三个|3\s*个|全部|所有)\s*领域|all\s+(?:three\s+)?agents?\b/i.test(text)) return defaultAgents.map(agent => agent.id);
  const selected = selectedDomains();
  return defaultAgents.map(agent => agent.id).filter(id => selected.includes(id) || mentioned.includes(id));
}
function applyDomains(domains) {
  document.querySelectorAll('input[name="domain"]').forEach(input => { input.checked = domains.includes(input.value); });
  if (session) session.domains = domains;
  renderAgentNodes(); updateControls();
}
function selectedSeed() { return defaultSeed; }
function selectedPipeline() { return 'raw_eeg_gpt_tools_seed2026'; }
function vrmsCatalogUrl() { return '/api/brain/vrms/catalog?seed=' + selectedSeed(); }
function agentName(id) { return (catalog?.agents || defaultAgents).find(agent => agent.id === id)?.name || id || 'Agent'; }
function storage(key, value) {
  try {
    if (value === undefined) return localStorage.getItem(key);
    if (value === null) localStorage.removeItem(key); else localStorage.setItem(key, value);
  } catch (_) { /* Conversation recovery remains optional when browser storage is disabled. */ }
  return null;
}
async function request(url, options) {
  const response = await fetch(url, options);
  let value;
  try { value = await response.json(); } catch (_) { throw Error('服务未返回有效数据（HTTP ' + response.status + '）。'); }
  if (!response.ok) throw Error(typeof value.detail === 'string' ? value.detail : asText(value.detail || value.error || value));
  return value;
}
function post(url, payload) {
  return request(url, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
}
function showError(error) { $('globalError').textContent = displayText(error.message || String(error)); $('globalError').hidden = false; }
function clearError() { $('globalError').textContent = ''; $('globalError').hidden = true; }
function safeUrl(value) {
  if (typeof value !== 'string' || !/^https?:\/\//i.test(value)) return null;
  try { const url = new URL(value); return ['http:', 'https:'].includes(url.protocol) ? url.href : null; } catch (_) { return null; }
}
function sourceLink(value, text) {
  const url = safeUrl(value);
  if (!url) return el('span', '', /(?:vrms|seed)/i.test(text || value) ? '本地研究资料（来源已登记）' : text || value);
  const node = el('a', '', text || value);
  node.href = url; node.target = '_blank'; node.rel = 'noopener noreferrer'; return node;
}

// Render a small Markdown subset as DOM nodes. Model output never enters innerHTML.
function inlineMarkdown(node, text) {
  const pattern = /(\*\*([^*]+)\*\*|`([^`]+)`|\[([^\]]+)\]\((https?:\/\/[^\s)]+)\))/g;
  let match, position = 0;
  while ((match = pattern.exec(text))) {
    node.append(document.createTextNode(text.slice(position, match.index)));
    if (match[2]) node.append(el('strong', '', match[2]));
    else if (match[3]) node.append(el('code', '', match[3]));
    else node.append(sourceLink(match[5], match[4]));
    position = match.index + match[0].length;
  }
  node.append(document.createTextNode(text.slice(position)));
}
function markdown(node, text) {
  const lines = displayReport(text).replace(/\r/g, '').split('\n');
  let paragraph = [], code = null, listNode = null;
  const flush = () => {
    if (paragraph.length) { const p = el('p'); inlineMarkdown(p, paragraph.join('\n')); node.append(p); paragraph = []; }
    listNode = null;
  };
  for (const line of lines) {
    if (/^\s*```/.test(line)) {
      flush();
      if (code !== null) { const pre = el('pre'); pre.append(el('code', '', code.join('\n'))); node.append(pre); code = null; } else code = [];
      continue;
    }
    if (code !== null) { code.push(line); continue; }
    if (!line.trim()) { flush(); continue; }
    const heading = line.match(/^(#{1,4})\s+(.+)$/);
    if (heading) { flush(); const h = el('h' + Math.min(4, heading[1].length + 1)); inlineMarkdown(h, heading[2]); node.append(h); continue; }
    const bullet = line.match(/^\s*(?:[-*]\s+|\d+[.)]\s+)(.+)$/);
    if (bullet) {
      if (paragraph.length) flush();
      const tag = /^\s*\d/.test(line) ? 'ol' : 'ul';
      if (!listNode || listNode.tagName.toLowerCase() !== tag) { listNode = el(tag); node.append(listNode); }
      const item = el('li'); inlineMarkdown(item, bullet[1]); listNode.append(item); continue;
    }
    if (listNode) listNode = null;
    paragraph.push(line);
  }
  flush();
  if (code !== null) { const pre = el('pre'); pre.append(el('code', '', code.join('\n'))); node.append(pre); }
}

function generationInfo(message) {
  const raw = message?.generation_status || message?.result?.generation_status;
  const status = typeof raw === 'object' ? raw.status : raw;
  if (message?.simulated || status === 'mock' || session?.backend === 'mock') return {status: 'mock', text: '模拟语言响应'};
  if (status === 'cloud_verified') return {status, text: '真实 API 已生成' + ((message?.workflow_status || message?.result?.workflow_status) === 'partial_fallback' ? ' · 部分 Agent 回退' : '')};
  if (status === 'fallback' || status === 'local_fallback') return {status: 'fallback', text: 'API 失败 · 本地回退'};
  return {status: 'unverified', text: '生成来源未验证'};
}
function displayBackend(value) {
  const api = value === 'api', generated = api && !running && !submitting ? latestGeneration : null;
  const text = !api ? 'mock · 模拟语言响应' : generated ? generated.text : apiConfigured === false ? '真实 API · 待配置' : running || submitting ? '真实 API · 正在调用' : '真实 API · 等待生成';
  $('backendBadge').className = 'badge ' + (generated?.status === 'fallback' || generated?.status === 'unverified' ? 'fallback' : api ? 'api' : 'mock');
  $('backendBadge').replaceChildren(el('span', 'status-dot'), document.createTextNode(text));
  $('backendNote').textContent = !api ? 'mock 是显式调试模式：Supervisor 回复为模拟文本；本地工具和检索仍使用实际证据。' : apiConfigured === false ? '请在左侧“API 与模型配置”保存可用 API；当前不会自动切换为 mock。' : generated?.status === 'fallback' ? '本轮云端调用失败，显示本地回退报告；回复不属于真实 Supervisor 云端生成。' : 'Supervisor 调用已配置 API，基于匿名工具摘要与检索证据回答；每轮显示实际生成状态。';
}
function renderCloudSettings(value) {
  cloudSettings = value; apiConfigured = !!value.configured;
  const option = $('backend').querySelector('option[value="api"]'); option.textContent = apiConfigured ? '真实 API' : '真实 API · 需配置';
  $('cloudProvider').value = value.active_profile || 'deepseek';
  for (const option of $('cloudProvider').options) {
    const profile = value.profiles?.[option.value], ready = profile ? profile.configured : option.value === (value.active_profile || 'deepseek') && value.configured;
    option.disabled = !ready;
    option.textContent = profileNames[option.value] + (ready ? '' : ' · 需填写独立密钥');
  }
  $('cloudModelSummary').textContent = (profileNames[value.active_profile] || '当前 API') + '：' + (value.model_id || '未配置');
  $('cloudConnectionSummary').textContent = apiConfigured ? '接口配置已保存 · 生成时验证连接' : '接口配置未完整 · 需保存地址、型号和密钥';
  $('cloudConnectionSummary').classList.toggle('unavailable', !apiConfigured);
  updateControls();
}
async function switchCloudProvider() {
  if (cloudSwitching || running || submitting) return;
  const selected = $('cloudProvider').value, profile = cloudSettings?.profiles?.[selected];
  if (!profile?.configured) { $('cloudProvider').value = cloudSettings?.active_profile || 'deepseek'; showError('请先在设置页填写此模型的独立接口与密钥，再切换。'); return; }
  cloudSwitching = true; clearError(); updateControls();
  try {
    const value = await request('/api/cloud-settings', {method: 'PUT', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({active_profile: selected})});
    latestGeneration = null; renderCloudSettings(value);
    renderVrmsCatalog(await request(vrmsCatalogUrl()));
  } catch (error) { $('cloudProvider').value = cloudSettings?.active_profile || 'deepseek'; showError(error); }
  finally { cloudSwitching = false; updateControls(); }
}
async function refreshCloudConfig() {
  $('refreshCloudConfig').disabled = true; clearError();
  try {
    const values = await Promise.allSettled([request('/api/cloud-settings'), request(vrmsCatalogUrl())]);
    if (values[0].status === 'fulfilled') renderCloudSettings(values[0].value); else showError(values[0].reason);
    if (values[1].status === 'fulfilled') renderVrmsCatalog(values[1].value); else renderVrmsCatalog({status: 'unavailable', reason: 'VRMS 目录暂不可用：' + values[1].reason.message});
  } finally { $('refreshCloudConfig').disabled = false; }
}
function updateControls() {
  const busy = submitting || running || cloudSwitching || uploading;
  $('sendQuestion').disabled = initializing || busy || !$('question').value.trim() || !questionDomains().length || ($('backend').value === 'api' && apiConfigured === false);
  $('newConversation').disabled = initializing || busy;
  $('newRealConversation').disabled = initializing || busy;
  $('backend').disabled = initializing || busy;
  $('cloudProvider').disabled = initializing || busy || !!cloudSettings?.in_use || evaluationActive();
  $('vrmsSeed').disabled = initializing || busy || evaluationSubmitting || evaluationActive();
  $('researchSeed').disabled = $('vrmsSeed').disabled;
  $('researchSeed').value = $('vrmsSeed').value = String(defaultSeed);
  for (const id of ['taskEvidence', 'vrmsPath']) $(id).disabled = initializing || busy || !!sessionId || (id === 'vrmsPath' && vrmsCatalog?.status !== 'ready');
  $('recordingSource').disabled = initializing || busy || !!sessionId;
  $('uploadEdfButton').disabled = initializing || busy;
  document.querySelectorAll('input[name="domain"]').forEach(input => { input.disabled = initializing || busy || !!sessionId; });
  document.querySelectorAll('.history-item').forEach(button => { button.disabled = initializing || (busy && button.dataset.id !== sessionId); });
  $('working').hidden = !(submitting || running);
  $('workingText').textContent = submitting ? '正在提交问题…' : $('workingText').textContent;
  $('conversationState').className = 'state-label' + (busy ? ' running' : session?.state === 'failed' ? ' failed' : '');
  $('conversationState').replaceChildren(el('span', 'status-dot'), document.createTextNode(initializing ? '正在恢复…' : busy ? '协作中' : stateNames[session?.state] || '就绪'));
  const plannedDomains = $('question').value.trim() ? questionDomains() : selectedDomains();
  const adjusted = plannedDomains.join(',') !== selectedDomains().join(',');
  $('scopeLabel').textContent = plannedDomains.length + ' 个领域' + (adjusted ? ' · 按问题调整' : '');
  $('downloadReport').disabled = !latestReport;
  displayBackend(session?.backend || $('backend').value);
  $('sessionModeNotice').hidden = !sessionId || (session?.backend || $('backend').value) !== 'mock';
  const recordingId = session?.recording_id || $('recordingSource').value;
  const recording = recordings.find(item => item.id === recordingId);
  $('sessionContext').textContent = [sessionId ? '当前对话：' + sessionId : '', recordingId ? (recording?.title || recordingId) + ' · 原始 EEG' : '在对话中指定被试或文件即可分析'].filter(Boolean).join(' · ');
  updateEvaluationControls();
  $('parameterSummary').textContent = plannedDomains.length + ' 个领域' + (adjusted ? ' · 按问题调整' : '') + ' · ' + (recordingId ? '原始 EEG 已关联' : '自然语言指定原始 EEG');
  {
    $('vrmsPipelineBadge').textContent = '原始 EEG 工具';
    $('vrmsPipelineNote').textContent = '原始 EEG 重新编码 → MIL 与八路工具 → GPT 工具 Agent 最终判断；路径按 mark20→mark22、休息段按 mark22→下一次 mark20 划分；直接比较疲劳和心理负荷指标。问卷仅用于事后评价。';
    $('vrmsPipelineNote').classList.remove('unavailable');
    $('vrmsBoundaryPanel').hidden = false;
    $('vrmsBoundaryText').replaceChildren(el('p', '', '高类时间定位到整路径区间。分类使用整条路径数据，在路径结束后形成。'), el('p', '', 'VRMS 输入须匹配1024Hz、30个指定EEG通道与M1/M2参考；其他EDF仍可运行指标工具。'));
  }
  $('evaluationConfig').textContent = 'seed2026 · 146 条评分路径 · 原始 EEG\n云端模型：' + (cloudSettings?.model_id || '未配置');
  if (cloudSettings) $('cloudConnectionSummary').textContent = latestGeneration?.status === 'cloud_verified' ? latestGeneration.text : latestGeneration?.status === 'fallback' ? 'Supervisor 本轮 API 失败 · 已回退' : apiConfigured ? '接口配置已保存 · 生成时验证连接' : '接口配置未完整 · 需保存地址、型号和密钥';
}
function switchPage(page, updateHash = true) {
  if (!Object.hasOwn(pageTitles, page)) page = 'conversation';
  document.querySelectorAll('.page').forEach(node => { node.hidden = node.id !== page; });
  document.querySelectorAll('button.nav-item').forEach(button => {
    const active = button.dataset.page === page;
    button.classList.toggle('active', active); if (active) button.setAttribute('aria-current', 'page'); else button.removeAttribute('aria-current');
  });
  $('pageTitle').textContent = pageTitles[page];
  document.body.dataset.page = page;
  document.title = pageTitles[page] + ' · BrainAgent 科研工作台';
  $('traceToggle').hidden = page !== 'conversation';
  closePanels(false);
  if (updateHash && location.hash !== '#' + page) history.pushState(null, '', '#' + page);
  window.scrollTo(0, 0);
}
function closePanels(restoreFocus = true) {
  const menu = $('appShell').classList.contains('sidebar-open');
  const trace = $('appShell').classList.contains('inspector-open');
  $('appShell').classList.remove('sidebar-open', 'inspector-open');
  $('menuButton').setAttribute('aria-expanded', 'false');
  $('traceToggle').setAttribute('aria-expanded', 'false');
  if (restoreFocus && (menu || trace)) $(menu ? 'menuButton' : 'traceToggle').focus();
}
function openPanel(name) {
  closePanels(false);
  $('appShell').classList.add(name + '-open');
  $(name === 'sidebar' ? 'menuButton' : 'traceToggle').setAttribute('aria-expanded', 'true');
  $(name === 'sidebar' ? 'closeMenu' : 'closeInspector').focus();
}
function switchInspector(tab) {
  document.querySelectorAll('.inspector-tab').forEach(button => {
    const active = button.dataset.tab === tab;
    button.classList.toggle('active', active); button.setAttribute('aria-selected', String(active)); button.tabIndex = active ? 0 : -1;
  });
  for (const name of ['activity', 'tools', 'citations']) $(name + 'Panel').hidden = name !== tab;
}
function showCitations(items) {
  if (items.length) { citations = mergeCitations(citations, items); renderCitations($('citationsPanel'), citations); updateCounts(); }
  switchInspector('citations');
  if (window.matchMedia('(max-width: 1279px)').matches) openPanel('inspector');
}
function scrollChat(force = false) {
  const node = $('chatScroll');
  if (force || node.scrollHeight - node.scrollTop - node.clientHeight < 180) node.scrollTop = node.scrollHeight;
}
function usesRetiredRawReference(message) {
  const context = message.measurement_context || message.result?.measurement_context || {};
  const rawTools = list(message.agent_reports || message.result?.agent_reports).flatMap(report => list(report.tool_results))
    .filter(item => /^(?:vrms\.raw_|fatigue\.raw_|emotion\.raw_)/.test(item.tool));
  const policies = [context, message, message.result || {}, ...rawTools.map(item => item.result?.measurements || {})];
  if (policies.some(value => value.comparison_policy === 'event_segments_only' || value.baseline_policy === 'disabled')) return false;
  if (context.source !== 'raw_recording' && !rawTools.length) return false;
  return rawTools.some(item => Object.keys(item.result?.measurements || {}).some(key => /^baseline_(?:value|change|scope|interval)$/.test(key))) ||
    /基线|baseline/i.test(String(message.text || message.markdown || ''));
}
function addMessage(message, fallbackKey) {
  const role = message.role === 'user' ? 'user' : 'assistant';
  const key = message.turn_id ? message.turn_id + ':' + role : fallbackKey || role + ':' + asText(message.text);
  if (messageKeys.has(key)) return; messageKeys.add(key);
  $('welcome').hidden = true;
  const node = el('article', 'message ' + role);
  node.append(el('div', 'message-avatar', role === 'user' ? '你' : 'S'));
  const main = el('div', 'message-main'), meta = el('div', 'message-meta', role === 'user' ? '你' : 'Supervisor');
  if (role === 'assistant') {
    latestGeneration = generationInfo(message);
    meta.append(el('span', 'subtle-badge generation-' + latestGeneration.status, latestGeneration.text));
    if (message.provider?.model_id) meta.append(el('span', 'subtle-badge', message.provider.model_id));
  }
  main.append(meta);
  if (role === 'assistant' && usesRetiredRawReference(message)) {
    const notice = el('p', 'model-status unavailable', '旧分析使用记录开头参考，已停用，请重新提问。以下保留当时的回答与工具结果。');
    notice.setAttribute('role', 'note'); main.append(notice);
  }
  const body = el('div', 'message-text');
  if (role === 'user') body.textContent = message.text || '';
  else markdown(body, message.text || message.markdown || '暂无回答文本。');
  main.append(body);
  const refs = list(message.citations);
  if (refs.length) { const buttons = el('div', 'message-tools'), button = el('button', 'reference-button', refs.length + ' 条引用证据 ↗'); button.type = 'button'; button.onclick = () => showCitations(refs); buttons.append(button); main.append(buttons); }
  const reports = list(message.agent_reports);
  if (reports.length) {
    const detail = el('details', 'agent-reports'); detail.append(el('summary', '', '查看 ' + reports.length + ' 份领域 Agent 子报告'));
    for (const report of reports) { const heading = el('h4', '', report.name || agentName(report.agent || report.id)), status = generationInfo(report); heading.append(el('span', 'subtle-badge generation-' + status.status, status.text)); detail.append(heading); const reportBody = el('div', 'message-text'); markdown(reportBody, asText(report.report || report.markdown || report.text || report.summary)); detail.append(reportBody); }
    main.append(detail);
  }
  node.append(main); $('messages').append(node);
  if (role === 'assistant') latestReport = message.markdown || message.text;
  else if ($('conversationTitle').textContent === '新对话') $('conversationTitle').textContent = message.text.slice(0, 32);
  scrollChat(true); updateControls();
}
function panelEmpty(container, title, description, name) {
  container.textContent = ''; const block = el('div', 'panel-empty'); block.append(icon(name), el('strong', '', title), el('span', '', description)); container.append(block);
}
function updateCounts() {
  $('activityCount').textContent = activity.length; $('toolsCount').textContent = tools.length; $('citationsCount').textContent = citations.length; $('traceCount').textContent = activity.length;
}
function resetTrace() {
  tools = []; citations = []; activity = []; agentReports = []; synthesisReview = null; agentStates.clear();
  panelEmpty($('activityPanel'), '过程将在这里展开', '发送问题后，查看每个 Agent 的调用与汇总。', 'trace');
  panelEmpty($('toolsPanel'), '等待本地工具结果', '没有 EEG 数据时，数值测量会明确返回证据不足。', 'tool');
  panelEmpty($('citationsPanel'), '等待检索来源', '共享库与专业库的引用会分别标注。', 'book');
  for (const id of ['planNode', 'ragNode', 'synthesisNode']) $(id).classList.remove('active', 'done');
  $('planStatus').textContent = '待命'; $('planDescription').textContent = '理解问题 · 制定协作计划';
  $('ragStatus').textContent = '待检索'; $('ragDescription').textContent = '共享方法与协议 → 各领域专业证据';
  $('synthesisStatus').textContent = '待汇总'; $('synthesisDescription').textContent = '比较结论，保留各领域证据边界';
  renderAgentNodes(); updateCounts();
}
function beginTurn(id) { if (id && id !== currentTurn) { currentTurn = id; resetTrace(); } }
function reviewDescription(review) {
  return '已核验 ' + review.verified_items + '/' + review.requested_items + ' 项' + (list(review.topics).length ? ' · ' + list(review.topics).join('、') : '');
}
function reviewCompleted(review) {
  return review && Number.isInteger(review.requested_items) && review.requested_items > 0 && review.verified_items === review.requested_items;
}
function renderAgentNodes() {
  const root = $('agentNodes'); root.textContent = '';
  const domains = session?.domains || selectedDomains();
  for (const agent of catalog?.agents || defaultAgents) {
    const status = agentStates.get(agent.id) || 'waiting';
    const node = el('div', 'agent-node ' + agent.id + (domains.includes(agent.id) ? '' : ' unselected'));
    node.classList.toggle('active', ['running', 'retrieving', 'analyzing'].includes(status));
    node.classList.toggle('done', ['completed', 'done'].includes(status));
    node.append(el('span', 'node-icon', agent.id === 'vrms' ? 'V' : agent.id === 'fatigue' ? 'F' : 'E'), el('strong', '', agent.name), el('small', '', agent.label || domainNames[agent.id]), el('span', 'agent-status', domains.includes(agent.id) ? agentStatusNames[status] || status : '未参与'));
    root.append(node);
  }
}
function eventDescription(event) {
  const p = event.payload || {};
  if (event.kind === 'domains_selected') return list(p.domains).map(agentName).join('、') + (list(p.added_domains).length ? ' · 已按问题补齐参与领域' : '');
  if (event.kind === 'plan') return p.explanation || (list(p.agents).length ? list(p.agents).map(agent => agent.name || agentName(agent.id)).join('、') : 'Supervisor 分派领域 Agent。');
  if (event.kind === 'agent_status') return (p.name || agentName(p.agent)) + ' · ' + (agentStatusNames[p.status] || p.status);
  if (event.kind === 'tool_result') return agentName(p.agent) + ' → ' + asText(p.tool?.name || p.tool?.id || p.tool) + ' · ' + toolStatus(p.result);
  if (event.kind === 'retrieval') return agentName(p.agent) + ' · ' + list(p.results || p.matches).length + ' 条知识证据';
  if (event.kind === 'agent_report') return (p.name || agentName(p.agent)) + ' 已提交领域分析与证据边界。';
  if (event.kind === 'synthesis_review') return reviewDescription(p);
  if (event.kind === 'synthesis') return generationInfo(p).text + ' · 综合 ' + list(p.agent_reports).length + ' 份子报告 · ' + list(p.citations).length + ' 条引用';
  if (event.kind === 'error') return p.message || p.reason || p.error || '运行失败';
  return p.text || '';
}
function appendActivity(event) {
  if (!eventNames[event.kind]) return;
  activity.push(event);
  const root = $('activityPanel'); if (activity.length === 1) root.textContent = '';
  const item = el('div', 'activity-item'); item.append(el('span', 'activity-dot', event.kind === 'tool_result' ? 'T' : event.kind === 'retrieval' ? 'R' : '·'));
  const main = el('div', 'activity-main'); main.append(el('strong', '', eventNames[event.kind]), el('p', '', eventDescription(event)));
  if (event.timestamp || event.created_at || event.time) {
    const rawTime = event.timestamp || event.created_at || event.time;
    const date = new Date(typeof rawTime === 'number' ? rawTime * 1000 : rawTime);
    if (!Number.isNaN(date.getTime())) main.append(el('time', '', date.toLocaleTimeString('zh-CN', {hour: '2-digit', minute: '2-digit', second: '2-digit'})));
  }
  item.append(main); root.append(item); updateCounts();
}
function toolStatus(result) {
  if (!result) return '无结果';
  const names = {ok: '已完成', available: '已完成', completed: '已完成', success: '已完成', partial_fallback: '部分 GPT 判断未完成', unavailable: '证据不可用', insufficient: '证据不足', missing_data: '缺少 EEG 数据', missing_model: '模型未部署', model_unavailable: '模型未部署', error: '工具失败'};
  const status = result.status || result.availability || (result.available === false ? 'unavailable' : null);
  return names[status] || status || (result.reason ? result.reason : '已返回结果');
}
function renderTools() {
  const root = $('toolsPanel'); if (!tools.length) return; root.textContent = '';
  for (const tool of tools) {
    const state = toolStatus(tool.result), unavailable = /unavailable|insufficient|missing|error/.test(tool.result?.status || '') || tool.result?.available === false;
    const detail = el('details', 'tool-result' + (unavailable ? ' unavailable' : ''));
    const summary = el('summary'); summary.append(el('strong', '', asText(tool.tool?.name || tool.tool?.id || tool.tool)), el('span', '', agentName(tool.agent) + ' · ' + state));
    detail.append(summary);
    if (tool.tool === 'vrms.raw_model' && tool.result?.gpt_requested) {
      detail.append(el('div', '', 'GPT 工具 Agent：' + (tool.result.gpt_successful_paths || 0) + ' 条完成，' + (tool.result.gpt_failed_paths || 0) + ' 条未完成'));
      for (const row of list(tool.result.measurements?.segments)) {
        const judgment = row.cloud_judgment || {};
        detail.append(el('div', '', '第' + row.ordinal + '条 · GPT 最终 ' + (judgment.available ? judgment.final_class : '未发布') +
          (judgment.available ? ' · 置信度 ' + ({low: '低', medium: '中', high: '高'}[judgment.confidence] || '未提供') : '')));
      }
    }
    detail.append(el('pre', '', asText(tool.result))); root.append(detail);
  }
  updateCounts();
}
function citationKey(item) { return item.id || item.citation_id || item.chunk_id || [item.domain, item.source, item.title, item.excerpt || item.text].join('|'); }
function mergeCitations(existing, incoming) {
  const seen = new Set(existing.map(citationKey)), output = existing.slice();
  for (const item of incoming) { const key = citationKey(item); if (!seen.has(key)) { seen.add(key); output.push(item); } }
  return output;
}
function renderCitations(root, items, emptyText = '本次检索未找到匹配证据。') {
  root.textContent = '';
  if (!items.length) { panelEmpty(root, '暂无匹配来源', emptyText, 'book'); return; }
  items.forEach((item, index) => {
    const node = el('article', 'citation-card'), top = el('div', 'citation-top');
    const domain = item.domain || item.knowledge_domain || (item.layer === 'shared' || item.layer === 'L1' ? 'shared' : null);
    top.append(el('span', 'citation-number', item.id || item.citation_id || '[' + (index + 1) + ']'), el('span', 'citation-layer', domain === 'shared' ? 'L1 · 共享知识库' : 'L2 · ' + (domainNames[domain] || item.layer || '领域知识库')));
    node.append(top, el('h3', '', item.title || item.document_title || item.source || '知识条目'));
    const excerpt = item.excerpt || item.text || item.chunk || item.content;
    if (excerpt) node.append(el('p', '', asText(excerpt).slice(0, 1800)));
    const source = el('div', 'citation-source');
    if (item.url || item.source) { source.append(document.createTextNode('来源：'), sourceLink(item.url || item.source, item.source || item.url)); }
    if (item.doi) { if (source.childNodes.length) source.append(document.createTextNode(' · ')); source.append(sourceLink('https://doi.org/' + encodeURIComponent(item.doi.replace(/^https?:\/\/doi\.org\//, '')), 'DOI: ' + item.doi)); }
    if (source.childNodes.length) node.append(source);
    const meta = [item.section, item.page ? '页 ' + asText(item.page) : '', item.version ? '版本 ' + item.version : ''].filter(Boolean).join(' · ');
    if (meta) node.append(el('div', 'citation-meta', meta)); root.append(node);
  });
}
function handleEvent(event, replay = false) {
  const sequence = Number(event.sequence);
  if (Number.isFinite(sequence) && sequence <= lastSequence) return;
  if (Number.isFinite(sequence)) lastSequence = sequence;
  const p = event.payload || {}; beginTurn(p.turn_id);
  if (event.kind === 'domains_selected' && Array.isArray(p.domains)) applyDomains(p.domains);
  if (event.kind === 'recording_selected') {
    if (session) { session.recording_id = p.recording_id; session.input_policy = 'raw_eeg_only'; session.vrms_path_id = session.task_id = null; }
    if (!recordings.some(item => item.id === p.recording_id)) recordings.push({id: p.recording_id, title: p.title});
    renderRecordings(recordings); $('recordingSource').value = p.recording_id;
    $('vrmsPath').value = $('taskEvidence').value = ''; updateControls();
    $('workingText').textContent = '正在从原始 EEG 提取事件、质控与工具输入…';
  }
  if (event.kind === 'user_message') addMessage({role: 'user', text: p.text, turn_id: p.turn_id}, 'user:' + sequence);
  if (event.kind === 'plan') {
    $('planNode').classList.add('done'); $('planStatus').textContent = '已分派'; $('planDescription').textContent = displayText(p.explanation || '领域 Agent 并行分析与检索。');
    $('workingText').textContent = '专业 Agent 正在并行调用工具…';
  }
  if (event.kind === 'agent_status') { agentStates.set(p.agent, p.status); renderAgentNodes(); }
  if (event.kind === 'tool_result') { tools.push(p); renderTools(); }
  if (event.kind === 'retrieval') {
    const incoming = list(p.results || p.matches); citations = mergeCitations(citations, incoming); renderCitations($('citationsPanel'), citations);
    $('ragNode').classList.add('done'); $('ragStatus').textContent = citations.length + ' 条证据'; $('ragDescription').textContent = '共享知识 + 领域知识检索已返回；引用保留各自来源。';
  }
  if (event.kind === 'agent_report') { agentReports.push(p); agentStates.set(p.agent, 'completed'); renderAgentNodes(); $('workingText').textContent = 'Supervisor 正在核对证据并汇总回答…'; $('synthesisNode').classList.add('active'); $('synthesisStatus').textContent = '汇总中'; }
  if (event.kind === 'synthesis_review') {
    synthesisReview = p;
    $('synthesisNode').classList.add('active'); $('synthesisStatus').textContent = reviewCompleted(p) ? '核验已完成' : '核验中';
    $('synthesisDescription').textContent = reviewDescription(p);
    $('workingText').textContent = 'Supervisor ' + reviewDescription(p) + '，正在形成综合回答…';
  }
  if (event.kind === 'synthesis') {
    const incoming = list(p.citations); citations = mergeCitations(citations, incoming); if (citations.length) renderCitations($('citationsPanel'), citations);
    addMessage({role: 'assistant', text: p.text || p.markdown, markdown: p.markdown, turn_id: p.turn_id, citations: incoming, agent_reports: p.agent_reports || agentReports, measurement_context: p.measurement_context || p.result?.measurement_context, simulated: p.simulated, generation_status: p.generation_status || p.result?.generation_status, workflow_status: p.workflow_status || p.result?.workflow_status, provider: p.provider || p.result?.provider}, 'assistant:' + sequence);
    synthesisReview = p.synthesis_review || p.result?.synthesis_review || synthesisReview;
    const generation = generationInfo(p);
    $('synthesisNode').classList.remove('active'); $('synthesisNode').classList.add('done'); $('synthesisStatus').textContent = generation.text;
    $('synthesisDescription').textContent = generation.status === 'fallback' ? '云端生成失败；当前为本地证据回退报告。' : generation.status === 'cloud_verified' && reviewCompleted(synthesisReview) ? '已逐项核验问题并形成综合回答。' + reviewDescription(synthesisReview) + '。' : '回答已返回，每轮实际生成来源在消息旁标注。';
  }
  if (event.kind === 'error') showError(p.message || p.error || p.reason || '协作过程中发生错误。');
  appendActivity(event);
  if (event.kind === 'status') {
    if (session) session.state = p.state;
    running = p.state === 'running'; updateControls();
    if (!replay && ['idle', 'failed'].includes(p.state)) { closeStream(); reconcile(sessionId).catch(showError); }
  }
}
function closeStream() {
  if (stream) stream.close(); stream = null;
  if (retryTimer) clearTimeout(retryTimer); retryTimer = null;
}
function connectEvents() {
  closeStream(); const id = sessionId; if (!id) return;
  const source = new EventSource('/api/brain/sessions/' + encodeURIComponent(id) + '/events?after=' + lastSequence); stream = source;
  source.onmessage = event => {
    if (sessionId !== id || stream !== source) return;
    try { handleEvent(JSON.parse(event.data)); } catch (error) { showError('事件读取失败：' + error.message); }
  };
  source.onerror = () => {
    if (sessionId !== id || stream !== source || recovering || retryTimer) return;
    source.close();
    retryTimer = setTimeout(async () => {
      retryTimer = null; if (sessionId !== id) return;
      recovering = true;
      try { await reconcile(id, true); } catch (error) { running = false; updateControls(); showError('连接暂时中断，可点击当前历史对话恢复：' + error.message); }
      finally { recovering = false; }
    }, 900);
  };
}
function restoreSnapshot(value) {
  session = value; sessionId = value.id; lastSequence = -1; currentTurn = null; messageKeys.clear(); latestReport = null; latestGeneration = null;
  $('messages').textContent = ''; $('welcome').hidden = !!value.messages?.length; resetTrace();
  $('backend').value = value.backend || value.request?.backend || 'mock';
  const domains = value.domains || value.request?.domains || defaultAgents.map(agent => agent.id);
  document.querySelectorAll('input[name="domain"]').forEach(input => { input.checked = domains.includes(input.value); });
  $('taskEvidence').value = value.task_id || value.request?.task_id || '';
  restoreVrmsPath(value.vrms_path_id || value.request?.vrms_path_id || '');
  $('recordingSource').value = value.recording_id || '';
  $('conversationTitle').textContent = displayText(value.title || value.messages?.find(message => message.role === 'user')?.text?.slice(0, 32) || '新对话');
  for (const [index, message] of (value.messages || []).entries()) addMessage(message, 'snapshot:' + index);
  for (const event of (value.events || []).slice().sort((a, b) => a.sequence - b.sequence)) handleEvent(event, true);
  if (Number.isFinite(value.last_sequence)) lastSequence = Math.max(lastSequence, value.last_sequence);
  running = value.state === 'running'; updateControls(); scrollChat(true); storage('brainagent-session-id', sessionId);
}
async function reconcile(id, reconnect = false) {
  if (!id) return; const value = await request('/api/brain/sessions/' + encodeURIComponent(id));
  if (sessionId !== id) return; restoreSnapshot(value);
  await refreshHistory();
  if (sessionId === id && reconnect && running) connectEvents();
}
async function openSession(id, activate = true) {
  if ((running || submitting) && id !== sessionId) return;
  clearError(); closeStream(); const previous = sessionId; sessionId = id;
  try { const value = await request('/api/brain/sessions/' + encodeURIComponent(id)); if (sessionId !== id) return; restoreSnapshot(value); if (activate) switchPage('conversation'); await refreshHistory(); if (running) connectEvents(); }
  catch (error) { sessionId = previous; throw error; }
}
async function refreshHistory() {
  const response = await request('/api/brain/sessions'), sessions = Array.isArray(response) ? response : response.sessions || [];
  const root = $('history'); root.textContent = '';
  if (!sessions.length) { root.append(el('p', 'sidebar-empty', '暂无对话')); return; }
  for (const item of sessions.slice(0, 30)) {
    const button = el('button', 'history-item' + (item.id === sessionId ? ' active' : '')); button.dataset.id = item.id;
    button.append(el('span', 'history-text', item.title || item.first_message || item.messages?.find(message => message.role === 'user')?.text || item.id));
    if (item.backend === 'mock') button.append(el('span', 'history-mode', 'mock'));
    if (item.state === 'running') button.append(el('span', 'history-date', '协作中'));
    else if (item.updated_at || item.created_at || item.created) { const rawTime = item.updated_at || item.created_at || item.created; const date = new Date(typeof rawTime === 'number' ? rawTime * 1000 : rawTime); if (!Number.isNaN(date.getTime())) button.append(el('span', 'history-date', (date.getMonth() + 1) + '/' + date.getDate())); }
    button.onclick = () => openSession(item.id).catch(showError); root.append(button);
  }
  updateControls();
}
function newConversation(backend = 'api', preserveQuestion = false) {
  if (initializing || running || submitting) return;
  const question = preserveQuestion ? $('question').value : '';
  closeStream(); session = null; sessionId = null; lastSequence = -1; currentTurn = null; latestReport = null; latestGeneration = null; messageKeys.clear();
  $('recordingSource').value = $('vrmsPath').value = $('taskEvidence').value = '';
  $('backend').value = backend === 'mock' ? 'mock' : 'api';
  $('messages').textContent = ''; $('welcome').hidden = false; $('conversationTitle').textContent = '新对话'; $('question').value = question; $('question').style.height = '';
  storage('brainagent-session-id', null); resetTrace(); clearError(); switchPage('conversation'); updateControls(); refreshHistory().catch(showError); $('question').focus();
}
async function submitQuestion(event) {
  event.preventDefault(); const text = $('question').value.trim(); if (!text || initializing || submitting || running) return;
  const domains = questionDomains(text); if (!domains.length) { showError('请选择至少一个协作领域，或在问题中指定领域。'); return; }
  submitting = true; clearError(); updateControls();
  try {
    if (!sessionId) {
      const payload = {backend: $('backend').value, domains, pipeline_id: selectedPipeline(), recording_id: $('recordingSource').value || null, input_policy: 'raw_eeg_only'};
      const value = await post('/api/brain/sessions', payload); session = value; sessionId = value.id; storage('brainagent-session-id', sessionId);
      applyDomains(value.domains || domains);
    }
    await post('/api/brain/sessions/' + encodeURIComponent(sessionId) + '/messages', {text});
    $('question').value = ''; $('question').style.height = ''; running = true; session.state = 'running';
    $('workingText').textContent = 'Supervisor 正在组织协作…'; connectEvents(); await refreshHistory();
  } catch (error) { showError(error); if (sessionId) { try { await reconcile(sessionId, true); } catch (_) { running = false; } } }
  finally { submitting = false; updateControls(); }
}

function modelStatus(status) {
  if (status === 'raw_vrms_gpt_tool_agent') return {text: '原始 EEG → MIL 与八路工具 → GPT 工具 Agent 最终判断', unavailable: false};
  if (status === 'raw_vrms_model') return {text: 'VRMS 模型 · 原始波形重新编码与整路径分类', unavailable: false};
  if (status === 'tools_only') return {text: 'EEG 工具分析模式 · 无需分类模型', unavailable: false};
  if (!status) return {text: '模型部署状态待读取', unavailable: true};
  if (typeof status === 'object') {
    const base = modelStatus(status.status || (status.available ? 'available' : 'unavailable'));
    return {text: status.label || status.description || (status.reason ? base.text + ' · ' + status.reason : base.text), unavailable: status.available === false || base.unavailable};
  }
  const names = {available: '模型可用', deployed: '模型已部署', unavailable: '模型预测不可用', missing: '模型尚未部署', configured: '已配置模型'};
  return {text: names[status] || status, unavailable: /unavailable|untrained|not_deployed|missing|pending|未|不可用|待/.test(status)};
}
function renderCatalog(value) {
  catalog = value;
  const agents = value.agents || defaultAgents, sidebar = $('sidebarAgents'), cards = $('toolkitCards'); sidebar.textContent = ''; cards.textContent = '';
  for (const agent of agents) {
    const initial = agent.id === 'vrms' ? 'V' : agent.id === 'fatigue' ? 'F' : 'E';
    const row = el('div', 'sidebar-agent'); row.append(el('span', 'mini-avatar ' + agent.id, initial), el('span', '', agent.name), el('i', 'domain-dot ' + agent.id)); sidebar.append(row);
    const card = el('article', 'toolkit-card ' + agent.id), top = el('div', 'toolkit-top'), heading = el('div');
    const vrms = agent.id === 'vrms';
    heading.append(el('h2', '', agent.name), el('small', '', agent.label || domainNames[agent.id])); top.append(el('span', 'toolkit-avatar', initial), heading); card.append(top, el('p', 'toolkit-description', vrms ? '原始 EEG 重新编码，由 GPT 调用实际工具证据并给出整路径 High/Low 判断。' : agent.description || '使用独立领域工具与知识库，汇总可追溯的专业证据。'));
    const model = vrms ? {text: vrmsCatalog?.capabilities?.raw_model_assets_available ? '冻结 VRMSModel 已配置 · GPT 工具 Agent' : '冻结 VRMSModel 待配置 · 描述性 EEG 工具可用', unavailable: !vrmsCatalog?.capabilities?.raw_model_assets_available} : modelStatus(agent.model_status); card.append(el('div', 'model-status' + (model.unavailable ? ' unavailable' : ''), model.text), el('div', 'toolkit-list-heading', '可调用工具'));
    const toolbox = el('div', 'toolkit-tools');
    const registered = list(agent.tools).filter(tool => !vrms || !/EEGNet|EEGConformer/i.test(asText(tool)));
    for (const tool of registered) {
      const item = el('div', 'toolkit-tool'), detail = el('div'); detail.append(el('strong', '', tool.name || tool.label || tool.id || tool.tool), el('small', '', tool.description || tool.summary || tool.text || ''));
      item.append(icon('tool'), detail); toolbox.append(item);
    }
    if (!toolbox.childNodes.length) toolbox.append(el('p', 'muted', '工具目录尚未返回。'));
    card.append(toolbox, el('p', 'toolkit-footnote', vrms ? '问卷仅用于事后评价；高类路径时间区间按原始事件定位，判定在路径结束后形成。' : '原始 EEG 按路径与 mark22→下一次 mark20 的休息段直接比较指标；向 Supervisor 提交工具结果与专业方法引用。')); cards.append(card);
  }
  renderAgentNodes(); renderKnowledgeCatalog(value.knowledge || {});
  $('retrievalMethod').textContent = value.knowledge?.description || '当前使用本地词项检索，共享库与各领域知识库保留独立来源。';
}
function renderKnowledgeCatalog(knowledge) {
  const root = $('knowledgeCards'); root.textContent = '';
  const descriptions = {shared: 'EEG 基础、信号处理、质量控制与评价协议', vrms: 'VR 晕动症机制、测量解释与研究方法', fatigue: '疲劳相关 EEG 特征、任务与评价方法', emotion: '情绪维度、EEG 特征与模型评价'};
  for (const domain of ['shared', 'vrms', 'fatigue', 'emotion']) {
    const data = knowledge[domain] || knowledge.domains?.[domain] || list(knowledge.databases || knowledge.collections).find(item => item.domain === domain || item.id === domain);
    const card = el('article', 'knowledge-card ' + domain); card.append(el('span', 'micro-label', domain === 'shared' ? 'L1 共享基础' : 'L2 领域知识'), el('h2', '', domain === 'shared' ? domainNames[domain] : domainNames[domain] + '知识库'), el('p', '', data?.description || descriptions[domain]));
    const documents = data?.documents ?? data?.document_count ?? data?.sources ?? data?.count, chunks = data?.chunks ?? data?.chunk_count;
    const stats = [];
    if (typeof documents === 'number') stats.push(documents + ' 份资料'); if (typeof chunks === 'number') stats.push(chunks + ' 个片段');
    if (stats.length) card.append(el('span', 'knowledge-stat' + (documents === 0 ? ' empty' : ''), documents === 0 ? '暂无资料 · 可在下方录入' : stats.join(' · ')));
    else card.append(el('span', 'knowledge-stat', '资料数量待读取'));
    root.append(card);
  }
}
function renderStages(stages) {
  const root = $('knowledgeStages'); root.textContent = '';
  for (const [index, stage] of list(stages).entries()) {
    const stageNames = {shared_and_domain_retrieval: '共享库 + 领域库召回', candidate_rerank: '候选片段重排'};
    const text = stage.label || stage.name || stageNames[stage.stage] || stage.stage || stage.domain || stage.id || stage.text || '检索阶段 ' + (index + 1);
    const count = stage.count ?? stage.result_count ?? stage.candidates ?? stage.returned ?? (Array.isArray(stage.results) ? stage.results.length : undefined);
    root.append(el('span', 'search-stage', asText(text) + (count !== undefined ? ' · ' + count + ' 条' : '')));
  }
}
async function searchKnowledge(event) {
  event.preventDefault(); const query = $('knowledgeQuery').value.trim(); if (!query) return;
  $('knowledgeSearchButton').disabled = true; clearError();
  try {
    const params = new URLSearchParams({q: query, domains: selectedDomains('searchDomain').join(',')});
    const value = await request('/api/brain/knowledge/search?' + params.toString());
    renderStages(value.stages); renderCitations($('knowledgeResults'), list(value.results || value.matches));
  } catch (error) { showError(error); } finally { $('knowledgeSearchButton').disabled = false; }
}
async function importKnowledge(event) {
  event.preventDefault(); $('importButton').disabled = true; $('importFeedback').className = ''; $('importFeedback').textContent = '正在索引资料…';
  const payload = {domain: $('importDomain').value, title: $('importTitle').value.trim(), source: $('importSource').value.trim(), text: $('importText').value.trim()};
  if (!payload.title || !payload.source || !payload.text) { $('importFeedback').textContent = '请填写标题、来源和知识文本。'; $('importButton').disabled = false; return; }
  if ($('importDoi').value.trim()) payload.doi = $('importDoi').value.trim(); if ($('importPage').value.trim()) payload.page = $('importPage').value.trim();
  try {
    const result = await post('/api/brain/knowledge/documents', payload);
    $('importFeedback').textContent = '已将“' + payload.title + '”录入' + (payload.domain === 'shared' ? '共享知识库' : domainNames[payload.domain] + '知识库') + (typeof result.chunks === 'number' ? '，新增 ' + result.chunks + ' 个索引片段。' : '，可用于后续检索。');
    $('importText').value = ''; renderCatalog(await request('/api/brain/catalog'));
  } catch (error) { $('importFeedback').className = 'error'; $('importFeedback').textContent = '录入失败：' + error.message; }
  finally { $('importButton').disabled = false; }
}
async function loadTasks() { return []; }
function renderRecordings(items) {
  recordings = items;
  const root = $('recordingSource'), selected = session?.recording_id || root.value;
  root.replaceChildren(el('option', '', '在对话中指定被试或文件')); root.firstChild.value = '';
  for (const item of recordings) { const option = el('option', '', item.title || item.id); option.value = item.id; root.append(option); }
  root.value = selected;
}
async function uploadEdf(file) {
  if (!file || uploading || running) return;
  if (!/\.edf$/i.test(file.name) || file.size > 512 * 1024 * 1024) { showError('请上传不超过512MB的原始 EDF 文件。'); return; }
  uploading = true; clearError(); $('recordingFeedback').textContent = '正在上传并检查原始 EDF…'; updateControls();
  try {
    const response = await fetch('/api/brain/recordings/edf?filename=' + encodeURIComponent(file.name), {method: 'POST', headers: {'Content-Type': 'application/octet-stream'}, body: file});
    const item = await response.json(); if (!response.ok) throw new Error(item.detail || '上传失败');
    const directory = await request('/api/brain/recordings'); renderRecordings(directory.recordings || []);
    $('recordingSource').value = item.id;
    $('recordingFeedback').textContent = item.title + ' 已保存；请在下方对话中提出问题。';
    $('question').value = '请分析 ' + item.title + ' 的原始 EEG：完成多少条路径，VRMS 分类及首条高类路径时间是什么？分别比较首末完整路径、首末完整休息段的疲劳和心理负荷指标，给出各状态的数值、方向及时间。';
    $('question').focus();
  } catch (error) { showError(error); $('recordingFeedback').textContent = 'EDF 上传未完成，请检查文件后重试。'; }
  finally { uploading = false; $('edfFile').value = ''; updateControls(); }
}
function restoreVrmsPath(id) {
  $('vrmsPath').value = '';
  $('taskEvidence').value = '';
}
function renderVrmsCatalog(value) {
  vrmsCatalog=value;
  $('vrmsPipelineBadge').textContent='VRMSModel · GPT 工具 Agent';
  $('vrmsPipelineNote').textContent='从原始 EEG 重新计算工具证据，由 GPT 返回整路径 High/Low；目标标签仅用于事后评价。';
  $('vrmsBoundaryText').textContent=value.boundary || '';
  $('vrmsBoundaryPanel').hidden=false;
  const result=value.current_metrics || {};
  const metrics=result.metrics || {};
  const visible={vrmsmodel:metrics.vrmsmodel,gpt:metrics.gpt};
  $('vrmsHistoricalSummary').hidden=false;
  $('vrmsHistoricalSummary').textContent='seed2026 · 146 条原始 EEG 评分路径 · 24 名被试 · 已独立审计';
  $('vrmsHistoricalMetrics').replaceChildren(buildEvaluationTable(visible,'completed'));
  $('runVrmsEvaluation').title='实际重新处理原始 EEG 并请求 GPT；需要本地数据、冻结权重及接口';
  updateControls();
}

function evaluationActive() { return !!evaluation && !evaluation.terminal && !evaluationTerminal.has(evaluation.state || evaluation.status); }
function updateEvaluationControls() {
  const ready = vrmsCatalog?.capabilities?.cloud_ready ?? vrmsCatalog?.capabilities?.gpt_ready;
  $('runVrmsEvaluation').disabled = initializing || running || submitting || cloudSwitching || evaluationSubmitting || evaluationActive() || !!cloudSettings?.in_use || !apiConfigured || vrmsCatalog?.status !== 'ready' || ready === false || !vrmsCatalog?.capabilities?.raw_model_assets_available || cloudSettings?.model_id !== 'gpt-6.1-sol';
  const resumable = evaluation && ['failed', 'completed'].includes(evaluation.state) &&
    evaluation.seed === defaultSeed && evaluation.can_resume === true;
  $('resumeVrmsEvaluation').hidden = !resumable;
  $('resumeVrmsEvaluation').disabled = $('runVrmsEvaluation').disabled || evaluation?.model_id !== cloudSettings?.model_id;
  $('refreshVrmsEvaluations').disabled = initializing || evaluationSubmitting;
  $('vrmsEvaluationHistory').disabled = evaluationSubmitting || evaluationActive();
}
function closeEvaluationPoll() { if (evaluationTimer) clearTimeout(evaluationTimer); evaluationTimer = null; }
function buildEvaluationTable(metrics, phase, historical = false) {
  const wrap = el('div', 'evaluation-table-wrap'), table = el('table', 'evaluation-metrics-table' + (historical ? ' historical' : ''));
  table.append(el('caption', 'sr-only', 'seed2026 真实 GPT 与 VRMSModel ACC'));
  const head = el('thead'), heading = el('tr'), body = el('tbody');
  for (const title of ['判断来源', '判对 / 全部路径', 'ACC']) { const th = el('th', '', title); th.scope = 'col'; heading.append(th); }
  head.append(heading); table.append(head, body);
  for (const [key, name, detail] of [['gpt', '真实 GPT', '实际函数调用后返回的最终类别'], ['vrmsmodel', 'VRMSModel', '冻结编码器与整路径 MIL 输出']]) {
    const metric = metrics?.[key] || {}, row = el('tr'), label = el('td', '', name);
    label.append(el('small', '', detail));
    const valid = phase === 'completed' && typeof metric.accuracy === 'number' && Number.isFinite(metric.accuracy);
    row.append(label, el('td', '', (Number.isInteger(metric.correct) ? metric.correct : '—') + ' / 146'), el('td', '', valid ? (metric.accuracy * 100).toFixed(2) + '%' : historical ? '未提供' : phase === 'failed' ? '未完成' : '等待完整结果'));
    body.append(row);
  }
  wrap.append(table); return wrap;
}
function renderEvaluation(value) {
  const wasActive = evaluationActive();
  evaluation = value; evaluationId = value.id;
  const phaseNames = {queued: '等待启动', running: '运行中', completed: '已完成', failed: '失败', interrupted: '已中断', stopped: '已停止', cancelled: '已取消'}, phase = value.state || value.status;
  const total = value.total || 146, completed = value.completed_paths || 0;
  $('evaluationResultTitle').textContent = evaluationActive() ? '本次执行进度' : '已保存的网页评价';
  $('evaluationPhase').textContent = phaseNames[phase] || phase || '等待';
  $('vrmsEvaluationStatus').classList.toggle('failed', ['failed', 'interrupted'].includes(phase));
  $('vrmsEvaluationStatus').textContent = '已处理 ' + completed + '/' + total + ' 条路径 · 云端完成 ' + (value.cloud_completed || 0) + '，失败 ' + (value.cloud_failed || 0) + (value.error ? ' · ' + asText(value.error) : '');
  $('evaluationRecordMeta').textContent = 'VRMS 模型 · ' + (value.model_id || '云端模型待读取') + '\n记录 ' + value.id + (value.resume_from ? '\n续跑来源 ' + value.resume_from : '');
  $('vrmsEvaluationProgress').hidden = false; $('vrmsEvaluationProgress').max = total; $('vrmsEvaluationProgress').value = Math.max(0, Math.min(total, completed));
  $('vrmsEvaluationMetrics').replaceChildren(buildEvaluationTable(value.metrics, phase));
  const cloud = value.metrics?.gpt;
  if (cloud && phase === 'completed') {
    const coverage = typeof cloud.coverage === 'number' && Number.isFinite(cloud.coverage) ? (cloud.coverage * 100).toFixed(2) + '%' : '—';
    $('vrmsEvaluationMetrics').append(el('p', 'evaluation-note', '云端有效输出覆盖率 ' + coverage + '；上表 ACC 使用全部 ' + total + ' 条路径作分母，失败路径不算判对。'));
  }
  if (phase === 'completed') { const link = el('a', 'evaluation-report-link'); link.append(icon('download'), document.createTextNode('下载此评价报告')); link.href = '/api/brain/vrms/evaluations/' + encodeURIComponent(value.id) + '/report'; link.download = 'vrms-evaluation-' + value.id + '.json'; $('vrmsEvaluationMetrics').append(link); }
  updateControls();
  if ((wasActive || cloudSettings?.in_use) && evaluationTerminal.has(phase)) refreshCloudConfig().catch(showError);
}
async function openEvaluation(id) {
  closeEvaluationPoll(); evaluationId = id; if (!id) return;
  const value = await request('/api/brain/vrms/evaluations/' + encodeURIComponent(id));
  if (evaluationId !== id) return;
  renderEvaluation(value);
  if (evaluationActive()) evaluationTimer = setTimeout(() => pollEvaluation(id), 3000);
}
async function pollEvaluation(id) {
  evaluationTimer = null; if (evaluationId !== id) return;
  try { await openEvaluation(id); }
  catch (error) {
    if (evaluationId !== id) return;
    $('vrmsEvaluationStatus').textContent = '进度读取暂时失败，将继续尝试：' + error.message;
    evaluationTimer = setTimeout(() => pollEvaluation(id), 5000);
  }
}
async function refreshVrmsEvaluations() {
  const seed = selectedSeed(), value = await request('/api/brain/vrms/evaluations');
  if (seed !== selectedSeed()) return;
  const all = Array.isArray(value) ? value : value.evaluations || [];
  const active = all.find(item => !item.terminal && !evaluationTerminal.has(item.state || item.status));
  const matching = all.filter(item => item.id !== active?.id);
  const items = active ? [active, ...matching] : matching;
  const select = $('vrmsEvaluationHistory'); select.replaceChildren(el('option', '', items.length ? '选择最近评价' : '暂无评价任务')); select.firstChild.value = '';
  select.firstChild.disabled = !!items.length;
  for (const item of items.slice(0, 20)) { const option = el('option', '', (item.model_id || '云端模型') + ' · ' + (item.state || item.status) + ' · ' + item.id.slice(0, 8)); option.value = item.id; select.append(option); }
  const chosen = active || items.find(item => item.id === evaluationId) || items.find(item => Number(item.seed) === defaultSeed) || items[0];
  if (chosen) { select.value = chosen.id; await openEvaluation(chosen.id); }
  else { closeEvaluationPoll(); evaluation = null; evaluationId = null; $('evaluationResultTitle').textContent = '网页评价记录'; $('evaluationPhase').textContent = '暂无记录'; $('evaluationRecordMeta').textContent = ''; $('vrmsEvaluationStatus').classList.remove('failed'); $('vrmsEvaluationStatus').textContent = '尚无网页评价，可运行新的评价。'; $('vrmsEvaluationProgress').hidden = true; $('vrmsEvaluationMetrics').textContent = ''; updateControls(); }
}
async function runVrmsEvaluation(resume = false) {
  if ($(resume ? 'resumeVrmsEvaluation' : 'runVrmsEvaluation').disabled || evaluationSubmitting || evaluationActive() || cloudSettings?.in_use) return;
  evaluationSubmitting = true; clearError(); updateControls();
  try {
    const payload = {seed: resume ? Number(evaluation.seed) : defaultSeed};
    if (resume) payload.resume_from = evaluation.id;
    const value = await post('/api/brain/vrms/evaluations', payload);
    switchPage('evaluation'); renderEvaluation(value); await refreshVrmsEvaluations();
  } catch (error) { $('vrmsEvaluationStatus').textContent = '评价未启动：' + error.message; showError(error); }
  finally { evaluationSubmitting = false; updateControls(); }
}
function downloadReport() {
  if (!latestReport) return; const blob = new Blob([displayReport(latestReport)], {type: 'text/markdown;charset=utf-8'}), url = URL.createObjectURL(blob), link = el('a');
  link.href = url; link.download = (sessionId || 'brainagent') + '.md'; document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}

$('questionForm').onsubmit = submitQuestion;
$('recordingSource').onchange = updateControls;
$('uploadEdfButton').onclick = () => $('edfFile').click();
$('edfFile').onchange = () => uploadEdf($('edfFile').files?.[0]);
$('question').addEventListener('input', () => { $('question').style.height = 'auto'; $('question').style.height = Math.min(160, $('question').scrollHeight) + 'px'; updateControls(); });
$('question').addEventListener('keydown', event => { if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); if (!$('sendQuestion').disabled) $('questionForm').requestSubmit(); } });
$('newConversation').onclick = () => newConversation();
$('newRealConversation').onclick = () => newConversation('api', true);
$('refreshCloudConfig').onclick = () => refreshCloudConfig().catch(showError);
$('cloudProvider').onchange = () => switchCloudProvider().catch(showError);
$('refreshHistory').onclick = () => refreshHistory().catch(showError);
$('downloadReport').onclick = downloadReport;
$('backend').onchange = () => { if (sessionId) newConversation($('backend').value, true); else updateControls(); };
$('taskEvidence').onchange = updateControls; $('vrmsPath').onchange = updateControls;
$('runVrmsEvaluation').onclick = () => runVrmsEvaluation().catch(showError);
$('resumeVrmsEvaluation').onclick = () => runVrmsEvaluation(true).catch(showError);
$('refreshVrmsEvaluations').onclick = () => refreshVrmsEvaluations().catch(showError);
$('vrmsEvaluationHistory').onchange = () => openEvaluation($('vrmsEvaluationHistory').value).catch(showError);
$('scopeButton').onclick = () => { const opened = $('scopePanel').hidden; $('scopePanel').hidden = !opened; $('scopeButton').setAttribute('aria-expanded', String(opened)); };
document.querySelectorAll('input[name="domain"]').forEach(input => input.onchange = () => { renderAgentNodes(); updateControls(); });
document.querySelectorAll('button.nav-item').forEach(button => button.onclick = () => switchPage(button.dataset.page));
document.querySelectorAll('.inspector-tab').forEach(button => button.onclick = () => switchInspector(button.dataset.tab));
document.querySelectorAll('.suggestion').forEach(button => button.onclick = () => { $('question').value = button.dataset.prompt; $('question').dispatchEvent(new Event('input')); $('question').focus(); });
$('traceToggle').onclick = () => { if ($('appShell').classList.contains('inspector-open')) closePanels(); else openPanel('inspector'); };
$('menuButton').onclick = () => { if ($('appShell').classList.contains('sidebar-open')) closePanels(); else openPanel('sidebar'); };
$('closeInspector').onclick = () => closePanels(); $('closeMenu').onclick = () => closePanels(); $('panelBackdrop').onclick = () => closePanels();
$('knowledgeSearchForm').onsubmit = searchKnowledge; $('knowledgeImportForm').onsubmit = importKnowledge;
document.addEventListener('click', event => {
  if (!$('scopePanel').hidden && !$('scopePanel').contains(event.target) && !$('scopeButton').contains(event.target)) { $('scopePanel').hidden = true; $('scopeButton').setAttribute('aria-expanded', 'false'); }
  if ($('appShell').classList.contains('sidebar-open') && !event.target.closest('.sidebar') && !event.target.closest('#menuButton')) closePanels(false);
});
document.addEventListener('keydown', event => {
  if (event.key === 'Escape') { $('scopePanel').hidden = true; $('scopeButton').setAttribute('aria-expanded', 'false'); closePanels(); }
  if (event.key === 'Tab') {
    const overlay = $('appShell').classList.contains('sidebar-open') && window.matchMedia('(max-width: 1099px)').matches ? document.querySelector('.sidebar') : $('appShell').classList.contains('inspector-open') && window.matchMedia('(max-width: 1279px)').matches ? $('inspector') : null;
    if (overlay) {
      const items = Array.from(overlay.querySelectorAll('button:not(:disabled), a[href], summary, input:not(:disabled), select:not(:disabled), [tabindex="0"]')).filter(node => node.getClientRects().length && node.tabIndex !== -1);
      const first = items[0], last = items[items.length - 1];
      if (items.length && (event.shiftKey ? document.activeElement === first : document.activeElement === last)) { event.preventDefault(); (event.shiftKey ? last : first).focus(); }
    }
  }
});
document.querySelectorAll('.inspector-tab').forEach(button => button.addEventListener('keydown', event => {
  const names = ['activity', 'tools', 'citations'], index = names.indexOf(button.dataset.tab);
  const next = event.key === 'ArrowRight' ? (index + 1) % 3 : event.key === 'ArrowLeft' ? (index + 2) % 3 : event.key === 'Home' ? 0 : event.key === 'End' ? 2 : -1;
  if (next >= 0) { event.preventDefault(); switchInspector(names[next]); document.querySelector('.inspector-tab[data-tab="' + names[next] + '"]').focus(); }
}));
window.addEventListener('hashchange', () => switchPage(pageFromHash(), false));
window.addEventListener('popstate', () => switchPage(pageFromHash(), false));
document.querySelector('.skip-link').onclick = event => { event.preventDefault(); switchPage('conversation'); $('question').focus(); };
window.addEventListener('beforeunload', () => { closeStream(); closeEvaluationPoll(); });

(async function initialize() {
  switchPage(initialPage, false); resetTrace(); updateControls();
  const results = await Promise.allSettled([request('/api/brain/catalog'), refreshHistory(), loadTasks(), request('/api/cloud-settings'), request(vrmsCatalogUrl()), request('/api/brain/recordings')]);
  if (results[0].status === 'fulfilled') renderCatalog(results[0].value); else showError('协作目录读取失败：' + results[0].reason.message);
  if (results[1].status === 'rejected') showError('历史对话读取失败：' + results[1].reason.message);
  if (results[2].status === 'rejected') $('taskEvidence').title = 'EEG 任务目录暂时不可用：' + results[2].reason.message;
  if (results[3].status === 'fulfilled') {
    renderCloudSettings(results[3].value);
  }
  else { $('cloudModelSummary').textContent = 'API 型号读取失败'; $('cloudConnectionSummary').textContent = '接口配置状态未知'; }
  if (results[4].status === 'fulfilled') renderVrmsCatalog(results[4].value);
  else renderVrmsCatalog({status: 'unavailable', reason: 'VRMS 路径目录读取失败：' + results[4].reason.message});
  if (results[5].status === 'fulfilled') renderRecordings(results[5].value.recordings || []);
  else $('recordingFeedback').textContent = '原始文件目录读取失败；请刷新后重试。';
  const id = storage('brainagent-session-id');
  if (id) { try { await openSession(id, false); } catch (error) { storage('brainagent-session-id', null); sessionId = null; session = null; showError('历史对话无法恢复：' + error.message); } }
  initializing = false; switchPage(pageFromHash(), false); switchInspector('activity'); updateControls();
  refreshVrmsEvaluations().catch(error => { $('vrmsEvaluationStatus').textContent = '评价目录读取失败：' + error.message; });
})();
