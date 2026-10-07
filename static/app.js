const fileInput = document.getElementById("file-input");
const uploadArea = document.getElementById("upload-area");
const uploadLabel = document.getElementById("upload-label");
const fileList = document.getElementById("file-list");
const messagesEl = document.getElementById("messages");
const form = document.getElementById("chat-form");
const questionEl = document.getElementById("question");
const sendBtn = document.getElementById("send-btn");
const statusText = document.getElementById("status-text");
const readyBadge = document.getElementById("ready-badge");
const newChatBtn = document.getElementById("new-chat");
const retryBtn = document.getElementById("retry-models");
const toastEl = document.getElementById("toast");

let busy = false;
let ready = false;
let uploading = false;
let embeddingsReady = false;
let llmError = "";
let embeddingError = "";
let currentJobId = null;
let jobPollTimer = null;

function updateSendButton() {
  sendBtn.disabled = busy || !questionEl.value.trim();
}

updateSendButton();

function toast(message) {
  toastEl.textContent = message;
  toastEl.classList.remove("hidden");
  setTimeout(() => toastEl.classList.add("hidden"), 4000);
}

function setUploadBusy(on, label) {
  uploading = on;
  uploadArea.classList.toggle("uploading", on);
  uploadLabel.textContent = label || (on ? "처리 중…" : "PDF 업로드");
  fileInput.disabled = on;
}

function setRetryVisible(on) {
  retryBtn.classList.toggle("hidden", !on);
  retryBtn.disabled = on ? false : retryBtn.disabled;
}

function renderFiles(files) {
  fileList.innerHTML = "";
  if (!files || files.length === 0) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "아직 업로드된 파일이 없습니다.";
    fileList.appendChild(li);
    return;
  }
  files.forEach((file) => {
    const li = document.createElement("li");
    const name = document.createElement("div");
    name.className = "name";
    name.textContent = file.name;
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = `청크 ${file.chunks}개`;
    li.appendChild(name);
    li.appendChild(meta);
    fileList.appendChild(li);
  });
}

function applyStatus(data) {
  ready = Boolean(data.ready);
  embeddingsReady = Boolean(data.embeddings_ready);
  llmError = data.llm_error || (!data.llm_ready ? data.error : "") || "";
  embeddingError = data.embedding_error || "";
  renderFiles(data.files || []);
  updateSendButton();
  setRetryVisible(!ready);

  if (data.upload_job && (data.upload_job.status === "queued" || data.upload_job.status === "running")) {
    if (!currentJobId) {
      currentJobId = data.upload_job.id;
      pollJob(currentJobId);
    }
  }

  if (uploading) return ready;

  if (embeddingError && !embeddingsReady) {
    readyBadge.className = "badge error";
    readyBadge.textContent = "임베딩 오류";
    statusText.textContent = embeddingError;
    return ready;
  }
  if ((llmError || data.error) && !ready) {
    readyBadge.className = "badge error";
    readyBadge.textContent = "LLM 오류";
    statusText.textContent = llmError || data.error;
    return ready;
  }
  if (ready) {
    readyBadge.className = "badge ready";
    readyBadge.textContent = "준비됨";
    statusText.textContent = "로컬 모델이 준비되었습니다.";
  } else if (data.initializing && data.embeddings_ready) {
    readyBadge.className = "badge waiting";
    readyBadge.textContent = "LLM 로딩";
    statusText.textContent = "임베딩은 준비됨. LLM을 불러오는 중입니다.";
  } else if (data.embeddings_ready) {
    readyBadge.className = "badge waiting";
    readyBadge.textContent = "LLM 대기";
    statusText.textContent = "임베딩은 준비됨. 답변 모델이 아직 준비되지 않았습니다.";
  } else {
    readyBadge.className = "badge waiting";
    readyBadge.textContent = data.initializing ? "준비 중" : "대기";
    statusText.textContent = "모델을 준비하는 중입니다…";
  }
  return ready;
}

function appendMessage(role, text) {
  document.getElementById("welcome")?.remove();
  const row = document.createElement("div");
  row.className = `msg ${role}`;
  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "나" : "AI";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  const content = document.createElement("div");
  content.className = "content";
  content.textContent = text;
  bubble.appendChild(content);
  row.appendChild(avatar);
  row.appendChild(bubble);
  messagesEl.appendChild(row);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return { row, content, bubble };
}

function renderSources(bubble, sources) {
  if (!sources || sources.length === 0) return;
  if (bubble.querySelector(".sources")) return;
  const wrap = document.createElement("div");
  wrap.className = "sources";
  const title = document.createElement("div");
  title.className = "sources-title";
  title.textContent = "참고 문서";
  wrap.appendChild(title);
  sources.forEach((src) => {
    const card = document.createElement("div");
    card.className = "source-card";
    const meta = document.createElement("div");
    meta.className = "source-meta";
    meta.textContent = `${src.filename} · ${src.page}페이지`;
    const excerpt = document.createElement("div");
    excerpt.className = "source-excerpt";
    excerpt.textContent = src.excerpt || "";
    card.appendChild(meta);
    card.appendChild(excerpt);
    wrap.appendChild(card);
  });
  bubble.appendChild(wrap);
}

async function refreshStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    return applyStatus(data);
  } catch (err) {
    if (!uploading) statusText.textContent = "상태 확인에 실패했습니다.";
    return false;
  }
}

function finishUpload(ok, message) {
  if (jobPollTimer) {
    clearTimeout(jobPollTimer);
    jobPollTimer = null;
  }
  currentJobId = null;
  fileInput.value = "";
  setUploadBusy(false);
  if (!ok && message) statusText.textContent = message;
  refreshStatus();
}

function applyJobView(job) {
  if (!job) return;
  if (job.status === "queued") {
    setUploadBusy(true, "대기 중…");
    statusText.textContent = "문서 처리를 기다리는 중입니다.";
  } else if (job.status === "running") {
    const label = job.stage_label || "처리";
    setUploadBusy(true, `${label} 중…`);
    statusText.textContent = `${label} 중입니다. 완료까지 시간이 걸릴 수 있습니다.`;
  }
}

async function pollJob(jobId) {
  try {
    const res = await fetch(`/api/jobs/${jobId}`);
    const data = await res.json().catch(() => ({}));
    if (res.status === 404 || data.missing) {
      toast(data.error || "작업 상태를 찾을 수 없습니다.");
      finishUpload(false, data.error || "서버가 재시작되어 작업 상태를 잃었습니다.");
      return;
    }
    if (!res.ok) {
      statusText.textContent = "작업 상태 조회에 실패했습니다. 처리 자체는 계속될 수 있습니다.";
      jobPollTimer = setTimeout(() => pollJob(jobId), 2000);
      return;
    }
    applyJobView(data);
    if (data.status === "queued" || data.status === "running") {
      jobPollTimer = setTimeout(() => pollJob(jobId), 1500);
      return;
    }
    if (data.status === "done") {
      const added = (data.added || []).length;
      toast(`${added}개 파일을 저장했습니다.`);
      renderFiles(data.files || []);
      const skipped = [...(data.skipped || []), ...(data.rejected || [])];
      if (skipped.length) {
        toast(skipped.map((item) => `${item.name}: ${item.reason}`).join(" / "));
      }
      const warnings = data.warnings || [];
      if (warnings.length) {
        toast(warnings.map((item) => `${item.name}: ${item.reason}`).join(" / "));
      }
      finishUpload(true);
      return;
    }
    toast(data.error || "업로드에 실패했습니다.");
    const skipped = [...(data.skipped || []), ...(data.rejected || [])];
    if (skipped.length) {
      toast(skipped.map((item) => `${item.name}: ${item.reason}`).join(" / "));
    }
    finishUpload(false, data.error || "업로드에 실패했습니다.");
  } catch (err) {
    statusText.textContent = "작업 상태 조회에 실패했습니다. 처리 자체는 계속될 수 있습니다.";
    jobPollTimer = setTimeout(() => pollJob(jobId), 2000);
  }
}

async function uploadFiles(fileListLike) {
  const files = Array.from(fileListLike || []).filter(Boolean);
  if (!files.length) {
    toast("PDF 파일을 선택하세요.");
    return;
  }
  const notPdf = files.filter((file) => !file.name.toLowerCase().endsWith(".pdf"));
  if (notPdf.length) {
    toast("PDF 파일만 업로드할 수 있습니다.");
    return;
  }
  if (uploading || currentJobId) {
    toast("이미 업로드를 처리 중입니다.");
    return;
  }

  const formData = new FormData();
  files.forEach((file) => formData.append("files", file, file.name));
  setUploadBusy(true, "업로드 중…");
  statusText.textContent = `${files.length}개 문서 처리를 등록하는 중입니다.`;

  try {
    const res = await fetch("/api/upload", { method: "POST", body: formData });
    const data = await res.json().catch(() => ({}));
    if (!res.ok || !data.ok || !data.job_id) {
      toast(data.error || "업로드에 실패했습니다.");
      finishUpload(false, data.error || "업로드에 실패했습니다.");
      return;
    }
    currentJobId = data.job_id;
    applyJobView(data);
    pollJob(currentJobId);
  } catch (err) {
    toast("서버와 연결하지 못했습니다. 앱이 실행 중인지 확인하세요.");
    finishUpload(false, "업로드 중 서버 연결에 실패했습니다.");
  }
}

fileInput.addEventListener("change", () => {
  uploadFiles(fileInput.files);
});

["dragenter", "dragover"].forEach((eventName) => {
  uploadArea.addEventListener(eventName, (event) => {
    event.preventDefault();
    uploadArea.classList.add("uploading");
  });
});
["dragleave", "drop"].forEach((eventName) => {
  uploadArea.addEventListener(eventName, (event) => {
    event.preventDefault();
    if (!uploading) uploadArea.classList.remove("uploading");
  });
});
uploadArea.addEventListener("drop", (event) => {
  const dropped = event.dataTransfer?.files;
  if (dropped?.length) uploadFiles(dropped);
});

async function sendChat() {
  if (busy) return;
  const message = questionEl.value.trim();
  if (!message) return;
  if (!ready) {
    toast(llmError || "모델이 아직 준비되지 않았습니다. 잠시 후 다시 시도하세요.");
    return;
  }
  if (uploading || currentJobId) {
    toast("업로드 중인 문서는 아직 검색에 포함되지 않습니다. 기존 문서에 대해서만 질문합니다.");
  }

  busy = true;
  updateSendButton();
  questionEl.value = "";
  questionEl.style.height = "auto";
  appendMessage("user", message);
  const assistant = appendMessage("assistant", "답변을 생성하는 중입니다…");
  let started = false;
  let finished = false;
  let sawError = false;
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      assistant.content.textContent = data.error || "답변을 생성하지 못했습니다.";
      return;
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let sources = [];
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop();
      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith("data:")) continue;
        const payload = JSON.parse(line.slice(5).trim());
        if (payload.type === "meta") {
          sources = payload.sources || [];
        } else if (payload.type === "token") {
          if (!started) {
            assistant.content.textContent = "";
            started = true;
          }
          assistant.content.textContent += payload.text || "";
          messagesEl.scrollTop = messagesEl.scrollHeight;
        } else if (payload.type === "done") {
          finished = true;
          if (payload.answer) assistant.content.textContent = payload.answer;
          if (!payload.no_info) renderSources(assistant.bubble, sources);
        } else if (payload.type === "error") {
          sawError = true;
          assistant.content.textContent = payload.error || "오류가 발생했습니다.";
        }
      }
    }
    if (!finished && !sawError) {
      assistant.content.textContent = "응답이 완료되지 않았습니다. 다시 시도하세요.";
    }
  } catch (err) {
    assistant.content.textContent = "서버 연결에 실패했습니다.";
  } finally {
    busy = false;
    updateSendButton();
    questionEl.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  sendChat();
});

newChatBtn.addEventListener("click", async () => {
  await fetch("/api/reset", { method: "POST" });
  messagesEl.innerHTML = "";
  const welcome = document.createElement("div");
  welcome.className = "welcome";
  welcome.id = "welcome";
  const heading = document.createElement("h3");
  heading.textContent = "PDF를 업로드한 뒤 질문해 보세요";
  const desc = document.createElement("p");
  desc.textContent = "왼쪽에서 문서를 올리면, 해당 내용과 페이지를 근거로 답변합니다.";
  welcome.appendChild(heading);
  welcome.appendChild(desc);
  messagesEl.appendChild(welcome);
  toast("새 대화를 시작합니다.");
});

retryBtn.addEventListener("click", async () => {
  retryBtn.disabled = true;
  statusText.textContent = "모델을 다시 불러오는 중입니다…";
  try {
    const res = await fetch("/api/retry-models", { method: "POST" });
    const data = await res.json().catch(() => ({}));
    applyStatus(data);
    if (!data.ready) {
      toast(data.llm_error || data.error || "모델이 아직 준비되지 않았습니다.");
    }
  } catch (err) {
    toast("모델 재시도 요청에 실패했습니다.");
  } finally {
    retryBtn.disabled = false;
  }
});

questionEl.addEventListener("keydown", (event) => {
  if (event.key !== "Enter" || event.shiftKey) return;
  if (event.isComposing || event.keyCode === 229) return;
  event.preventDefault();
  sendChat();
});

questionEl.addEventListener("input", () => {
  questionEl.style.height = "auto";
  questionEl.style.height = `${Math.min(questionEl.scrollHeight, 160)}px`;
  updateSendButton();
});

async function pollUntilReady() {
  await refreshStatus();
  if (!ready) setTimeout(pollUntilReady, llmError || embeddingError ? 8000 : 2500);
}

pollUntilReady();
