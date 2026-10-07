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
const toastEl = document.getElementById("toast");

let busy = false;
let ready = false;
let uploading = false;
let embeddingsReady = false;
let llmError = "";

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
  uploadLabel.textContent = label || (on ? "업로드 중…" : "PDF 업로드");
  fileInput.disabled = on;
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
    li.innerHTML = `<div class="name">${file.name}</div><div class="meta">청크 ${file.chunks}개</div>`;
    fileList.appendChild(li);
  });
}

function applyStatus(data) {
  ready = Boolean(data.ready);
  embeddingsReady = Boolean(data.embeddings_ready);
  llmError = data.error || "";
  renderFiles(data.files || []);
  updateSendButton();

  if (data.error && !ready) {
    readyBadge.className = "badge error";
    readyBadge.textContent = "LLM 오류";
    if (!uploading) {
      statusText.textContent = data.error
        || (embeddingsReady
          ? "문서 업로드는 가능합니다. 답변용 LLM은 아직 준비되지 않았습니다."
          : "모델을 준비하지 못했습니다.");
    }
    return ready;
  }
  if (ready) {
    readyBadge.className = "badge ready";
    readyBadge.textContent = "준비됨";
    if (!uploading) statusText.textContent = "로컬 모델이 준비되었습니다.";
  } else if (data.embeddings_ready) {
    readyBadge.className = "badge waiting";
    readyBadge.textContent = "LLM 로딩";
    if (!uploading) statusText.textContent = "임베딩은 준비됨. LLM을 불러오는 중입니다.";
  } else {
    readyBadge.className = "badge waiting";
    readyBadge.textContent = "준비 중";
    if (!uploading) statusText.textContent = "모델을 준비하는 중입니다…";
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
  const wrap = document.createElement("div");
  wrap.className = "sources";
  wrap.innerHTML = `<div class="sources-title">참고 문서</div>`;
  sources.forEach((src) => {
    const card = document.createElement("div");
    card.className = "source-card";
    card.innerHTML = `
      <div class="source-meta">${src.filename} · ${src.page}페이지</div>
      <div class="source-excerpt">${src.excerpt || ""}</div>
    `;
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
  if (uploading) {
    toast("이미 업로드를 처리 중입니다.");
    return;
  }

  const formData = new FormData();
  files.forEach((file) => formData.append("files", file, file.name));
  setUploadBusy(true, "임베딩 중…");
  statusText.textContent = `${files.length}개 문서를 읽고 임베딩하는 중입니다. 완료까지 시간이 걸릴 수 있습니다.`;

  try {
    const res = await fetch("/api/upload", { method: "POST", body: formData });
    const data = await res.json().catch(() => ({}));
    if (!res.ok || !data.ok) {
      toast(data.error || "업로드에 실패했습니다.");
      statusText.textContent = data.error || "업로드에 실패했습니다.";
    } else {
      const added = (data.added || []).length;
      toast(`${added}개 파일을 저장했습니다.`);
      renderFiles(data.files || []);
      statusText.textContent = `${added}개 파일을 벡터 DB에 저장했습니다.`;
      const skipped = [...(data.skipped || []), ...(data.rejected || [])];
      if (skipped.length) {
        toast(skipped.map((s) => `${s.name}: ${s.reason}`).join(" / "));
      }
    }
  } catch (err) {
    toast("서버와 연결하지 못했습니다. 앱이 실행 중인지 확인하세요.");
    statusText.textContent = "업로드 중 서버 연결에 실패했습니다.";
  } finally {
    fileInput.value = "";
    setUploadBusy(false);
    refreshStatus();
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

  busy = true;
  updateSendButton();
  questionEl.value = "";
  questionEl.style.height = "auto";
  appendMessage("user", message);
  const assistant = appendMessage("assistant", "답변을 생성하는 중입니다…");
  let started = false;
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
          if (!payload.no_info) renderSources(assistant.bubble, sources);
        } else if (payload.type === "error") {
          assistant.content.textContent = payload.error || "오류가 발생했습니다.";
        }
      }
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
  messagesEl.innerHTML = `
    <div class="welcome" id="welcome">
      <h3>PDF를 업로드한 뒤 질문해 보세요</h3>
      <p>왼쪽에서 문서를 올리면, 해당 내용과 페이지를 근거로 답변합니다.</p>
    </div>
  `;
  toast("새 대화를 시작합니다.");
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
  if (!ready && !llmError) setTimeout(pollUntilReady, 2500);
}

pollUntilReady();
