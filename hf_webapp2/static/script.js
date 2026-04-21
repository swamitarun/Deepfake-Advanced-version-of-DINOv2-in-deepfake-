/**
 * DeepShield AI — script.js (Full-Stack HF Space version)
 * API_URL = "" means: same server, same domain. No CORS needed!
 */
const API_URL = "";  // Empty = same HF Space serves both UI and API

// ── Server Status Ping ──
async function checkServerStatus() {
  const statusMenu = document.getElementById("server-status");
  const statusText = document.getElementById("status-text");
  const dropZone = document.getElementById("drop-zone");

  if (!statusMenu) return;

  try {
    const res = await fetch(`${API_URL}/health`);
    if (!res.ok) throw new Error("Server not OK");
    const data = await res.json();

    statusMenu.className = "server-status";

    if (data.model_loaded === true) {
      statusMenu.classList.add("status-connected");
      statusText.textContent = "AI Ready ✓";

      dropZone.style.pointerEvents = "auto";
      dropZone.style.opacity = "1";
      document.querySelector(".drop-title").innerHTML = "Drop your video or photo here";
      document.querySelector(".drop-sub").innerHTML = 'or <span class="link-text">browse files</span>';
    } else {
      statusMenu.classList.add("status-error");
      statusText.textContent = "Model Missing";

      dropZone.style.pointerEvents = "none";
      dropZone.style.opacity = "0.5";
      document.querySelector(".drop-title").innerHTML = "⚠️ Model Not Uploaded";
      document.querySelector(".drop-sub").textContent = "Admin: Upload best_model.pth to this HF Space.";
    }
  } catch (err) {
    statusMenu.className = "server-status status-error";
    statusText.textContent = "Server Waking Up...";

    dropZone.style.pointerEvents = "none";
    dropZone.style.opacity = "0.5";
    document.querySelector(".drop-title").innerHTML = "⚠️ Server is starting...";
    document.querySelector(".drop-sub").textContent = "Takes ~60 sec. Page will auto-refresh status.";
  }
}

// Check on load, then every 10 seconds (also keeps server alive!)
checkServerStatus();
setInterval(checkServerStatus, 10000);

const MAX_FILE_MB = 30;
const MAX_FILE_BYTES = MAX_FILE_MB * 1024 * 1024;

let currentFile = null;
let lastResult = null;

function showSection(id) {
  const sections = ["upload-section", "loading-section", "results-section", "error-section"];
  sections.forEach(s => {
    document.getElementById(s).classList.toggle("hidden", s !== id);
  });
}

function onDragOver(e) {
  e.preventDefault();
  document.getElementById("drop-zone").classList.add("dragging");
}

function onDragLeave() {
  document.getElementById("drop-zone").classList.remove("dragging");
}

function onDrop(e) {
  e.preventDefault();
  document.getElementById("drop-zone").classList.remove("dragging");
  const file = e.dataTransfer?.files?.[0];
  if (file) processFile(file);
}

function onFileSelected(e) {
  const file = e.target.files?.[0];
  if (file) processFile(file);
}

function processFile(file) {
  const allowedExt = [".mp4", ".mov", ".avi", ".mkv", ".jpg", ".jpeg", ".png", ".webp"];
  const ext = "." + file.name.split(".").pop().toLowerCase();
  if (!allowedExt.includes(ext)) {
    showError(`❌ Unsupported file type: "${ext}". Please upload video (MP4/MOV/AVI/MKV) or image (JPG/PNG/WEBP).`);
    return;
  }

  if (file.size > MAX_FILE_BYTES) {
    const sizeMB = (file.size / 1024 / 1024).toFixed(1);
    showError(`❌ File too large (${sizeMB} MB). Maximum allowed: ${MAX_FILE_MB} MB.`);
    return;
  }

  currentFile = file;
  document.getElementById("file-name").textContent = file.name;
  document.getElementById("file-size").textContent = formatBytes(file.size);

  const url = URL.createObjectURL(file);
  const isVideo = [".mp4", ".mov", ".avi", ".mkv"].includes(ext);
  
  if (isVideo) {
    document.getElementById("video-preview").src = url;
    document.getElementById("video-preview").style.display = "block";
    const imgPreview = document.getElementById("image-preview");
    if(imgPreview) imgPreview.style.display = "none";
  } else {
    document.getElementById("video-preview").style.display = "none";
    const imgPreview = document.getElementById("image-preview");
    if(imgPreview) {
        imgPreview.src = url;
        imgPreview.style.display = "block";
    }
  }

  document.getElementById("drop-zone").classList.add("hidden");
  document.getElementById("file-preview").classList.remove("hidden");
}

function resetUpload() {
  currentFile = null;
  lastResult = null;

  document.getElementById("file-input").value = "";

  const video = document.getElementById("video-preview");
  if (video.src) URL.revokeObjectURL(video.src);
  video.src = "";

  const img = document.getElementById("image-preview");
  if (img && img.src) URL.revokeObjectURL(img.src);
  if (img) img.src = "";

  document.getElementById("ring-fill").style.strokeDashoffset = "314";
  document.getElementById("frame-chart").innerHTML = "";

  document.getElementById("drop-zone").classList.remove("hidden");
  document.getElementById("file-preview").classList.add("hidden");
  showSection("upload-section");

  const btn = document.getElementById("analyze-btn");
  if (btn) {
    btn.disabled = false;
    btn.innerHTML = '<span class="btn-icon">🔍</span><span>Analyze for Deepfakes</span>';
  }
}

async function analyzeVideo() {
  if (!currentFile) return;

  const btn = document.getElementById("analyze-btn");
  btn.disabled = true;
  btn.innerHTML = '<span class="btn-icon">⏳</span><span>Uploading...</span>';

  showSection("loading-section");
  animateLoadingSteps();

  const formData = new FormData();
  formData.append("file", currentFile);

  try {
    const response = await fetch(`${API_URL}/predict`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => ({ detail: "Unknown server error." }));
      throw new Error(errData.detail || `Server error: ${response.status}`);
    }

    const data = await response.json();
    lastResult = data;
    renderResults(data);
    showSection("results-section");

  } catch (err) {
    console.error("Analysis failed:", err);
    let msg = err.message || "Analysis failed.";
    if (msg.includes("fetch") || msg.includes("NetworkError") || msg.includes("Failed to fetch")) {
      msg = "⚠️ Cannot reach the AI server. The server might be waking up. Please wait ~60 sec and try again.";
    }
    showError(msg);
  }
}

function renderResults(data) {
  const isFake = data.verdict === "FAKE";
  const fakePct = data.fake_probability;
  const realPct = data.real_probability;

  const card = document.getElementById("verdict-card");
  card.classList.remove("is-fake", "is-real");
  card.classList.add(isFake ? "is-fake" : "is-real");

  const ring = document.getElementById("ring-fill");
  const circumference = 314;
  ring.style.stroke = isFake ? "#ef4444" : "#22c55e";
  setTimeout(() => {
    ring.style.strokeDashoffset = circumference - (fakePct / 100) * circumference;
  }, 100);

  document.getElementById("verdict-pct").textContent = `${fakePct}%`;
  const lbl = document.getElementById("verdict-label");
  lbl.textContent = data.verdict;
  lbl.style.color = isFake ? "#f87171" : "#4ade80";

  const badge = document.getElementById("verdict-badge");
  badge.textContent = isFake ? "⚠️  FAKE DETECTED" : "✅ REAL VIDEO";
  badge.className = "verdict-badge " + (isFake ? "fake" : "real");

  document.getElementById("stat-fake").textContent = `${fakePct}%`;
  document.getElementById("stat-real").textContent = `${realPct}%`;
  document.getElementById("stat-frames").textContent = `${data.frame_count} frames`;
  document.getElementById("stat-size").textContent = `${data.file_size_mb} MB`;

  renderFrameChart(data.per_frame_scores || []);
}

function renderFrameChart(scores) {
  const container = document.getElementById("frame-chart");
  container.innerHTML = "";

  if (!scores.length) {
    container.innerHTML = '<p style="color:var(--text-sub);font-size:13px;padding:20px 0;">No per-frame data.</p>';
    return;
  }

  scores.forEach((score, i) => {
    const isFakeBar = score > 50;
    const wrap = document.createElement("div");
    wrap.className = "bar-wrap";

    const bar = document.createElement("div");
    bar.className = `bar ${isFakeBar ? "bar-fake" : "bar-real"}`;
    bar.style.height = "0%";
    bar.setAttribute("data-tip", `Frame ${i + 1}: ${score}%`);

    wrap.appendChild(bar);
    container.appendChild(wrap);

    setTimeout(() => { bar.style.height = `${Math.max(4, score)}%`; }, 100 + i * 30);
  });
}

function showError(msg) {
  document.getElementById("error-msg").textContent = msg;
  showSection("error-section");
}

function animateLoadingSteps() {
  const steps = ["step-1", "step-2", "step-3"];
  steps.forEach(s => {
    document.getElementById(s).classList.remove("active", "done");
  });

  let i = 0;
  function next() {
    if (i > 0) {
      document.getElementById(steps[i - 1]).classList.remove("active");
      document.getElementById(steps[i - 1]).classList.add("done");
    }
    if (i < steps.length) {
      document.getElementById(steps[i]).classList.add("active");
      i++;
      setTimeout(next, i < steps.length ? 4000 : 99999);
    }
  }
  next();
}

function copyResult() {
  if (!lastResult) return;
  const { verdict, fake_probability, real_probability, frame_count, filename } = lastResult;
  const text =
    `DeepShield AI — DINO-G50 Result\n` +
    `File: ${filename}\n` +
    `Verdict: ${verdict}\n` +
    `Fake: ${fake_probability}% | Real: ${real_probability}%\n` +
    `Frames Analyzed: ${frame_count}`;

  navigator.clipboard?.writeText(text).then(() => {
    const btn = document.querySelector(".action-secondary");
    if (btn) {
      const orig = btn.textContent;
      btn.textContent = "✅ Copied!";
      setTimeout(() => { btn.textContent = orig; }, 2000);
    }
  }).catch(() => alert(text));
}

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / 1024 / 1024).toFixed(1) + " MB";
}
