/* ═══════════════════════════════════════════════════════════════════════════
   VisionTrack AI — Frontend Controller
   Handles: particles, loading, nav, stream, stats polling, file upload,
            drag-drop, toggles, zones, snapshots, reports
   ═══════════════════════════════════════════════════════════════════════════ */

"use strict";

// ── Particle System ───────────────────────────────────────────────────────────
(function initParticles() {
  const canvas = document.getElementById("particleCanvas");
  const ctx = canvas.getContext("2d");
  let W, H, particles;

  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }

  function mkParticle() {
    return {
      x: Math.random() * W,
      y: Math.random() * H,
      r: Math.random() * 1.5 + 0.3,
      vx: (Math.random() - 0.5) * 0.3,
      vy: (Math.random() - 0.5) * 0.3,
      alpha: Math.random() * 0.4 + 0.05,
      color: Math.random() < 0.5 ? "0,245,212" : "182,85,255",
    };
  }

  function init() {
    resize();
    particles = Array.from({ length: 120 }, mkParticle);
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    for (const p of particles) {
      p.x += p.vx;
      p.y += p.vy;
      if (p.x < 0) p.x = W;
      if (p.x > W) p.x = 0;
      if (p.y < 0) p.y = H;
      if (p.y > H) p.y = 0;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${p.color},${p.alpha})`;
      ctx.fill();
    }
    requestAnimationFrame(draw);
  }

  window.addEventListener("resize", resize);
  init();
  draw();
})();


// ── Loading Screen ────────────────────────────────────────────────────────────
window.addEventListener("load", () => {
  setTimeout(() => {
    const ls = document.getElementById("loadingScreen");
    ls.classList.add("fade-out");
    setTimeout(() => ls.remove(), 600);
  }, 2400);
});


// ── Section Navigation ────────────────────────────────────────────────────────
const SECTIONS = {
  dashboard:  document.getElementById("dashboardSection"),
  analytics:  document.getElementById("analyticsSection"),
  tracking:   document.getElementById("trackingSection"),
  zones:      document.getElementById("zonesSection"),
  snapshots:  document.getElementById("snapshotsSection"),
  reports:    document.getElementById("reportsSection"),
};
const heroSection = document.getElementById("heroSection");
let currentSection = null;

document.querySelectorAll(".nav-link").forEach(link => {
  link.addEventListener("click", e => {
    e.preventDefault();
    const key = link.dataset.section;
    document.querySelectorAll(".nav-link").forEach(l => l.classList.remove("active"));
    link.classList.add("active");
    showSection(key);
    if (key === "snapshots") loadSnapshots();
  });
});

function showSection(key) {
  heroSection.classList.add("hidden");
  Object.values(SECTIONS).forEach(s => s.classList.add("hidden"));
  if (SECTIONS[key]) SECTIONS[key].classList.remove("hidden");
  currentSection = key;
}

function showHero() {
  heroSection.classList.remove("hidden");
  Object.values(SECTIONS).forEach(s => s.classList.add("hidden"));
  currentSection = null;
}


// ── Stream State ──────────────────────────────────────────────────────────────
let streamActive = false;

function setStreamActive(active, source = "unknown") {
  streamActive = active;
  const dot    = document.getElementById("statusDot");
  const text   = document.getElementById("statusText");
  const badge  = document.getElementById("sourceBadge");
  const overlay= document.getElementById("streamOverlay");

  if (active) {
    dot.className  = "status-dot online";
    text.textContent = "ONLINE";
    badge.textContent = source.toUpperCase();
    badge.className   = "badge badge-green";
    if (overlay) overlay.classList.add("hidden");
    if (!currentSection) showSection("dashboard");
  } else {
    dot.className  = "status-dot offline";
    text.textContent = "OFFLINE";
    badge.textContent = "NO SOURCE";
    badge.className   = "badge badge-red";
    if (overlay) overlay.classList.remove("hidden");
  }
}


// ── Webcam ────────────────────────────────────────────────────────────────────
async function startWebcam() {
  try {
    const res = await fetch("/api/start_webcam", { method: "POST" });
    const data = await res.json();
    if (data.success) {
      setStreamActive(true, "webcam");
      toast("Webcam stream started ◉");
    } else {
      toast("Webcam error: " + data.error, true);
    }
  } catch (err) {
    toast("Network error: " + err.message, true);
  }
}


// ── Video Upload ──────────────────────────────────────────────────────────────
async function uploadVideo(event) {
  const file = event.target.files[0];
  if (!file) return;
  const formData = new FormData();
  formData.append("video", file);
  toast(`Uploading ${file.name}…`);
  try {
    const res = await fetch("/api/upload_video", { method: "POST", body: formData });
    const data = await res.json();
    if (data.success) {
      setStreamActive(true, "video");
      toast(`Playing: ${data.filename}`);
    } else {
      toast("Upload error: " + data.error, true);
    }
  } catch (err) {
    toast("Upload failed: " + err.message, true);
  }
  // Reset input so same file can be reselected
  event.target.value = "";
}


// ── Stop & Reset ──────────────────────────────────────────────────────────────
async function stopStream() {
  await fetch("/api/stop", { method: "POST" });
  setStreamActive(false);
  toast("Stream stopped ■");
}

async function resetSession() {
  await fetch("/api/reset", { method: "POST" });
  toast("Session analytics reset ↺");
}


// ── Toggles ───────────────────────────────────────────────────────────────────
function toggleTrails() {
  fetch("/api/toggle_trails", { method: "POST" })
    .then(r => r.json())
    .then(d => {
      const el = document.getElementById("trailsToggle");
      el.classList.toggle("on", d.show_trails);
    });
}

function toggleHeatmap() {
  fetch("/api/toggle_heatmap", { method: "POST" })
    .then(r => r.json())
    .then(d => {
      const el = document.getElementById("heatmapToggle");
      el.classList.toggle("on", d.show_heatmap);
    });
}

function updateConf(val) {
  document.getElementById("confVal").textContent = `${val}%`;
  fetch("/api/set_confidence", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ confidence: val / 100 }),
  });
}


// ── Stats Polling (every 1 second) ───────────────────────────────────────────
let _prevTotal = 0;

function animateCounter(elId, newVal) {
  const el = document.getElementById(elId);
  if (!el) return;
  const cur = parseInt(el.textContent) || 0;
  if (cur === newVal) return;
  const step = Math.ceil(Math.abs(newVal - cur) / 10);
  const dir  = newVal > cur ? 1 : -1;
  let v = cur;
  const timer = setInterval(() => {
    v += dir * step;
    if ((dir === 1 && v >= newVal) || (dir === -1 && v <= newVal)) {
      v = newVal;
      clearInterval(timer);
    }
    el.textContent = v;
  }, 30);
}

function updateStats(d) {
  // Stat cards
  animateCounter("statTotal",  d.total_detections || 0);
  animateCounter("statActive", d.active_tracks || 0);
  animateCounter("statFPS",    Math.round(d.fps || 0));

  setText("statMostFreq",  d.most_frequent || "—");
  setText("statAccuracy",  `${d.avg_confidence || 0}%`);
  setText("statSession",   formatTime(d.session_seconds || 0));
  setText("fpsBadge",      `${Math.round(d.fps || 0)} FPS`);

  // Analytics section
  setText("anTotal",  d.total_detections || 0);
  setText("anFrames", d.frame_count || 0);
  setText("anFPS",    `${Math.round(d.fps || 0)}`);
  setText("anConf",   `${d.avg_confidence || 0}%`);

  // Label bars
  renderLabelBars("labelBars", d.label_counts);
  renderLabelBars("analyticsLabelBars", d.label_counts);

  // Counting line
  setText("inboundCount",  d.inbound_count  || 0);
  setText("outboundCount", d.outbound_count || 0);

  // Tracking table
  renderTrackTable(d.tracks_table || []);

  // Zone alerts
  if (d.zone_alerts && d.zone_alerts.length > 0) {
    showZoneBanner("🚨 ZONE ALERT: " + d.zone_alerts.join(", "));
  }
}

function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

function formatTime(secs) {
  if (secs < 60)   return `${secs}s`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m ${secs % 60}s`;
  return `${Math.floor(secs / 3600)}h ${Math.floor((secs % 3600) / 60)}m`;
}

function renderLabelBars(containerId, labelCounts) {
  const el = document.getElementById(containerId);
  if (!el || !labelCounts) return;
  const entries = Object.entries(labelCounts);
  if (entries.length === 0) { el.innerHTML = ""; return; }
  const maxVal = Math.max(...entries.map(e => e[1]));
  el.innerHTML = entries.slice(0, 10).map(([label, count]) => `
    <div class="label-row">
      <span class="label-name">${label}</span>
      <div class="label-track">
        <div class="label-fill" style="width:${(count / maxVal * 100).toFixed(1)}%"></div>
      </div>
      <span class="label-count">${count}</span>
    </div>
  `).join("");
}

function renderTrackTable(tracks) {
  const tbody = document.getElementById("trackTableBody");
  if (!tbody) return;
  if (!tracks || tracks.length === 0) {
    tbody.innerHTML = `<tr><td colspan="4" style="text-align:center;color:var(--muted)">No active tracks</td></tr>`;
    return;
  }
  tbody.innerHTML = tracks.map(t => `
    <tr>
      <td><span class="track-id">#${t.id}</span></td>
      <td>${t.label}</td>
      <td>${t.confidence}</td>
      <td><span class="badge badge-green">ACTIVE</span></td>
    </tr>
  `).join("");
}

// Poll API every 1000ms
setInterval(async () => {
  try {
    const res  = await fetch("/api/stats");
    const data = await res.json();
    updateStats(data);
    if (streamActive !== data.running) {
      setStreamActive(data.running, data.source);
    }
  } catch (_) { /* ignore network blips */ }
}, 1000);


// ── Drag & Drop Video ─────────────────────────────────────────────────────────
const streamWrapper = document.querySelector(".stream-wrapper");
const dropZone = document.getElementById("dropZone");

if (streamWrapper) {
  streamWrapper.addEventListener("dragover", e => {
    e.preventDefault();
    dropZone.classList.add("active");
  });
  streamWrapper.addEventListener("dragleave", () => dropZone.classList.remove("active"));
  streamWrapper.addEventListener("drop", async e => {
    e.preventDefault();
    dropZone.classList.remove("active");
    const file = e.dataTransfer.files[0];
    if (!file || !file.type.startsWith("video/")) {
      toast("Please drop a video file.", true);
      return;
    }
    const formData = new FormData();
    formData.append("video", file);
    toast(`Uploading ${file.name}…`);
    const res  = await fetch("/api/upload_video", { method: "POST", body: formData });
    const data = await res.json();
    if (data.success) {
      setStreamActive(true, "video");
      toast(`Playing: ${data.filename}`);
    } else {
      toast("Upload error: " + data.error, true);
    }
  });
}


// ── Counting Line ─────────────────────────────────────────────────────────────
function setCountingLine() {
  const y = parseInt(prompt("Enter Y pixel position for counting line (e.g. 360):", "360"));
  if (isNaN(y)) return;
  fetch("/api/set_counting_line", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pt1: [0, y], pt2: [1280, y] }),
  }).then(() => toast(`Counting line set at Y=${y}`));
}

function setCountingLineY() {
  const y = parseInt(document.getElementById("lineY").value);
  if (isNaN(y)) { toast("Enter a valid Y value.", true); return; }
  fetch("/api/set_counting_line", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pt1: [0, y], pt2: [1280, y] }),
  }).then(() => toast(`Counting line set at Y=${y}`));
}


// ── Zones ─────────────────────────────────────────────────────────────────────
const registeredZones = [];

async function addZone() {
  const name = document.getElementById("zoneName").value.trim() || "Zone";
  const polyStr = document.getElementById("zonePolygon").value.trim();
  let polygon;
  try {
    polygon = JSON.parse(polyStr);
    if (!Array.isArray(polygon) || polygon.length < 3) throw new Error();
  } catch {
    toast("Invalid polygon JSON — need at least 3 [x,y] pairs.", true);
    return;
  }
  const res  = await fetch("/api/add_zone", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, polygon }),
  });
  const data = await res.json();
  if (data.success) {
    registeredZones.push({ name, polygon });
    renderZoneList();
    toast(`Zone "${name}" added.`);
    document.getElementById("zoneName").value = "";
    document.getElementById("zonePolygon").value = "";
  } else {
    toast(data.error, true);
  }
}

function renderZoneList() {
  const el = document.getElementById("zoneList");
  if (!el) return;
  el.innerHTML = registeredZones.map((z, i) => `
    <div class="zone-item">
      <span>⬠ ${z.name}</span>
      <span style="font-size:0.7rem;color:var(--muted)">${z.polygon.length} pts</span>
    </div>
  `).join("");
}

async function clearZones() {
  await fetch("/api/clear_zones", { method: "POST" });
  registeredZones.length = 0;
  renderZoneList();
  toast("All zones cleared.");
}


// ── Snapshot ──────────────────────────────────────────────────────────────────
async function takeSnapshot() {
  const res  = await fetch("/api/snapshot", { method: "POST" });
  const data = await res.json();
  if (data.success) {
    toast(`Snapshot saved 📷`);
    loadSnapshots();
  } else {
    toast("Snapshot failed: " + data.error, true);
  }
}

async function loadSnapshots() {
  const res  = await fetch("/api/list_snapshots");
  const data = await res.json();
  const grid = document.getElementById("snapshotGrid");
  if (!grid) return;
  if (!data.snapshots || data.snapshots.length === 0) {
    grid.innerHTML = `<div class="snap-empty">No snapshots yet — start a stream and capture a frame.</div>`;
    return;
  }
  grid.innerHTML = data.snapshots.map(url => `
    <div class="snap-thumb">
      <a href="${url}" target="_blank">
        <img src="${url}" alt="snapshot" loading="lazy" />
      </a>
    </div>
  `).join("");
}


// ── Reports ───────────────────────────────────────────────────────────────────
async function exportCSV() {
  const res  = await fetch("/api/export_csv", { method: "POST" });
  const data = await res.json();
  if (data.success) {
    document.getElementById("csvLink").innerHTML =
      `<a href="${data.url}" download>⬇ Download ${data.filename}</a>`;
    toast("CSV report exported ✓");
  } else {
    toast("Export failed: " + data.error, true);
  }
}

async function exportJSON() {
  const res  = await fetch("/api/export_json", { method: "POST" });
  const data = await res.json();
  if (data.success) {
    document.getElementById("jsonLink").innerHTML =
      `<a href="${data.url}" download>⬇ Download ${data.filename}</a>`;
    toast("JSON report exported ✓");
  } else {
    toast("Export failed: " + data.error, true);
  }
}


// ── Toast ─────────────────────────────────────────────────────────────────────
let _toastTimer = null;
function toast(msg, isError = false) {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.className = "toast show" + (isError ? " error" : "");
  if (_toastTimer) clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => { el.className = "toast"; }, 3200);
}


// ── Zone Alert Banner ─────────────────────────────────────────────────────────
let _bannerTimer = null;
function showZoneBanner(msg) {
  const el = document.getElementById("zoneBanner");
  el.textContent = msg;
  el.classList.remove("hidden");
  if (_bannerTimer) clearTimeout(_bannerTimer);
  _bannerTimer = setTimeout(() => el.classList.add("hidden"), 4000);
}


// ── Initial state ─────────────────────────────────────────────────────────────
setStreamActive(false);
document.getElementById("trailsToggle").classList.add("on");  // trails on by default
