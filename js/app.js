import { ALL_LABELS, STATIC_LABELS, MOTION_LABELS } from './labels.js';
import { initHandTracker, detectFrame, normalizeLandmarks, drawLandmarks } from './handTracker.js';
import * as store from './dataStore.js';
import * as classifier from './classifier.js';
import { classifyMotion } from './dtw.js';

// ---------- DOM ----------
const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const octx = overlay.getContext('2d');
const cameraStatus = document.getElementById('cameraStatus');
const recIndicator = document.getElementById('recIndicator');
const startCameraBtn = document.getElementById('startCamera');
const labelSelect = document.getElementById('labelSelect');
const telHand = document.getElementById('telHand');
const telWrist = document.getElementById('telWrist');
const telFps = document.getElementById('telFps');

const captureBtn = document.getElementById('captureBtn');
const burstBtn = document.getElementById('burstBtn');
const motionRecordBtn = document.getElementById('motionRecordBtn');
const motionControls = document.getElementById('motionControls');
const sampleCounts = document.getElementById('sampleCounts');
const exportBtn = document.getElementById('exportData');
const importInput = document.getElementById('importData');
const clearBtn = document.getElementById('clearData');

const epochsInput = document.getElementById('epochsInput');
const trainBtn = document.getElementById('trainBtn');
const trainProgressWrap = document.getElementById('trainProgressWrap');
const trainProgressFill = document.getElementById('trainProgressFill');
const trainProgressLabel = document.getElementById('trainProgressLabel');
const lossChart = document.getElementById('lossChart');
const lctx = lossChart.getContext('2d');
const saveModelBtn = document.getElementById('saveModel');
const downloadModelBtn = document.getElementById('downloadModel');
const trainStatus = document.getElementById('trainStatus');

const predLabel = document.getElementById('predLabel');
const predConf = document.getElementById('predConf');
const topKEl = document.getElementById('topK');
const speakToggle = document.getElementById('speakToggle');

// ---------- State ----------
let currentTab = 'collect';
let cameraRunning = false;
let latestDetection = null; // { landmarks, handedness, vector }
let lastFrameTime = performance.now();
let motionBuffer = []; // rolling {x,y,t} of index fingertip, wrist-relative
let motionTemplatesByLabel = {}; // loaded lazily for recognize tab
let lastSpoken = '';

// ---------- Tabs ----------
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => switchTab(btn.dataset.tab));
});

function switchTab(tab) {
  currentTab = tab;
  document.querySelectorAll('.tab-btn').forEach(b => {
    const active = b.dataset.tab === tab;
    b.classList.toggle('active', active);
    b.setAttribute('aria-selected', String(active));
  });
  document.querySelectorAll('.panel').forEach(p => {
    p.hidden = p.dataset.panel !== tab;
  });
  if (tab === 'recognize') loadMotionTemplates();
}

// ---------- Label select ----------
ALL_LABELS.forEach(l => {
  const opt = document.createElement('option');
  opt.value = l;
  opt.textContent = l;
  labelSelect.appendChild(opt);
});

function isMotionLabel(l) {
  return MOTION_LABELS.includes(l);
}

document.querySelectorAll('input[name="captureMode"]').forEach(r => {
  r.addEventListener('change', updateCaptureModeUI);
});
labelSelect.addEventListener('change', updateCaptureModeUI);

function updateCaptureModeUI() {
  const mode = document.querySelector('input[name="captureMode"]:checked').value;
  motionControls.hidden = mode !== 'motion';
  document.getElementById('captureBtn').hidden = mode !== 'static';
  document.getElementById('burstBtn').hidden = mode !== 'static';
}
updateCaptureModeUI();

// ---------- Camera ----------
startCameraBtn.addEventListener('click', startCamera);

async function startCamera() {
  startCameraBtn.disabled = true;
  startCameraBtn.textContent = 'Starting…';
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
    video.srcObject = stream;
    await video.play();
    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;
    cameraStatus.textContent = 'loading model…';
    await initHandTracker();
    cameraStatus.textContent = 'camera live';
    cameraRunning = true;
    captureBtn.disabled = false;
    burstBtn.disabled = false;
    motionRecordBtn.disabled = false;
    startCameraBtn.textContent = 'Camera running';
    requestAnimationFrame(loop);
  } catch (err) {
    console.error(err);
    cameraStatus.textContent = 'camera error';
    startCameraBtn.textContent = 'Retry camera';
    startCameraBtn.disabled = false;
    alert('Could not start the camera: ' + err.message + '\nMake sure this page is served over HTTPS (or localhost) and camera permission is allowed.');
  }
}

function loop() {
  if (!cameraRunning) return;
  const now = performance.now();
  const detection = detectFrame(video, now);

  if (detection) {
    const vector = normalizeLandmarks(detection.landmarks);
    latestDetection = { ...detection, vector };
    drawLandmarks(octx, detection.landmarks, overlay.width, overlay.height);
    telHand.textContent = detection.handedness;
    const wrist = detection.landmarks[0];
    telWrist.textContent = `${wrist.x.toFixed(2)}, ${wrist.y.toFixed(2)}`;

    // maintain rolling motion buffer: index fingertip (8) relative to wrist, from normalized vector
    const idx8 = { x: vector[8 * 3], y: vector[8 * 3 + 1], t: now };
    motionBuffer.push(idx8);
    motionBuffer = motionBuffer.filter(p => now - p.t <= 1600);

    if (currentTab === 'recognize' && classifier.isModelReady()) {
      runRecognition(vector);
    }
  } else {
    latestDetection = null;
    octx.clearRect(0, 0, overlay.width, overlay.height);
    telHand.textContent = 'none';
    telWrist.textContent = '—';
  }

  const dt = now - lastFrameTime;
  lastFrameTime = now;
  telFps.textContent = dt > 0 ? Math.round(1000 / dt) : '—';

  requestAnimationFrame(loop);
}

// ---------- Collect: static ----------
captureBtn.addEventListener('click', () => captureStaticSample());
burstBtn.addEventListener('click', () => burstCapture());

async function captureStaticSample() {
  if (!latestDetection) { flashStatus('no hand detected'); return; }
  const label = labelSelect.value;
  await store.addStaticSample(label, latestDetection.vector);
  refreshCounts();
  flashCameraBadge('captured ' + label);
}

async function burstCapture() {
  const label = labelSelect.value;
  showRecording(true);
  let captured = 0;
  const start = performance.now();
  while (performance.now() - start < 2000) {
    if (latestDetection) {
      await store.addStaticSample(label, latestDetection.vector);
      captured++;
    }
    await new Promise(r => setTimeout(r, 120));
  }
  showRecording(false);
  refreshCounts();
  flashCameraBadge(`captured ${captured} frames of ${label}`);
}

// ---------- Collect: motion ----------
motionRecordBtn.addEventListener('click', recordMotionSample);

async function recordMotionSample() {
  const label = labelSelect.value;
  showRecording(true);
  const start = performance.now();
  const points = [];
  while (performance.now() - start < 1500) {
    if (latestDetection) {
      const v = latestDetection.vector;
      points.push({ x: v[8 * 3], y: v[8 * 3 + 1] });
    }
    await new Promise(r => setTimeout(r, 40));
  }
  showRecording(false);
  if (points.length < 5) { flashStatus('too few frames, try again'); return; }
  await store.addMotionSample(label, points);
  refreshCounts();
  flashCameraBadge(`captured motion sample of ${label}`);
}

function showRecording(on) {
  recIndicator.hidden = !on;
}

function flashCameraBadge(msg) {
  const prev = cameraStatus.textContent;
  cameraStatus.textContent = msg;
  setTimeout(() => { cameraStatus.textContent = cameraRunning ? 'camera live' : prev; }, 1200);
}

function flashStatus(msg) {
  flashCameraBadge(msg);
}

// ---------- Sample counts ----------
async function refreshCounts() {
  const counts = await store.getCounts();
  sampleCounts.innerHTML = '';
  ALL_LABELS.forEach(l => {
    const n = counts[l] || 0;
    const chip = document.createElement('div');
    chip.className = 'count-chip ' + (n === 0 ? '' : n < 20 ? 'low' : 'ok');
    chip.innerHTML = `<span class="lbl">${l}</span><span class="cnt">${n}</span>`;
    sampleCounts.appendChild(chip);
  });
}
refreshCounts();

// ---------- Export / import / clear ----------
exportBtn.addEventListener('click', async () => {
  const data = await store.exportDataset();
  const blob = new Blob([JSON.stringify(data)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `msl-dataset-${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
});

importInput.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  try {
    const text = await file.text();
    const json = JSON.parse(text);
    await store.importDataset(json);
    await refreshCounts();
    alert('Dataset imported.');
  } catch (err) {
    alert('Import failed: ' + err.message);
  }
  importInput.value = '';
});

clearBtn.addEventListener('click', async () => {
  if (!confirm('Delete all collected samples? This cannot be undone.')) return;
  await store.clearAll();
  await refreshCounts();
});

// ---------- Train ----------
trainBtn.addEventListener('click', runTraining);

async function runTraining() {
  const statics = await store.getAllStaticSamples();
  if (statics.length < 10) {
    alert('Collect more static samples first (aim for at least 20 per label).');
    return;
  }
  trainBtn.disabled = true;
  trainProgressWrap.hidden = false;
  trainProgressFill.style.width = '0%';
  const epochs = parseInt(epochsInput.value, 10) || 60;
  const lossHistory = [];

  try {
    await classifier.trainModel(statics, epochs, (epoch, total, logs) => {
      const pct = Math.round(((epoch + 1) / total) * 100);
      trainProgressFill.style.width = pct + '%';
      trainProgressLabel.textContent = `epoch ${epoch + 1}/${total} — loss ${logs.loss.toFixed(3)}${logs.val_loss ? ', val_loss ' + logs.val_loss.toFixed(3) : ''}`;
      lossHistory.push(logs.loss);
      drawLossChart(lossHistory);
    });
    trainStatus.textContent = `model trained on ${statics.length} samples across ${classifier.getLabelList().length} labels`;
    saveModelBtn.disabled = false;
    downloadModelBtn.disabled = false;
  } catch (err) {
    alert('Training failed: ' + err.message);
  } finally {
    trainBtn.disabled = false;
  }
}

function drawLossChart(history) {
  const w = lossChart.width, h = lossChart.height;
  lctx.clearRect(0, 0, w, h);
  if (history.length < 2) return;
  const max = Math.max(...history);
  const min = Math.min(...history);
  const range = (max - min) || 1;
  lctx.strokeStyle = '#e8a33d';
  lctx.lineWidth = 2;
  lctx.beginPath();
  history.forEach((v, i) => {
    const x = (i / (history.length - 1)) * (w - 12) + 6;
    const y = h - 10 - ((v - min) / range) * (h - 20);
    if (i === 0) lctx.moveTo(x, y); else lctx.lineTo(x, y);
  });
  lctx.stroke();
}

saveModelBtn.addEventListener('click', async () => {
  await classifier.saveModelToBrowser();
  trainStatus.textContent = 'model saved to browser storage';
});

downloadModelBtn.addEventListener('click', () => classifier.downloadModel());

// try loading a previously saved model on startup
classifier.loadModelFromBrowser().then(ok => {
  if (ok) {
    trainStatus.textContent = `loaded saved model (${classifier.getLabelList().length} labels)`;
    saveModelBtn.disabled = false;
    downloadModelBtn.disabled = false;
  }
});

// ---------- Recognize ----------
async function loadMotionTemplates() {
  const motions = await store.getAllMotionSamples();
  motionTemplatesByLabel = {};
  motions.forEach(rec => {
    (motionTemplatesByLabel[rec.label] ??= []).push(rec.sequence);
  });
}

function runRecognition(vector) {
  const ranked = classifier.predict(vector);
  if (!ranked) return;

  // Check motion buffer against DTW templates; motion match overrides static
  // prediction since a static classifier will otherwise fire mid-gesture.
  let motionMatch = null;
  if (Object.keys(motionTemplatesByLabel).length && motionBuffer.length > 10) {
    motionMatch = classifyMotion(motionBuffer.map(p => ({ x: p.x, y: p.y })), motionTemplatesByLabel);
  }

  if (motionMatch) {
    predLabel.textContent = motionMatch.label;
    predConf.textContent = `motion match — dtw distance ${motionMatch.distance.toFixed(3)}`;
    renderTopK([{ label: motionMatch.label, prob: 1 }]);
    speak(motionMatch.label);
  } else {
    const top = ranked[0];
    predLabel.textContent = top.label;
    predConf.textContent = `confidence ${(top.prob * 100).toFixed(1)}%`;
    renderTopK(ranked.slice(0, 5));
    if (top.prob > 0.85) speak(top.label);
  }
}

function renderTopK(ranked) {
  topKEl.innerHTML = '';
  ranked.forEach(r => {
    const row = document.createElement('div');
    row.className = 'top-k-row';
    row.innerHTML = `
      <span class="k-label">${r.label}</span>
      <span class="k-bar-track"><span class="k-bar-fill" style="width:${(r.prob * 100).toFixed(0)}%"></span></span>
      <span class="k-pct">${(r.prob * 100).toFixed(0)}%</span>`;
    topKEl.appendChild(row);
  });
}

function speak(label) {
  if (!speakToggle.checked) return;
  if (label === lastSpoken) return;
  lastSpoken = label;
  const u = new SpeechSynthesisUtterance(label);
  speechSynthesis.speak(u);
  setTimeout(() => { lastSpoken = ''; }, 1200);
}
