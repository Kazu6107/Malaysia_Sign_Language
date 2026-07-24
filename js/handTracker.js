import {
  HandLandmarker,
  FilesetResolver
} from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs';

const WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

let landmarker = null;

export async function initHandTracker() {
  const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
  landmarker = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
    runningMode: 'VIDEO',
    numHands: 1,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5
  });
  return landmarker;
}

/**
 * Runs detection on a video frame.
 * Returns { landmarks: [{x,y,z}, ...21] , handedness } or null if no hand found.
 */
export function detectFrame(video, timestampMs) {
  if (!landmarker) return null;
  const result = landmarker.detectForVideo(video, timestampMs);
  if (!result.landmarks || result.landmarks.length === 0) return null;
  return {
    landmarks: result.landmarks[0],
    handedness: result.handedness?.[0]?.[0]?.categoryName ?? 'Unknown'
  };
}

/**
 * Normalizes 21 landmarks relative to the wrist (point 0) and scales by the
 * distance from wrist to middle-finger MCP (point 9), so the same sign looks
 * the same regardless of hand size or distance from the camera.
 * Returns a flat Float32Array of length 63 (21 points * x,y,z).
 */
export function normalizeLandmarks(landmarks) {
  const wrist = landmarks[0];
  const mid = landmarks[9];
  const scale = Math.hypot(mid.x - wrist.x, mid.y - wrist.y, mid.z - wrist.z) || 1;

  const out = new Float32Array(landmarks.length * 3);
  landmarks.forEach((p, i) => {
    out[i * 3] = (p.x - wrist.x) / scale;
    out[i * 3 + 1] = (p.y - wrist.y) / scale;
    out[i * 3 + 2] = (p.z - wrist.z) / scale;
  });
  return out;
}

// Hand connections for skeleton drawing (MediaPipe's standard 21-point topology).
export const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],
  [13,17],[17,18],[18,19],[19,20],
  [0,17]
];

export function drawLandmarks(ctx, landmarks, width, height) {
  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = '#8b98a3';
  ctx.lineWidth = 2;
  HAND_CONNECTIONS.forEach(([a, b]) => {
    const pa = landmarks[a], pb = landmarks[b];
    ctx.beginPath();
    ctx.moveTo(pa.x * width, pa.y * height);
    ctx.lineTo(pb.x * width, pb.y * height);
    ctx.stroke();
  });
  landmarks.forEach((p, i) => {
    ctx.beginPath();
    ctx.arc(p.x * width, p.y * height, i === 0 ? 5 : 3.5, 0, Math.PI * 2);
    ctx.fillStyle = i === 0 ? '#4fb3a9' : '#e8a33d';
    ctx.fill();
  });
}
