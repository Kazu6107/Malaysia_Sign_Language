const DB_NAME = 'msl-recognizer';
const DB_VERSION = 1;
const STATIC_STORE = 'staticSamples';
const MOTION_STORE = 'motionSamples';

let dbPromise = null;

function openDb() {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STATIC_STORE)) {
        const s = db.createObjectStore(STATIC_STORE, { keyPath: 'id', autoIncrement: true });
        s.createIndex('label', 'label', { unique: false });
      }
      if (!db.objectStoreNames.contains(MOTION_STORE)) {
        const s = db.createObjectStore(MOTION_STORE, { keyPath: 'id', autoIncrement: true });
        s.createIndex('label', 'label', { unique: false });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return dbPromise;
}

async function addRecord(storeName, record) {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(storeName, 'readwrite');
    tx.objectStore(storeName).add(record);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

async function getAll(storeName) {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(storeName, 'readonly');
    const req = tx.objectStore(storeName).getAll();
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function clearStore(storeName) {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(storeName, 'readwrite');
    tx.objectStore(storeName).clear();
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

export async function addStaticSample(label, vector63) {
  return addRecord(STATIC_STORE, { label, vector: Array.from(vector63) });
}

export async function addMotionSample(label, sequence) {
  // sequence: array of {x,y,z} wrist-relative points over time
  return addRecord(MOTION_STORE, { label, sequence });
}

export async function getAllStaticSamples() {
  return getAll(STATIC_STORE);
}

export async function getAllMotionSamples() {
  return getAll(MOTION_STORE);
}

export async function getCounts() {
  const [statics, motions] = await Promise.all([getAllStaticSamples(), getAllMotionSamples()]);
  const counts = {};
  statics.forEach(r => { counts[r.label] = (counts[r.label] || 0) + 1; });
  motions.forEach(r => { counts[r.label] = (counts[r.label] || 0) + 1; });
  return counts;
}

export async function clearAll() {
  await clearStore(STATIC_STORE);
  await clearStore(MOTION_STORE);
}

export async function exportDataset() {
  const [statics, motions] = await Promise.all([getAllStaticSamples(), getAllMotionSamples()]);
  return { version: 1, exportedAt: new Date().toISOString(), statics, motions };
}

export async function importDataset(json) {
  if (!json || !Array.isArray(json.statics) || !Array.isArray(json.motions)) {
    throw new Error('Invalid dataset file — expected { statics: [...], motions: [...] }');
  }
  for (const rec of json.statics) await addStaticSample(rec.label, rec.vector);
  for (const rec of json.motions) await addMotionSample(rec.label, rec.sequence);
}
