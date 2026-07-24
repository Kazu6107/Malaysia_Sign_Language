/* global tf */
const MODEL_STORAGE_KEY = 'indexeddb://msl-recognizer-model';
const INPUT_SIZE = 63; // 21 landmarks * (x,y,z)

let model = null;
let labelList = [];

export function getLabelList() {
  return labelList;
}

export function isModelReady() {
  return !!model;
}

function buildModel(numClasses) {
  const m = tf.sequential();
  m.add(tf.layers.dense({ inputShape: [INPUT_SIZE], units: 128, activation: 'relu' }));
  m.add(tf.layers.dropout({ rate: 0.25 }));
  m.add(tf.layers.dense({ units: 64, activation: 'relu' }));
  m.add(tf.layers.dense({ units: numClasses, activation: 'softmax' }));
  m.compile({ optimizer: tf.train.adam(0.001), loss: 'categoricalCrossentropy', metrics: ['accuracy'] });
  return m;
}

/**
 * samples: [{label, vector: number[63]}, ...]
 * onEpoch: (epoch, totalEpochs, logs) => void
 */
export async function trainModel(samples, epochs, onEpoch) {
  const labelsSeen = [...new Set(samples.map(s => s.label))].sort();
  if (labelsSeen.length < 2) {
    throw new Error('Need samples for at least 2 different labels to train.');
  }
  labelList = labelsSeen;

  const xs = tf.tensor2d(samples.map(s => s.vector));
  const labelIdx = samples.map(s => labelList.indexOf(s.label));
  const ys = tf.oneHot(tf.tensor1d(labelIdx, 'int32'), labelList.length);

  model = buildModel(labelList.length);

  const history = await model.fit(xs, ys, {
    epochs,
    batchSize: 16,
    shuffle: true,
    validationSplit: samples.length > 40 ? 0.15 : 0,
    callbacks: {
      onEpochEnd: (epoch, logs) => onEpoch && onEpoch(epoch, epochs, logs)
    }
  });

  xs.dispose();
  ys.dispose();
  return history;
}

export function predict(vector63) {
  if (!model) return null;
  return tf.tidy(() => {
    const input = tf.tensor2d([Array.from(vector63)]);
    const probs = model.predict(input).dataSync();
    const ranked = Array.from(probs)
      .map((p, i) => ({ label: labelList[i], prob: p }))
      .sort((a, b) => b.prob - a.prob);
    return ranked;
  });
}

export async function saveModelToBrowser() {
  if (!model) throw new Error('No model to save.');
  await model.save(MODEL_STORAGE_KEY);
  localStorage.setItem('msl-recognizer-labels', JSON.stringify(labelList));
}

export async function loadModelFromBrowser() {
  try {
    model = await tf.loadLayersModel(MODEL_STORAGE_KEY);
    labelList = JSON.parse(localStorage.getItem('msl-recognizer-labels') || '[]');
    return labelList.length > 0;
  } catch (e) {
    return false;
  }
}

export async function downloadModel() {
  if (!model) throw new Error('No model to download.');
  await model.save('downloads://msl-recognizer-model');
  const blob = new Blob([JSON.stringify(labelList)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'msl-recognizer-labels.json';
  a.click();
  URL.revokeObjectURL(url);
}
