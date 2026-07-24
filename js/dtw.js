// Each motion sample is an array of {x, y} points — the index fingertip's
// position relative to the wrist, sampled across ~1.5s of recording. Using a
// wrist-relative point means the shape of the motion matters, not where the
// hand is in frame.

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

/** Classic DTW distance between two point sequences, normalized by path length. */
export function dtwDistance(seqA, seqB) {
  const n = seqA.length, m = seqB.length;
  if (n === 0 || m === 0) return Infinity;

  const cost = new Float32Array((n + 1) * (m + 1)).fill(Infinity);
  const idx = (i, j) => i * (m + 1) + j;
  cost[idx(0, 0)] = 0;

  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      const d = dist(seqA[i - 1], seqB[j - 1]);
      const best = Math.min(cost[idx(i - 1, j)], cost[idx(i, j - 1)], cost[idx(i - 1, j - 1)]);
      cost[idx(i, j)] = d + best;
    }
  }
  // Normalize by the warped path length so longer/shorter clips are comparable.
  return cost[idx(n, m)] / (n + m);
}

/**
 * templatesByLabel: { [label]: Array<sequence> }
 * Returns { label, distance } for the closest-matching template's label,
 * or null if the best distance exceeds `threshold`.
 */
export function classifyMotion(sequence, templatesByLabel, threshold = 0.35) {
  let best = null;
  for (const [label, templates] of Object.entries(templatesByLabel)) {
    for (const template of templates) {
      const d = dtwDistance(sequence, template);
      if (!best || d < best.distance) best = { label, distance: d };
    }
  }
  if (!best || best.distance > threshold) return null;
  return best;
}
