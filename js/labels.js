// Static signs: held in one position, classified frame-by-frame.
export const STATIC_LABELS = [
  'A','B','C','D','E','F','G','H','I','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y',
  '0','1','2','3','4','5','6','7','8','9'
];

// Motion signs: traced through space, matched with DTW against recorded templates.
export const MOTION_LABELS = ['J', 'Z', '10'];

export const ALL_LABELS = [...STATIC_LABELS, ...MOTION_LABELS];
