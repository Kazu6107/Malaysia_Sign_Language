# MSL Recognizer

An in-browser Malaysian Sign Language recognizer — hand tracking, training data
collection, and gesture classification, all running client-side with no
backend. Built as the successor to an early prototype made in Scratch (via
Machine Learning for Kids) to validate the label set and hand-landmark
approach.

- **Hand tracking:** MediaPipe Tasks Vision `HandLandmarker` (21 landmarks/hand)
- **Static signs (A–Y minus J, 0–9):** a small neural net trained in-browser with TensorFlow.js
- **Motion signs (J, Z, 10):** matched against recorded templates with Dynamic Time Warping
- **Storage:** IndexedDB, entirely local to the browser — nothing leaves the device

## Running locally

Because camera access and ES modules require a real origin (not `file://`),
serve the folder instead of opening `index.html` directly:

```bash
python3 -m http.server 8000
# then open http://localhost:8000
```

## Using it

1. **Collect** — pick a label, hold the sign in frame, hit *Capture sample*
   (or *Burst capture* for a 2s run of frames). Aim for 40–60 samples per
   static label from a few angles/distances. For J, Z, and 10, switch to
   *motion sign* mode and record the traced motion instead.
2. **Train** — once you have samples for at least two labels, hit *Train
   model*. Training runs entirely in your browser; nothing is uploaded.
3. **Recognize** — live predictions from the trained model, plus DTW
   matching against your recorded motion templates for J/Z/10.

Use **Export dataset** regularly to back up your collected samples as JSON —
IndexedDB is per-browser-profile and won't survive a cache clear.

## Deploying to GitHub Pages

1. Create a new GitHub repo and push this folder's contents to it:
   ```bash
   git init
   git add .
   git commit -m "Initial MSL recognizer"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git push -u origin main
   ```
2. In the repo, go to **Settings → Pages**.
3. Under **Build and deployment → Source**, choose **Deploy from a branch**,
   branch `main`, folder `/ (root)`. Save.
4. After a minute or two, your site will be live at
   `https://<your-username>.github.io/<repo-name>/`.

GitHub Pages serves over HTTPS by default, so camera access
(`getUserMedia`) works without any extra configuration.

### Notes specific to this setup

- MediaPipe's model and WASM files load from Google's CDN, and TensorFlow.js
  loads from jsDelivr — both are fetched at runtime, so there's nothing large
  to commit to the repo.
- GitHub Pages serves your site from a subpath
  (`<username>.github.io/<repo-name>/`), which is why every path in this
  project (`css/style.css`, `js/app.js`, etc.) is relative rather than
  absolute — no changes needed there.
- Training data lives in each visitor's own browser (IndexedDB), so a model
  trained on your laptop won't automatically appear for someone else opening
  the site. If you want a shared, pre-trained model, train it locally, click
  **Download model files** on the Train tab, and commit the resulting
  `model.json`/`.bin` files — then wire up a "load bundled model" option in
  `classifier.js` alongside the local-storage one.

## Project structure

```
index.html
css/style.css
js/
  labels.js        static + motion label sets
  handTracker.js    MediaPipe wrapper, wrist-relative normalization, skeleton drawing
  dataStore.js      IndexedDB storage for collected samples
  classifier.js     TensorFlow.js model build/train/predict/save
  dtw.js            Dynamic Time Warping matcher for motion signs
  app.js            camera loop + UI wiring
```
