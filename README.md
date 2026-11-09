# Hand gesture recognition: A2 demo

Upload / webcam -> class -> Grad-CAM -> latency, with a stress-test panel
(brightness, blur, rotation, noise) and clearly labelled synthetic examples.

## Layout
```
app.py
requirements.txt
model/final_real+synth_aug.pt     # weights (copy from Drive: cv_a2/results/)
assets/synthetic/*.jpg            # a few synthetic frames, shown with a SYNTHETIC label
notebooks/A2_train_eval.ipynb     # training / evaluation / ablation
```

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py                                          # opens http://127.0.0.1:7860
```
Webcam in the browser works on `localhost` or HTTPS.

## Run in Colab (public link)
```python
!pip install -q gradio
import os; os.environ["SHARE"] = "1"
!python app.py
```

## Data and weights
- Real frames: 24 short phone videos (4 gestures x 3 backgrounds x 2), cut into 256x256 frames.
- Synthetic frames: clips generated with Wan2.1 T2V 1.3B (offline, Colab T4), filtered by hand.
- Weights: MobileNetV2, `real+synth_aug` configuration (see report).

## Notes
- The app does not save uploaded images on purpose.
- Latency shown in the app = preprocessing + forward pass for one frame on the machine running the app.
