"""Hand gesture recognition demo: upload / webcam -> prediction -> Grad-CAM -> latency.
Stress-test panel (brightness / blur / rotation / noise) and labelled synthetic examples.
Run:  python app.py   (weights expected in model/final_real+synth_aug.pt)
"""
import os
import glob
import time
import platform

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFilter, ImageEnhance
from torchvision import models, transforms as T
import matplotlib
import gradio as gr

CLASSES = ["fist", "peace", "stop", "thumbs_up"]
MODEL_PATH = os.environ.get("MODEL_PATH", "model/final_real+synth_aug.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
TF = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN, STD)])


def build_model():
    m = models.mobilenet_v2(weights=None)
    m.classifier[1] = nn.Linear(m.last_channel, len(CLASSES))
    for mod in m.modules():
        if isinstance(mod, nn.ReLU6):
            mod.inplace = False          # required for Grad-CAM hooks
    return m


MODEL = build_model()
MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
MODEL.to(DEVICE).eval()


def crop_square(img, v_offset=0.2):
    """Same square crop as used for the training frames."""
    w, h = img.size
    s = min(w, h)
    x0 = (w - s) // 2
    y0 = int((h - s) * v_offset)
    return img.crop((x0, y0, x0 + s, y0 + s))


def degrade(img, brightness, blur, rotate, noise):
    if brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brightness)
    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    if rotate:
        img = img.rotate(rotate, resample=Image.BILINEAR)
    if noise > 0:
        a = np.asarray(img).astype(np.float32)
        a += np.random.default_rng(0).normal(0, noise, a.shape)
        img = Image.fromarray(a.clip(0, 255).astype("uint8"))
    return img


def infer(sq):
    """Preprocessing + forward pass. Returns (probs, tensor, milliseconds)."""
    t0 = time.perf_counter()
    x = TF(sq)
    with torch.no_grad():
        probs = torch.softmax(MODEL(x.unsqueeze(0).to(DEVICE)), 1)[0].cpu().numpy()
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return probs, x, (time.perf_counter() - t0) * 1000


def gradcam(x):
    store = {}
    layer = MODEL.features[-1]

    def fwd_hook(m, i, o):
        store["a"] = o
        o.register_hook(lambda g: store.__setitem__("g", g))

    h = layer.register_forward_hook(fwd_hook)
    try:
        with torch.enable_grad():
            out = MODEL(x.unsqueeze(0).to(DEVICE))
            cls = int(out.argmax(1))
            MODEL.zero_grad()
            out[0, cls].backward()
    finally:
        h.remove()
    w = store["g"].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * store["a"]).sum(1, keepdim=True))
    cam = torch.nn.functional.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.detach().cpu().numpy()


def overlay(img, cam, alpha=0.45):
    heat = (matplotlib.colormaps["jet"](cam)[..., :3] * 255).astype(np.float32)
    base = np.asarray(img.convert("RGB").resize((224, 224))).astype(np.float32)
    return Image.fromarray(((1 - alpha) * base + alpha * heat).clip(0, 255).astype("uint8"))


# warm-up so the first measured latency is not inflated
for _ in range(3):
    infer(Image.new("RGB", (256, 256), (128, 128, 128)))

LAT = []


def device_name():
    if DEVICE == "cuda":
        return torch.cuda.get_device_name(0)
    return platform.processor() or platform.machine()


def predict(img, brightness, blur, rotate, noise, request: gr.Request = None):
    if img is None:
        return None, None, None, "Загрузите изображение или сделайте снимок с камеры."
    sq = crop_square(img.convert("RGB"))
    p_orig, _, _ = infer(sq)
    deg = degrade(sq, brightness, blur, rotate, noise)
    p_deg, x, ms = infer(deg)
    LAT.append(ms)
    del LAT[:-50]
    t0 = time.perf_counter()
    cam = gradcam(x)
    cam_ms = (time.perf_counter() - t0) * 1000
    top = int(np.argmax(p_deg))
    ua = ""
    if request is not None:
        ua = request.headers.get("user-agent", "")
    info = (
        f"**Предсказание:** {CLASSES[top]} ({p_deg[top]:.1%})\n\n"
        f"**Latency** (предобработка + модель, 1 кадр): {ms:.1f} мс; "
        f"среднее за последние {len(LAT)} запросов: {np.mean(LAT):.1f} мс; "
        f"Grad-CAM: {cam_ms:.0f} мс\n\n"
        f"**Среда:** устройство инференса: {DEVICE} ({device_name()}); "
        f"torch {torch.__version__}; {platform.system()} {platform.release()}"
        + (f"; браузер: {ua}" if ua else "")
    )
    return (
        {c: float(p) for c, p in zip(CLASSES, p_deg)},
        {c: float(p) for c, p in zip(CLASSES, p_orig)},
        overlay(deg, cam),
        info,
    )


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_SYN = os.path.join(BASE_DIR, "assets", "synthetic")
SYNTH_FILES = sorted(p for ext in ("jpg", "jpeg", "png", "webp")
                     for p in glob.glob(os.path.join(_SYN, f"*.{ext}")))
SYNTH_VIDEOS = sorted(glob.glob(os.path.join(_SYN, "*.mp4")))
print("Синтетических: картинок", len(SYNTH_FILES), "| видео", len(SYNTH_VIDEOS))

ABOUT = """
### Модель
MobileNetV2 (ImageNet, дообучение в 2 стадии), 4 класса: fist, peace, stop, thumbs_up.
Обучение: реальные кадры + небольшая доля синтетических кадров (видео сгенерировано моделью Wan2.1 T2V 1.3B) + аугментация.
Метрики, абляция и разбор ошибок приведены в отчёте.

### Ограничения
- Малый датасет: 24 короткие видео, руки небольшого числа людей, ограниченное разнообразие тона кожи и фонов.
- Точность на новых окружениях (тёмная сцена, фон с яркими огнями, другой человек) заметно ниже, чем на тесте из знакомых условий.
- Синтетические кадры составляют малую долю обучающей выборки и могут быть смещены по цвету кожи.
- Не используйте для задач, где ошибка критична.

### Приватность и маркировка
- Приложение намеренно не сохраняет загруженные изображения (Gradio может временно кэшировать загрузки на сервере).
- На вкладке «Синтетические примеры» все изображения подписаны как сгенерированные ИИ.
- Для съёмки используйте только свои руки или руки людей, давших согласие; лица в кадре не нужны.
"""

with gr.Blocks(title="Hand gesture recognition") as demo:
    gr.Markdown("# Распознавание жестов: fist / peace / stop / thumbs_up\n"
                "Загрузите фото или сделайте снимок с камеры. Приложение покажет класс, "
                "карту Grad-CAM, время инференса и среду.")
    with gr.Tab("Распознавание"):
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", sources=["webcam", "upload"], label="Снимок или загрузка")
                with gr.Accordion("Стресс-тест: искусственные искажения (не реальные условия)", open=False):
                    s_br = gr.Slider(0.1, 2.0, value=1.0, step=0.05, label="Яркость (1 = без изменений)")
                    s_bl = gr.Slider(0, 8, value=0, step=0.5, label="Размытие (радиус)")
                    s_ro = gr.Slider(-90, 90, value=0, step=5, label="Поворот, градусы")
                    s_no = gr.Slider(0, 50, value=0, step=1, label="Шум (сигма)")
                btn = gr.Button("Распознать", variant="primary")
            with gr.Column():
                lab_deg = gr.Label(num_top_classes=4, label="Предсказание (с учётом искажений)")
                lab_orig = gr.Label(num_top_classes=4, label="Предсказание на исходном кадре")
                cam_out = gr.Image(label="Grad-CAM (красное = куда смотрит модель)")
                info = gr.Markdown()
        ins = [inp, s_br, s_bl, s_ro, s_no]
        outs = [lab_deg, lab_orig, cam_out, info]
        btn.click(predict, ins, outs)
        inp.change(predict, ins, outs)
        for s in (s_br, s_bl, s_ro, s_no):
            s.release(predict, ins, outs)
    with gr.Tab("Синтетические примеры"):
        gr.Markdown("**СИНТЕТИКА: видео и изображения сгенерированы ИИ (Wan2.1 T2V 1.3B), "
                    "это не фотографии и не записи реальных людей.**")
        if SYNTH_FILES:
            gr.Gallery([(f, "СИНТЕТИКА (ИИ): " + os.path.basename(f)) for f in SYNTH_FILES], columns=4)
        for i in range(0, len(SYNTH_VIDEOS), 3):
            with gr.Row():
                for v in SYNTH_VIDEOS[i:i + 3]:
                    gr.Video(value=v, label="СИНТЕТИКА (ИИ): " + os.path.basename(v).replace(" (1)", ""),
                             interactive=False, height=360)
        if not SYNTH_FILES and not SYNTH_VIDEOS:
            gr.Markdown("_Положите файлы в папку assets/synthetic/_")
    with gr.Tab("О модели и ограничения"):
        gr.Markdown(ABOUT)

if __name__ == "__main__":
    demo.launch(share=os.environ.get("SHARE") == "1")
