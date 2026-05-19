"""
Envision 2026 - D08  |  Laptop Inference Script
================================================
Just run:  python envision_laptop.py
The script will ask you everything it needs interactively — no flags needed.

Keyboard shortcuts during video playback
-----------------------------------------
  Space  -> pause / resume
  U      -> run UNet on current frame and save it
  Q      -> quit

FIX APPLIED
-----------
preprocess_unet(): added cv2.COLOR_BGR2RGB conversion before normalisation.

The training pipeline (load_sample in the Kaggle notebook) does:
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # <-- explicit RGB conversion
    img = cv2.resize(img, (256, 256))
    img = tf.cast(img, tf.float32) / 255.0

The original inference code skipped the BGR->RGB step, so the model received
channels in the wrong order (B and R swapped). Segmentation is highly
colour-sensitive, which is why the output was patchy and wrong even though
the isolated TFLite validation (done with correctly-ordered frames) looked fine.

No other changes were needed:
  - Input size 256x256 confirmed from unet_model(input_size=(256, 256, 3))
  - Normalisation is /255.0 only (no ImageNet mean/std; not used in training)
  - Output shape is (1, 256, 256, 124) channels-last, confirmed from
    Conv2D(n_classes, (1,1), activation='softmax') as the final layer
  - argmax on axis=-1 is therefore correct
  - Quantisation handling (scale/zero_point) was already correct
"""

import os
import sys
import shutil
import time
import threading
import numpy as np
import cv2

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    try:
        from tensorflow.lite.python.interpreter import Interpreter
    except ImportError:
        print("\n[!] Neither tflite-runtime nor tensorflow is installed.")
        print("    Fix:  pip install tensorflow\n")
        sys.exit(1)


# =============================================================================
# CONSTANTS
# =============================================================================
DEFAULT_YOLO_MODEL  = "models/yolo_int8.tflite"
DEFAULT_UNET_MODEL  = "models/unet_mapillary_int8.tflite"
DEFAULT_OUTPUT_DIR  = "output"
CONF_THRESH         = 0.25
IOU_THRESH          = 0.45
UNET_CLASSES        = 124
NUM_THREADS         = 4

UNET_SAMPLE_FRAMES  = 10
YOLO_SAMPLE_FRAMES  = 15

INFER_WIDTH         = 640
INFER_EVERY_N       = 30


# =============================================================================
# COLOUR PALETTE
# =============================================================================
def _make_palette(n):
    palette = []
    for i in range(n):
        h   = int(180 * i / n)
        c   = np.uint8([[[h, 210, 210]]])
        bgr = cv2.cvtColor(c, cv2.COLOR_HSV2BGR)[0][0]
        palette.append(tuple(int(x) for x in bgr))
    return palette

SEG_PALETTE = _make_palette(UNET_CLASSES)


# =============================================================================
# INTERACTIVE SETUP
# =============================================================================
def _ask(prompt, valid=None, default=None):
    while True:
        suffix = f" [{default}]" if default is not None else ""
        ans = input(f"{prompt}{suffix}: ").strip()
        if ans == "" and default is not None:
            return str(default)
        if valid is None or ans.lower() in valid:
            return ans
        print(f"    Please enter one of: {', '.join(valid)}")


def interactive_setup():
    SEP = "-" * 58

    print()
    print("=" * 58)
    print("      Envision 2026 - D08  Inference Pipeline")
    print("=" * 58)

    print()
    print(SEP)
    print("  INPUT SOURCE")
    print("  [1]  Webcam  (built-in or USB camera, live feed)")
    print("  [2]  Video file stored on this device")
    print(SEP)
    choice     = _ask("  Choose", valid=["1", "2"])
    use_webcam = choice == "1"

    cam_index  = 0
    video_path = ""

    if use_webcam:
        print()
        cam_index = int(_ask("  Camera index  (0 = built-in, 1 = USB, ...)", default=0))
    else:
        print()
        while True:
            video_path = _ask("  Full path to video file")
            video_path = video_path.strip('"').strip("'")
            if os.path.isfile(video_path):
                break
            print(f"    [!] File not found: {video_path!r}  -- please try again.")

    print()
    print(SEP)
    print("  MODEL PATHS  (press Enter to use the default)")
    print(SEP)
    yolo_model = _ask("  YOLO .tflite", default=DEFAULT_YOLO_MODEL).strip('"').strip("'")
    unet_model = _ask("  UNet .tflite", default=DEFAULT_UNET_MODEL).strip('"').strip("'")

    for label, path in [("YOLO", yolo_model), ("UNet", unet_model)]:
        if not os.path.isfile(path):
            print(f"\n  [!] {label} model not found: {path!r}")
            sys.exit(1)

    print()
    print(SEP)
    print("  OUTPUT FOLDER")
    print(SEP)
    output_dir = _ask("  Root output folder", default=DEFAULT_OUTPUT_DIR).strip('"').strip("'")

    yolo_dir = os.path.join(output_dir, "yolo_output")
    unet_dir = os.path.join(output_dir, "unet_output")

    for d, label in [(yolo_dir, "yolo_output"), (unet_dir, "unet_output")]:
        if os.path.exists(d):
            print(f"  [~] '{d}' already exists -- clearing it ...")
            shutil.rmtree(d)
        os.makedirs(d)
        print(f"  [+] Created '{d}'")

    print()
    print(SEP)
    print("  SUMMARY")
    print(SEP)
    if use_webcam:
        print(f"  Source     : webcam  (index {cam_index})")
    else:
        print(f"  Source     : {video_path}")
        print(f"  YOLO save  : {YOLO_SAMPLE_FRAMES} randomly-sampled frames auto-saved")
        print(f"  UNet       : press U during playback to run on any frame")
    print(f"  YOLO model : {yolo_model}")
    print(f"  UNet model : {unet_model}")
    print(f"  YOLO out   : {yolo_dir}")
    print(f"  UNet out   : {unet_dir}")
    print(SEP)
    input("\n  Press Enter to load models and start ...\n")

    class Cfg: pass
    cfg            = Cfg()
    cfg.use_webcam = use_webcam
    cfg.cam_index  = cam_index
    cfg.video_path = video_path
    cfg.yolo_model = yolo_model
    cfg.unet_model = unet_model
    cfg.output_dir = output_dir
    cfg.yolo_dir   = yolo_dir
    cfg.unet_dir   = unet_dir
    return cfg


# =============================================================================
# MODEL LOADING
# =============================================================================
def load_model(path):
    interp = Interpreter(model_path=path, num_threads=NUM_THREADS)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    print(f"  [OK] {os.path.basename(path)}")
    print(f"         input  {list(inp['shape'])}  {inp['dtype'].__name__}")
    print(f"         output {list(out['shape'])}  {out['dtype'].__name__}")
    return interp, inp, out


# =============================================================================
# PRE-PROCESSING
# =============================================================================
def _downscale(frame):
    h, w = frame.shape[:2]
    if w <= INFER_WIDTH:
        return frame
    scale = INFER_WIDTH / w
    return cv2.resize(frame, (INFER_WIDTH, int(h * scale)),
                      interpolation=cv2.INTER_LINEAR)


def preprocess_yolo(frame):
    img = cv2.resize(frame, (640, 640))
    img = img.astype(np.float32) / 255.0
    return img[np.newaxis, ...]


def preprocess_unet(frame, unet_inp):
    """
    Pre-process a BGR OpenCV frame for the INT8 UNet model.

    Mirrors the training pipeline exactly (load_sample in the Kaggle notebook):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # training does this
        img = cv2.resize(img, (256, 256))
        img = tf.cast(img, tf.float32) / 255.0        # /255 only, no mean/std

    THE FIX: cv2.COLOR_BGR2RGB was present during training but absent here.
    The model weights encode all colour statistics in RGB channel order.
    Passing BGR input swaps the red and blue channels, corrupting every
    feature map from the very first convolution and producing the patchy /
    wrong segmentation seen in the live output even though the isolated
    TFLite validation used correctly-ordered RGB frames and looked fine.
    """
    img = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LINEAR)

    # THE FIX: BGR (OpenCV default) -> RGB (matches training pipeline)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32) / 255.0      # /255 only — matches training

    scale, zero_point = unet_inp['quantization']
    if scale > 0:
        img = (img / scale + zero_point).astype(np.int8)
    else:
        img = img.astype(np.int8)

    return img[np.newaxis, ...]               # (1, 256, 256, 3)


# =============================================================================
# YOLO INFERENCE
# =============================================================================
def _run_yolo(frame, yolo_interp, yolo_inp, yolo_out):
    h, w = frame.shape[:2]
    t0 = time.perf_counter()
    yolo_interp.set_tensor(yolo_inp["index"], preprocess_yolo(frame))
    yolo_interp.invoke()
    raw = yolo_interp.get_tensor(yolo_out["index"])
    scale, zero_point = yolo_out["quantization"]
    if scale > 0:
        raw = (raw.astype(np.float32) - zero_point) * scale
    detections = postprocess_yolo(raw, 640, 640)
    ms = (time.perf_counter() - t0) * 1000
    sx, sy = w / 640.0, h / 640.0
    return [(int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy), conf, cls)
            for x1, y1, x2, y2, conf, cls in detections], ms


# =============================================================================
# YOLO POST-PROCESSING
# =============================================================================
def _xywh2xyxy(b):
    o = b.copy()
    o[..., 0] = b[..., 0] - b[..., 2] / 2
    o[..., 1] = b[..., 1] - b[..., 3] / 2
    o[..., 2] = b[..., 0] + b[..., 2] / 2
    o[..., 3] = b[..., 1] + b[..., 3] / 2
    return o


def postprocess_yolo(raw, h, w):
    preds      = raw[0].T
    boxes      = preds[:, :4]
    cls_scores = preds[:, 4:]

    conf      = np.max(cls_scores, axis=1)
    class_ids = np.argmax(cls_scores, axis=1)

    mask = conf > CONF_THRESH
    boxes, conf, class_ids = boxes[mask], conf[mask], class_ids[mask]
    if len(boxes) == 0:
        return []

    xyxy = _xywh2xyxy(boxes)
    xyxy[:, [0, 2]] *= w
    xyxy[:, [1, 3]] *= h

    results = []
    for cls in np.unique(class_ids):
        idx = class_ids == cls
        b   = xyxy[idx].tolist()
        s   = conf[idx].tolist()
        nms = cv2.dnn.NMSBoxes(b, s, CONF_THRESH, IOU_THRESH)
        for i in (nms.flatten() if len(nms) else []):
            x1, y1, x2, y2 = [int(v) for v in b[i]]
            results.append((x1, y1, x2, y2, round(s[i], 3), int(cls)))
    return results


# =============================================================================
# UNET POST-PROCESSING
# =============================================================================
def postprocess_unet(raw, h, w, unet_out):
    """
    Dequantise INT8 output then build a colour segmentation mask.

    Output shape is (1, 256, 256, 124) — channels-last — confirmed by the
    training model's final layer:
        layers.Conv2D(n_classes, (1, 1), activation='softmax')
    argmax on axis=-1 is therefore correct (unchanged from original).
    """
    scale, zero_point = unet_out['quantization']
    if scale > 0:
        raw = (raw.astype(np.float32) - zero_point) * scale

    seg       = raw[0]                          # (256, 256, 124)
    class_map = np.argmax(seg, axis=-1)         # (256, 256)
    mask      = np.zeros((256, 256, 3), dtype=np.uint8)
    for cls_id, colour in enumerate(SEG_PALETTE):
        mask[class_map == cls_id] = colour
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


# =============================================================================
# RENDER HELPERS
# =============================================================================
def render_yolo(frame, detections, ms, fps=None):
    out = frame.copy()
    h, w = out.shape[:2]
    for x1, y1, x2, y2, conf, cls in detections:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"cls{cls}  {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, max(y1-th-6, 0)),
                      (x1+tw+4, max(y1, th+6)), (0, 255, 0), -1)
        cv2.putText(out, label, (x1+2, max(y1-4, th+2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    tag = f"YOLO | {len(detections)} obj | {ms:.0f} ms"
    cv2.rectangle(out, (0, 0), (len(tag)*9+10, 28), (0, 0, 0), -1)
    cv2.putText(out, tag, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    if fps is not None:
        fps_tag = f"FPS: {fps:.1f}"
        (fw, _), _ = cv2.getTextSize(fps_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        fx = w - fw - 12
        cv2.rectangle(out, (fx - 6, 0), (w, 32), (0, 0, 0), -1)
        cv2.putText(out, fps_tag, (fx, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return out


def render_unet(seg_mask, ms):
    out = seg_mask.copy()
    tag = f"UNet Segmentation | {ms:.0f} ms"
    cv2.rectangle(out, (0, 0), (len(tag)*9+10, 28), (0, 0, 0), -1)
    cv2.putText(out, tag, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return out


# =============================================================================
# INFERENCE FUNCTIONS
# =============================================================================
def infer_yolo_only(frame, yolo_interp, yolo_inp, yolo_out, fps=None):
    detections, ms_yolo = _run_yolo(frame, yolo_interp, yolo_inp, yolo_out)
    return render_yolo(frame, detections, ms_yolo, fps=fps), len(detections), ms_yolo


def infer_unet_only(frame, unet_interp, unet_inp, unet_out):
    h, w = frame.shape[:2]
    t0 = time.perf_counter()
    unet_interp.set_tensor(unet_inp["index"], preprocess_unet(frame, unet_inp))
    unet_interp.invoke()
    raw      = unet_interp.get_tensor(unet_out["index"])
    seg_mask = postprocess_unet(raw, h, w, unet_out)
    ms_unet  = (time.perf_counter() - t0) * 1000
    return render_unet(seg_mask, ms_unet), ms_unet


def infer_frame(frame, yolo_interp, yolo_inp, yolo_out,
                unet_interp, unet_inp, unet_out):
    yolo_img, n_det, ms_y = infer_yolo_only(frame, yolo_interp, yolo_inp, yolo_out)
    unet_img, ms_u        = infer_unet_only(frame, unet_interp, unet_inp, unet_out)
    return yolo_img, unet_img, n_det, ms_y, ms_u


# =============================================================================
# SAVE PAIR
# =============================================================================
def save_pair(yolo_img, unet_img, yolo_dir, unet_dir, name):
    yp = os.path.join(yolo_dir, name)
    up = os.path.join(unet_dir, name)
    cv2.imwrite(yp, yolo_img)
    cv2.imwrite(up, unet_img)
    return yp, up


# =============================================================================
# WEBCAM MODE
# =============================================================================
def run_webcam(cfg, yolo_interp, yolo_inp, yolo_out,
               unet_interp, unet_inp, unet_out):

    cap = cv2.VideoCapture(cfg.cam_index)
    if not cap.isOpened():
        print(f"\n[!] Cannot open camera index {cfg.cam_index}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\n[Camera] Opened at {aw}x{ah}")
    print("[Camera]  S = save frame   |   U = run UNet   |   Q = quit\n")

    saved           = 0
    last_detections = []
    last_unet_img   = None
    ms_y = ms_u     = 0.0
    cam_frame_n     = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Camera read failed")
            break

        if cam_frame_n % INFER_EVERY_N == 0:
            last_detections, ms_y = _run_yolo(frame, yolo_interp, yolo_inp, yolo_out)

        yolo_img = render_yolo(frame, last_detections, ms_y)
        cv2.imshow("YOLO -- Detections  (S=save  U=run UNet  Q=quit)", yolo_img)
        if last_unet_img is not None:
            cv2.imshow("UNet -- Segmentation  (U to refresh)", last_unet_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            fname     = f"webcam_{saved:05d}.jpg"
            unet_save = last_unet_img if last_unet_img is not None else np.zeros_like(frame)
            yp, up    = save_pair(yolo_img, unet_save, cfg.yolo_dir, cfg.unet_dir, fname)
            saved += 1
            print(f"\n[OK] Saved  YOLO -> {yp}")
        elif key == ord('u'):
            print("\n  [UNet] running ...")
            last_unet_img, ms_u = infer_unet_only(
                frame, unet_interp, unet_inp, unet_out)
            print(f"  [UNet] done ({ms_u:.0f} ms)")
        elif key == ord('q'):
            break

        cam_frame_n += 1
        sys.stdout.write(
            f"\r  YOLO {ms_y:5.0f}ms  {len(last_detections):2d} obj  |  "
            f"UNet {ms_u:5.0f}ms  saved={saved}   ")
        sys.stdout.flush()

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[*] Done. {saved} frame pair(s) saved to {cfg.output_dir}/")


# =============================================================================
# VIDEO FILE MODE
# =============================================================================
def run_video(cfg, yolo_interp, yolo_inp, yolo_out,
              unet_interp, unet_inp, unet_out):

    cap = cv2.VideoCapture(cfg.video_path)
    if not cap.isOpened():
        print(f"\n[!] Cannot open video: {cfg.video_path}")
        sys.exit(1)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    vw    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n[Video] {os.path.basename(cfg.video_path)}")
    print(f"        {vw}x{vh}  |  {total} frames  |  {fps:.1f} fps")

    for d in (cfg.yolo_dir, cfg.unet_dir):
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    def _sample_frames(n, total):
        n = min(n, max(1, total))
        if n == 1:
            return {total // 2}
        evenly = [int(round(i * (total - 1) / (n - 1))) for i in range(n)]
        rng    = np.random.default_rng()
        spread = max(1, total // (n * 4))
        return set(
            min(max(0, f + int(rng.integers(-spread, spread + 1))), total - 1)
            for f in evenly
        )

    yolo_frames = _sample_frames(YOLO_SAMPLE_FRAMES, total)

    print(f"        YOLO saves on {len(yolo_frames)} random frames: {sorted(yolo_frames)}")
    print("        Space = pause  |  U = run UNet on current frame  |  Q = quit\n")

    lock         = threading.Lock()
    infer_req    = {"frame": None, "idx": -1}
    infer_result = {"detections": None, "n_det": 0, "ms": 0.0}
    stop_event   = threading.Event()

    def _inference_worker():
        while not stop_event.is_set():
            with lock:
                frame = infer_req["frame"]
                idx   = infer_req["idx"]
            if frame is None:
                time.sleep(0.001)
                continue
            detections, ms = _run_yolo(frame, yolo_interp, yolo_inp, yolo_out)
            with lock:
                infer_result["detections"] = detections
                infer_result["n_det"]      = len(detections)
                infer_result["ms"]         = ms
                infer_req["frame"]         = None

    worker = threading.Thread(target=_inference_worker, daemon=True)
    worker.start()

    frame_idx       = 0
    yolo_saved      = 0
    unet_saved      = 0
    paused          = False
    display_frame   = None
    last_detections = []
    last_ms         = 0.0
    live_fps        = 0.0
    _fps_t          = time.perf_counter()
    _fps_count      = 0
    next_infer_idx  = 0
    frame_delay_ms  = max(1, int(1000.0 / fps))

    while True:
        loop_start = time.perf_counter()

        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\n[*] End of video reached.")
                break

            display_frame = _downscale(frame)

            _fps_count += 1
            _elapsed = time.perf_counter() - _fps_t
            if _elapsed >= 0.5:
                live_fps   = _fps_count / _elapsed
                _fps_count = 0
                _fps_t     = time.perf_counter()

            if frame_idx >= next_infer_idx:
                with lock:
                    if infer_req["frame"] is None:
                        infer_req["frame"] = display_frame.copy()
                        infer_req["idx"]   = frame_idx
                        next_infer_idx     = frame_idx + INFER_EVERY_N

            with lock:
                if infer_result["detections"] is not None:
                    last_detections            = infer_result["detections"]
                    last_ms                    = infer_result["ms"]
                    infer_result["detections"] = None

            if frame_idx in yolo_frames:
                fname      = f"frame_{frame_idx:06d}.jpg"
                save_frame = render_yolo(display_frame, last_detections, last_ms)
                cv2.imwrite(os.path.join(cfg.yolo_dir, fname), save_frame)
                yolo_saved += 1

            pct = frame_idx / max(total, 1) * 100
            sys.stdout.write(
                f"\r  Frame {frame_idx:>6}/{total}  ({pct:5.1f}%)  "
                f"YOLO {last_ms:5.0f}ms  {live_fps:.1f}fps  "
                f"yolo_saved={yolo_saved}  unet_saved={unet_saved}   ")
            sys.stdout.flush()

            frame_idx += 1

        if display_frame is not None:
            show = render_yolo(display_frame, last_detections, last_ms, fps=live_fps)
            cv2.imshow("YOLO -- Live  (Space=pause  U=run UNet  Q=quit)", show)

        elapsed_ms = int((time.perf_counter() - loop_start) * 1000)
        wait_ms    = max(1, frame_delay_ms - elapsed_ms)
        key        = cv2.waitKey(wait_ms) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            print(f"\n  [{'Paused -- press Space to resume' if paused else 'Resumed'}]")
        elif key == ord('u') and display_frame is not None:
            print(f"\n  [UNet] running on frame {frame_idx-1} ...")
            unet_img, ms_u = infer_unet_only(
                display_frame, unet_interp, unet_inp, unet_out)
            fname = f"frame_{frame_idx-1:06d}.jpg"
            cv2.imwrite(os.path.join(cfg.unet_dir, fname), unet_img)
            unet_saved += 1
            print(f"  [UNet] done ({ms_u:.0f} ms) -> saved [{unet_saved}]")
            cv2.imshow("UNet -- Segmentation  (close anytime)", unet_img)

    stop_event.set()
    worker.join(timeout=2)
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[*] Done.")
    print(f"    YOLO frames saved : {yolo_saved}  ->  {cfg.yolo_dir}/")
    print(f"    UNet frames saved : {unet_saved}  ->  {cfg.unet_dir}/")


# =============================================================================
# MAIN
# =============================================================================
def main():
    cfg = interactive_setup()

    print("\n[*] Loading models ...")
    yolo_interp, yolo_inp, yolo_out = load_model(cfg.yolo_model)
    unet_interp, unet_inp, unet_out = load_model(cfg.unet_model)
    print("\n[*] Both models ready. Starting inference ...\n")

    if cfg.use_webcam:
        run_webcam(cfg, yolo_interp, yolo_inp, yolo_out,
                   unet_interp, unet_inp, unet_out)
    else:
        run_video(cfg, yolo_interp, yolo_inp, yolo_out,
                  unet_interp, unet_inp, unet_out)


if __name__ == "__main__":
    main()
