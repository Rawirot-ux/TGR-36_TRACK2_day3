# TGR-36 · TRACK2 Day 3 — Coffee bean segmentation + class counts on Raspberry Pi, sent over HTTP + MQTT

Classes: `black, broken, good, husk, immature, parchment, shell`
Model: YOLO26n-seg fine-tuned, exported to NCNN fp32 640 · inference uses only `ncnn numpy opencv pillow PyYAML` (no torch / ultralytics on the Pi).

## Files

| Path | What |
|---|---|
| `infer.py` | **Inference (submission)** — fast, never aborts on mask holes, refines mask edges with image colour (default) |
| `client.py` | **Day 3 integration** — GET test images over HTTP → `infer.py` → POST `results.json` + `team_36_Day3.zip` and publish one MQTT message per image, in parallel |
| `client_config.example.json` | Template for organiser endpoints/credentials — copy to `client_config.json` (git-ignored, never committed) |
| `train.py` | Reproduces `weights/best.pt` (2 training stages + NCNN export) |
| `weights/best.pt` | Trained model (stage-2 `last.pt`, md5 `76628505935e23a38a835a87faf26ed5`) |
| `weights/best_ncnn_model/` | NCNN export used by inference (`model.ncnn.param/.bin`, `metadata.yaml`, imgsz 640) |
| `requirements.txt` | Inference + client dependencies (Pi) |

## Install (Raspberry Pi, 64-bit OS, Python 3.11+)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run inference

```bash
python infer.py --input test/ --weights weights/ --out outputs/
```

- `--input`: image file or folder (searched recursively; `.jpg .jpeg .png`, EXIF orientation applied)
- `--weights`: folder containing exactly one NCNN export (`*.ncnn.param` + `.bin` + `metadata.yaml`)
- `--out`: output folder; may already exist — only `seg_mask/`, `vis/`, `mqtt/`, `predictions.json` are replaced
- Exit code 1 with an `Error:` message if input/weights are missing, no images, an image cannot be read, or output fails validation.

Outputs:
- `seg_mask/<image stem>.png` — single-channel **uint16**, same size as the image (after EXIF), 0 = background, 1..N = bean id
- `predictions.json` — one entry per image: `{"image_name", "objects": [{"id", "class", "polygon": [[x, y], ...]}]}`; `id` matches `seg_mask`, polygon = `cv2.findContours(..., CHAIN_APPROX_NONE)` of that mask, full-image pixels
- `vis/<image stem>.jpg` — overlay for review (half size)
- `mqtt/<image stem>.json` — the MQTT message for that image: `{"image", "latency_ms", "counts": {good, shell, black, husk, broken, parchment, immature}, "defect_ratio"}`; all 7 keys always present; counts come from the same objects as `predictions.json`/`seg_mask`; `defect_ratio` = non-`good` beans ÷ all beans (0.0 if none); `latency_ms` = that image's processing time (read → write, model load excluded)

Defaults (no extra arguments needed): conf 0.25 · class-agnostic NMS IoU 0.7 · `--refine color --band 20 --band-method dt` (re-decides a ±20 px band around each mask edge by Lab distance from the paper colour, excluding grey shadow) · 4 threads.
`--refine none` gives the raw model mask (blocky at 640 and includes shadow).

## Run the Day 3 client (HTTP + MQTT)

```bash
cp client_config.example.json client_config.json   # fill in organiser URLs / broker / topic / user
export MQTT_PASSWORD='...'                          # or put it in client_config.json (git-ignored)
python client.py --config client_config.json --infer infer.py --weights weights/ --out outputs/ --repo .
```

1. GET `get_url` → images saved to `input/` (zip, single image, or JSON list of URLs are accepted)
2. Runs `python infer.py --input input/ --weights weights/ --out outputs/`, then checks that every image has a `predictions.json` entry and an `mqtt/` file with matching counts
3. In parallel: **HTTP** POST multipart `results.json` (= `outputs/predictions.json`) + `team_36_Day3.zip` (this repo incl. `outputs/`, without `.git`, `input/`, configs) · **MQTT** publish each `outputs/mqtt/*.json` to `mqtt_topic` (QoS 1, waits for broker ack)
4. Each step retries 3×; exit code 1 if HTTP or any MQTT message failed (the other task still completes)

Options: `--local-input <folder>` skip GET · `--skip-infer` resend existing outputs · `--only http|mqtt` · `--retries` · `--timeout`.

## Pipeline (per image)

1. Read + EXIF transpose → letterbox to 640 (ultralytics-compatible) → NCNN (`in0` → `out0` 43×8400, `out1` 32×160×160)
2. Confidence filter → class-agnostic NMS
3. For each bean (highest confidence first): mask logits computed **only inside its box** at full resolution (bilinear) → colour edge refinement → pixels already owned by another bean removed → keep largest component, fill holes → write id
4. Write `seg_mask` PNG, polygon from the same mask, `vis` JPEG; `predictions.json` is read back and checked.

## Train (Mac/GPU, not the Pi)

Requires `ultralytics==8.4.208` (torch 2.15.0). Data = YOLO-seg polygons, 7 classes in the order above.

```bash
python train.py --data1 <stage1>/data.yaml --data2 <stage2>/data.yaml --out runs_train --start yolo26n-seg.pt --device mps --export-to weights_new
```

| Stage | Start | Data | imgsz | Epochs |
|---|---|---|---|---|
| 1 | `yolo26n-seg.pt` | 56 synthetic piles, separated beans (from the 14 circle/square training photos) | 1024 | 100 |
| 2 | stage-1 `last.pt` | 140 synthetic piles, touching beans + shadow cut (same photos) | 640 | 40 |

Both: batch 8, AdamW lr0 0.001 lrf 0.01, seed 0 deterministic, patience 0. Validation points at the training images (too few labelled photos), so the fixed final epoch (`last.pt`) is used.
Bean masks: each bean is cut from the organiser photos by colour (Lab distance from the paper + Otsu, shadow removed) and pasted into synthetic piles, so the pile generator knows every bean's exact mask (labels checked by eye). Piles for training use only the circle/square photos; the grid photos are kept for testing. The dataset-building scripts live in our workspace (`toolkit/make_gap_piles.py`, `toolkit/piles_to_yolo.py`), not in this repo.

## Results (our test set: 70 synthetic touching piles from the held-out grid photos, 1,971 beans)

| Inference | mean IoU | counting | beans IoU ≥ 0.95 | Pi 5 median s/image |
|---|---|---|---|---|
| refine none | 0.77 | 0.98 | 0 % | 0.78 |
| **refine color band 20 (default)** | **0.935** | **0.987** | **58 %** | **0.90** |

Pi 5 timing: 7 piles of 3072×4096, 4 threads, includes reading and writing all outputs, excludes model loading.
Caveat: our ground truth was also built with colour rules, so the IoU gain from refinement may be smaller on hand-checked masks.
