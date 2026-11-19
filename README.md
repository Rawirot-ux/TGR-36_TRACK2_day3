# TGR-36 · TRACK2 Day 3 — Coffee bean segmentation + class counts on Raspberry Pi, sent over HTTP + MQTT

Classes: `black, broken, good, husk, immature, parchment, shell`
Model: YOLO26n-seg fine-tuned, exported to NCNN fp32 640 · inference uses only `ncnn numpy opencv pillow PyYAML` (no torch / ultralytics on the Pi).

## Files

| Path | What |
|---|---|
| `infer.py` | **Inference (submission)** — fast, never aborts on mask holes, refines mask edges with image colour (default) |
| `client.py` | **Day 3 integration** (organiser API docs) — per image: GET `/api/v1/image` → predict → POST `/api/v1/submit` (JSON polygons) → MQTT publish over WebSocket TLS 443; MQTT connected *before* the first GET |
| `client_config.example.json` | Template for organiser endpoints/credentials — copy to `client_config.json` (git-ignored, never committed) |
| `train.py` | Reproduces `weights/best.pt` (2 training stages + NCNN export) |
| `weights/best.pt` | Trained model (Day 3 stage-3 `last.pt` = run E4, md5 `2c3c89bcf78492e6b41b6103c9b20cfd`) |
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
- `predictions.json` — one entry per image: `{"image_name", "objects": [{"id", "class", "polygon": [[x, y], ...]}]}`; `id` matches `seg_mask`, polygon = outer contour of that mask from `cv2.findContours`, full-image pixels, corner points only (`--poly-simple 1`, default: same shape as listing every boundary pixel; `--poly-simple 0` lists every pixel)
- `vis/<image stem>.jpg` — overlay for review (half size) with a `#id class` label next to each bean (`--vis-classes 1`, default; `--vis-classes 0` = no labels)
- `mqtt/<image stem>.json` — the MQTT message for that image: `{"image", "latency_ms", "counts": {good, shell, black, husk, broken, parchment, immature}, "defect_ratio"}`; all 7 keys always present; counts come from the same objects as `predictions.json`/`seg_mask`; `defect_ratio` = non-`good` beans ÷ all beans (0.0 if none); `latency_ms` = that image's processing time (read → write, model load excluded)

Defaults (no extra arguments needed): conf 0.25 · class-agnostic NMS IoU 0.7 · `--refine color --band 20 --band-method dt` (re-decides a ±20 px band around each mask edge by Lab distance from the paper colour, excluding grey shadow) · 4 threads.
`--refine none` gives the raw model mask (blocky at 640 and includes shadow).

## Run the Day 3 client (HTTP + MQTT)

```bash
cp client_config.example.json client_config.json   # http_url, team_id, track (Embedded|EdgeAI|Market), event
export HTTP_TOKEN='...'   # team website login password (HTTP Bearer) — or "http_token" in client_config.json
export MQTT_KEY='...'     # key generated on the website (MQTT password + topic) — or "mqtt_key"
python client.py --config client_config.json --infer infer.py --weights weights/ --out outputs/ --count 3
python client.py --config client_config.json --infer infer.py --weights weights/ --out outputs/ --count N --api-mode test
```

`client_config.json` is git-ignored (this repo is public): never commit credentials. Default `--api-mode train` (unlimited); `test` uses the quota and must be given explicitly.

0. load the NCNN model, warm up, connect MQTT once (`tesa-score.c333m.site:443`, WebSocket TLS, path `/mqtt/`, MQTTv5, user = team id, password = MQTT key) and read `GET /api/v1/stats` — all **before** the first image; if MQTT cannot connect the client stops before using any quota (`--skip-mqtt` = HTTP only)
1. per image (`--count` times): `GET /api/v1/image?mode=…` → `X-Image-ID`, `X-Image-Name` + image bytes, decoded in memory (EXIF-transposed)
2. predict → **POST `/api/v1/submit` immediately** `{"img_id", "predictions": {"image_name", "objects": [{"id","class","polygon"}]}}` — the server's `duration_seconds` (image sent → JSON received) is the organiser's latency
3. MQTT publish to `/event/<MQTT_KEY>/<TRACK>/<EVENT>` `{"image","latency_ms","counts","defect_ratio"}` and wait for the QoS 1 ack (one reconnect + resend on failure)
4. `outputs/mqtt/<stem>.json`; `seg_mask/` and `vis/` are written in a background thread; at the end `outputs/predictions.json`, a check that every MQTT count matches it, and `outputs/client_times.csv` (per-image GET / decode / predict / POST / MQTT seconds, HTTP status, `duration_seconds`) plus its median

HTTP status: 401 / 403 (quota full or mode closed) / 503 = stop · 409 (duplicate `img_id`) = skip that image · 408 (too late) = logged, continue. Exit code 1 if any image failed or the run stopped early.
The prediction functions are imported from `infer.py`, so `outputs/` is byte-identical to running `infer.py` on the same images (except the measured `latency_ms`). If the server sends the same image name twice, both are submitted; local files keep the latest.

Options: `--skip-mqtt` · `--save-input <dir>` (keep the raw downloaded images, written after POST/MQTT, outside `--out`) · `--timeout` (s, per HTTP request / MQTT ack, default 30) · `--infer-extra "--conf 0.3"`.

## Day 3 model (E4)

Stage 3: fine-tuned from the Day 2 model (stage-2 `last.pt`) for 40 epochs at 640 (AdamW lr0 0.001, batch 8, seed 0, `last.pt` used) on 436 images: the 140 Day 2 touching piles + all 16 Day 3 organiser photos + 140 synthetic mixed-class piles (15–35 beans) + 140 dense overlapping piles (35–50 beans, up to 55 % occlusion), all piles made from those 16 photos.
E4 is the E3 recipe with one change, the data: E3 kept 3 Day 3 photos (`Mixed_square2`, `Mixed_grid1`, `Mixed_triangle4`, 23 beans) out of training as a test set; E4 also trains on them (more real beans: 101 → 124). So the first three rows below test the recipe (E3 never saw those beans), not E4. E3 and E1 (E3 without the 140 dense piles) are kept as fallbacks.

| Test set (default inference, IoU+count score /15) | Day 2 model | E1 | E3 | **E4 (submitted)** |
|---|---|---|---|---|
| 3 held-out Day 3 photos, 23 beans | 14.46 | 14.71 | 14.71 (class correct 16/23 vs 10/23) | trained on |
| 30 synthetic mixed-class piles from those 23 beans, 745 beans | 4.96 | 12.28 | 12.68 | trained on |
| 30 dense overlapping piles from those 23 beans, 1,272 beans | 3.19 | 10.11 | 11.23 | trained on |
| 70 touching piles from held-out Day 1 grid photos, 1,971 beans | 14.42 | 14.39 | 14.48 | **14.41** |
| 3 external photos (Amostra 100g v13, CC BY 4.0), 798 beans | — | 5.36 | 7.41 | not measured |

Raspberry Pi, `infer.py` median per image on 3 dense piles (read + infer + write, model load excluded): E1 0.77 s · E3 0.77 s. Second run, same Pi and piles, alternating E3/E4/E3/E4: E3 0.79 / 0.82 s · E4 0.78 / 0.78 s (same network, `model.ncnn.param` identical).

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
