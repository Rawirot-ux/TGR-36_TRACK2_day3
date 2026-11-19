#!/usr/bin/env python3
"""client.py — Day 3 (ตามเอกสาร API ผู้จัด day3_brief/API_DOCS.md): GET ภาพทีละภาพ → ทาย → POST ผลทันที → MQTT

ลำดับ (ต่อ 1 ภาพ):
  1. GET  {http_url}/api/v1/image?mode=<train|test>   (Bearer = รหัสล็อกอินเว็บของทีม) → ไบต์ภาพ + X-Image-ID / X-Image-Name
  2. decode ในหน่วยความจำ (exif_transpose) → inf.predict (ฟังก์ชันจาก infer.py ตัวเดียวกับงานส่ง)
  3. POST {http_url}/api/v1/submit  JSON {"img_id", "predictions": {"image_name", "objects"}}  **ก่อน MQTT**
     → เซิร์ฟเวอร์ตอบ duration_seconds = latency ที่ผู้จัดใช้ (ภาพส่งเสร็จ → ได้ JSON)
  4. MQTT publish (WebSocket TLS 443 path /mqtt/, MQTTv5, ต่อครั้งเดียวก่อนเริ่ม) แล้วรอ ack
  5. mqtt/<stem>.json + seg_mask/vis (เธรดเบื้องหลัง) · จบ: predictions.json + ตรวจ counts + client_times.csv

ก่อน GET ภาพแรก: โหลดโมเดล + อุ่นเครื่อง + ต่อ MQTT (ต่อไม่ได้ = หยุด ไม่เสียโควตา test) + GET /api/v1/stats
สถานะ HTTP: 401/403/503 = หยุด · 409 (img_id ซ้ำ) = ข้ามภาพนั้น · 408 (ส่งช้า) = จดแล้วไปต่อ · มีภาพไหนไม่ผ่าน → exit 1

ค่าทีม/URL อยู่ใน --config (ดู client_config.example.json) · รหัส: env HTTP_TOKEN / MQTT_KEY หรือช่อง http_token /
mqtt_key ในไฟล์ config (client_config.json อยู่ใน .gitignore — repo เป็น public ห้าม commit) · สคริปต์ไม่พิมพ์รหัส

ใช้:
  python client.py --config client_config.json --infer infer.py --weights weights/ --out outputs/ --count 3
  python client.py ... --api-mode test --count N      # โหมด test (กินโควตา) ต้องใส่เอง
  python client.py ... --skip-mqtt                     # ทดสอบ HTTP อย่างเดียว
  python client.py ... --save-input ~/day3_rehearsal/r1  # เก็บภาพที่ได้ไว้ดูทีหลัง (ไม่กระทบ duration_seconds)
"""
import argparse
import csv
import importlib.util
import io
import json
import os
import shlex
import ssl
import statistics
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

TRACKS = {"Embedded", "EdgeAI", "Market"}
STOP_STATUS = {401: "รหัส HTTP (Bearer) ผิด", 403: "โควตาเต็ม หรือโหมดนี้ยังไม่เปิด", 503: "ผู้จัดหยุดระบบชั่วคราว"}
CSV_COLS = ["i", "image_name", "img_id", "get_s", "decode_s", "predict_s", "post_s", "http_status",
            "duration_seconds", "mqtt_s", "mqtt_ok", "n_objects"]


def log(tag, msg):
    print(f"[{time.strftime('%H:%M:%S')}] [{tag}] {msg}", flush=True)


def load_infer(path):
    spec = importlib.util.spec_from_file_location("infer_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_outputs(out, n_images):
    preds = json.loads((out / "predictions.json").read_text(encoding="utf-8"))
    msgs = sorted((out / "mqtt").glob("*.json"))
    if len(preds) != n_images or len(msgs) != n_images:
        raise SystemExit(f"Error: ภาพ {n_images} แต่ predictions {len(preds)} / mqtt {len(msgs)}")
    by_name = {p["image_name"]: p for p in preds}
    for m in msgs:
        d = json.loads(m.read_text(encoding="utf-8"))
        cnt = {}
        for o in by_name[d["image"]]["objects"]:
            cnt[o["class"]] = cnt.get(o["class"], 0) + 1
        if len(d["counts"]) != 7 or any(d["counts"][k] != cnt.get(k, 0) for k in d["counts"]):
            raise SystemExit(f"Error: counts ใน {m.name} ไม่ตรง predictions.json")
    log("CHECK", f"predictions {len(preds)} ภาพ = mqtt {len(msgs)} ไฟล์ · counts ตรงกันทุกภาพ")


def load_config(path, skip_mqtt):
    cfg = json.loads(path.read_text(encoding="utf-8"))
    cfg["http_token"] = os.environ.get("HTTP_TOKEN") or cfg.get("http_token", "")
    cfg["mqtt_key"] = os.environ.get("MQTT_KEY") or cfg.get("mqtt_key", "")
    need = ["http_url", "http_token"] + ([] if skip_mqtt else ["team_id", "track", "event", "mqtt_host", "mqtt_key"])
    miss = [k for k in need if not str(cfg.get(k, "")).strip() or "<" in str(cfg.get(k, ""))]
    if miss:
        raise SystemExit(f"Error: config ขาดค่า {miss} (รหัสใส่ทาง env HTTP_TOKEN / MQTT_KEY ได้)")
    if not skip_mqtt and cfg["track"] not in TRACKS:
        raise SystemExit(f"Error: track ต้องเป็น {sorted(TRACKS)} (ตัวพิมพ์ใหญ่เล็กตรง)")
    cfg["http_url"] = cfg["http_url"].rstrip("/")
    return cfg


# ---------- MQTT (ค่าการต่อลอกจาก day3_brief/client_test.py ของผู้จัด) ----------
class Mqtt:
    def __init__(self, cfg, timeout):
        self.cfg, self.timeout = cfg, timeout
        self.topic = f"/event/{cfg['mqtt_key']}/{cfg['track']}/{cfg['event']}"
        self.client = None

    def connect(self):
        import paho.mqtt.client as mqtt
        self.mqtt = mqtt
        self.connected, self.lock, self.acked, self.errors = threading.Event(), threading.Lock(), {}, []

        def on_connect(client, userdata, flags, reason_code, properties):
            if reason_code.is_failure:
                self.errors.append(str(reason_code))
            self.connected.set()

        def on_publish(client, userdata, mid, reason_code, properties):
            with self.lock:
                self.acked[mid] = None if not reason_code.is_failure else str(reason_code)

        c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"{self.cfg['team_id']}:python-{uuid.uuid4().hex[:12]}",
                        transport="websockets", protocol=mqtt.MQTTv5, reconnect_on_failure=False)
        c.username_pw_set(self.cfg["team_id"], self.cfg["mqtt_key"])
        c.ws_set_options(path="/mqtt/")
        c.tls_set_context(ssl.create_default_context())
        c.connect_timeout = 10
        c.on_connect, c.on_publish = on_connect, on_publish
        self.client = c
        c.connect(self.cfg["mqtt_host"], int(self.cfg.get("mqtt_port", 443)), keepalive=60)
        c.loop_start()
        if not self.connected.wait(15):
            self.close()
            raise RuntimeError("ต่อ MQTT ไม่ทันใน 15 s (ตรวจ host)")
        if self.errors:
            self.close()
            raise RuntimeError("MQTT ปฏิเสธการล็อกอิน: " + self.errors[0])

    def _publish_once(self, payload):
        r = self.client.publish(self.topic, payload, qos=1, retain=False)
        if r.rc != self.mqtt.MQTT_ERR_SUCCESS:
            raise RuntimeError("publish ไม่ได้: " + self.mqtt.error_string(r.rc))
        end = time.monotonic() + self.timeout
        while time.monotonic() < end:
            with self.lock:
                if r.mid in self.acked:
                    err = self.acked.pop(r.mid)
                    if err:
                        raise RuntimeError("broker ปฏิเสธข้อความ: " + err)
                    return
            time.sleep(0.005)
        raise RuntimeError(f"ไม่ได้ ack ใน {self.timeout:.0f} s")

    def publish(self, payload):
        """คืน (ok, ข้อความ) · ล้มครั้งแรก → ต่อใหม่ 1 ครั้งแล้วส่งซ้ำ"""
        if len(payload) > 65536:
            return False, "payload เกิน 64 KiB"
        try:
            self._publish_once(payload)
            return True, ""
        except Exception as e:  # noqa: BLE001
            log("MQTT", f"ส่งไม่ผ่าน ({e}) — ต่อใหม่แล้วส่งซ้ำ 1 ครั้ง")
        try:
            self.close()
            self.connect()
            self._publish_once(payload)
            return True, "ส่งซ้ำผ่าน"
        except Exception as e:  # noqa: BLE001
            return False, str(e)

    def close(self):
        if self.client is not None:
            try:
                self.client.disconnect()
                self.client.loop_stop()
            except Exception:  # noqa: BLE001
                pass
            self.client = None


# ---------- HTTP ----------
def get_stats(s, cfg, timeout):
    try:
        r = s.get(f"{cfg['http_url']}/api/v1/stats", timeout=timeout)
        if r.status_code == 401:
            raise SystemExit("Error: /stats ตอบ 401 รหัส HTTP (Bearer) ผิด — หยุดก่อน GET ภาพ")
        if r.status_code != 200:
            log("STATS", f"HTTP {r.status_code} (ไม่หยุด)")
            return None
        d = r.json()
        log("STATS", f"train {d.get('train_mode')} · test {d.get('test_mode')}")
        return d
    except Exception as e:  # noqa: BLE001
        log("STATS", f"อ่านไม่ได้ ({e}) (ไม่หยุด)")
        return None


def post_submit(s, cfg, body, timeout):
    """คืน (status, json|None) · ต่อเซิร์ฟเวอร์ไม่ได้ → ลองซ้ำ 1 ครั้ง (ถ้าได้ 409 ตอนซ้ำ = ครั้งแรกอาจบันทึกแล้ว)"""
    for attempt in (1, 2):
        try:
            r = s.post(f"{cfg['http_url']}/api/v1/submit", json=body, timeout=timeout)
            try:
                d = r.json()
            except ValueError:
                d = {"raw": r.text[:200]}
            return r.status_code, d
        except requests.RequestException as e:
            log("POST", f"ส่งไม่ได้ ({type(e).__name__}) ครั้งที่ {attempt}")
    return 0, None


def run(cfg, a):
    import cv2
    import numpy as np
    from PIL import Image, ImageOps

    inf = load_infer(a.infer)
    ia = inf.build_parser().parse_args(["--input", ".", "--weights", str(a.weights), "--out", str(a.out)]
                                       + shlex.split(a.infer_extra))
    cv2.setNumThreads(ia.threads)
    net, imgsz = inf.load_model(ia.weights, ia.threads)
    inf.run_net(net, np.full((imgsz[0], imgsz[1], 3), 114, np.uint8))  # warm-up
    a.out.mkdir(parents=True, exist_ok=True)
    inf.clear_out(a.out)
    for d in ("seg_mask", "vis", "mqtt"):
        (a.out / d).mkdir()

    mq = None
    if not a.skip_mqtt:
        mq = Mqtt(cfg, a.timeout)
        try:
            mq.connect()
        except Exception as e:  # noqa: BLE001
            raise SystemExit(f"Error: MQTT ต่อไม่ได้ ({e}) — หยุดก่อน GET (ใช้ --skip-mqtt ถ้าจะทดสอบ HTTP อย่างเดียว)")
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {cfg['http_token']}"
    stats = get_stats(s, cfg, a.timeout)
    if a.api_mode == "test" and stats and isinstance(stats.get("test_mode"), dict):
        left = stats["test_mode"].get("remaining_attempts")
        if isinstance(left, int) and left < a.count:
            log("STATS", f"เตือน: โควตา test เหลือ {left} < --count {a.count}")
    log("READY", f"โมเดล imgsz {imgsz} · MQTT {'พร้อม' if mq else 'ข้าม (--skip-mqtt)'} · "
                 f"โหมด {a.api_mode} · {a.count} ภาพ")

    pool = ThreadPoolExecutor(max_workers=1)
    pending, preds, rows, failed, stop = [], {}, [], 0, ""
    for i in range(1, a.count + 1):
        row = {k: "" for k in CSV_COLS}
        row["i"] = i
        t = time.perf_counter()
        try:
            r = s.get(f"{cfg['http_url']}/api/v1/image", params={"mode": a.api_mode}, timeout=a.timeout)
        except requests.RequestException as e:
            stop = f"GET ไม่ได้ ({type(e).__name__})"
            break
        row["get_s"] = round(time.perf_counter() - t, 3)
        if r.status_code != 200:
            stop = f"GET HTTP {r.status_code} {STOP_STATUS.get(r.status_code, r.text[:120])}"
            break
        img_id, name = r.headers.get("X-Image-ID", ""), r.headers.get("X-Image-Name", "")
        if not img_id or not name:
            stop = "GET ไม่มี header X-Image-ID / X-Image-Name"
            break
        name = Path(name).name  # กันชื่อมี path
        row.update(image_name=name, img_id=img_id)

        t0 = time.perf_counter()
        rgb, objects, inst = None, [], None
        try:
            with Image.open(io.BytesIO(r.content)) as im:
                rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
        except Exception as e:  # noqa: BLE001
            log("IMG", f"{name}: decode ไม่ได้ ({e}) — ส่ง objects ว่าง")
        t1 = time.perf_counter()
        if rgb is not None:
            inst, objects = inf.predict(net, imgsz, rgb, ia)
        t2 = time.perf_counter()
        row.update(decode_s=round(t1 - t0, 3), predict_s=round(t2 - t1, 3), n_objects=len(objects))

        status, resp = post_submit(s, cfg, {"img_id": img_id, "predictions": {"image_name": name, "objects": objects}},
                                   a.timeout)
        row["post_s"] = round(time.perf_counter() - t2, 3)
        row["http_status"] = status
        if status == 200 and isinstance(resp, dict):
            row["duration_seconds"] = resp.get("duration_seconds", "")
        else:
            failed += 1

        msg = inf.mqtt_message(name, objects, t2 - t0)
        payload = json.dumps(msg, ensure_ascii=False)
        if mq:
            t3 = time.perf_counter()
            ok, why = mq.publish(payload.encode("utf-8"))
            row.update(mqtt_s=round(time.perf_counter() - t3, 3), mqtt_ok=int(ok))
            if not ok:
                failed += 1
                log("MQTT", f"{name}: ไม่ผ่าน ({why})")
        stem = Path(name).stem
        (a.out / "mqtt" / f"{stem}.json").write_text(payload, encoding="utf-8")
        if a.save_input:  # หลัง POST + MQTT แล้ว → ไม่กระทบ duration_seconds · ชื่อซ้ำ → ต่อท้าย img_id ไม่เขียนทับ
            dst = a.save_input / name
            if dst.exists():
                dst = a.save_input / f"{stem}__{img_id}{Path(name).suffix}"
            pending.append(pool.submit(dst.write_bytes, r.content))
        if inst is not None:
            pending.append(pool.submit(inf.write_outputs, a.out, stem, rgb, inst, ia.vis_scale))
        if name in preds:
            log("IMG", f"{name}: ชื่อซ้ำภาพก่อนหน้า — ไฟล์ในเครื่องเก็บผลครั้งล่าสุด")
        preds[name] = {"image_name": name, "objects": objects}
        rows.append(row)
        note = {408: " ส่งช้าเกินกำหนด", 409: " img_id ซ้ำ — ข้าม"}.get(status, "")
        if status not in (200, 408, 409):
            note = f" {STOP_STATUS.get(status, '') or (str(resp)[:120] if resp else 'ต่อไม่ได้')}"
        log("IMG", f"{i}/{a.count} {name}: {len(objects)} เม็ด · GET {row['get_s']} decode {row['decode_s']} "
                   f"ทาย {row['predict_s']} POST {row['post_s']} s · HTTP {status}{note} · "
                   f"duration_seconds {row['duration_seconds'] or '-'}"
                   + (f" · MQTT {row['mqtt_s']} s {'ok' if row['mqtt_ok'] else 'ไม่ผ่าน'}" if mq else ""))
        if status in STOP_STATUS:
            stop = f"POST HTTP {status} {STOP_STATUS[status]}"
            break

    for p in pending:
        p.result()
    pool.shutdown(wait=True)
    if mq:
        mq.close()
    if rows:
        with (a.out / "client_times.csv").open("w", newline="", encoding="utf-8") as fp:
            w = csv.DictWriter(fp, fieldnames=CSV_COLS)
            w.writeheader()
            w.writerows(rows)
    if preds:
        plist = sorted(preds.values(), key=lambda p: p["image_name"].casefold())  # ลำดับเดียวกับ infer.py
        inf.write_predictions(a.out / "predictions.json", plist)
        check_outputs(a.out, len(plist))
    durs = [float(r["duration_seconds"]) for r in rows if r["duration_seconds"] != ""]
    if durs:
        log("SUMMARY", f"duration_seconds (นิยามผู้จัด) {len(durs)} ภาพ: median {statistics.median(durs):.3f} · "
                       f"max {max(durs):.3f} · min {min(durs):.3f}")
    log("SUMMARY", f"ส่ง {len(rows)}/{a.count} ภาพ · ไม่ผ่าน {failed}"
                   + (f" · ผลเวลา {a.out / 'client_times.csv'}" if rows else ""))
    if stop:
        log("STOP", stop)
    return not stop and failed == 0 and len(rows) == a.count


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--infer", required=True, type=Path, help="infer.py ที่ใช้")
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="โฟลเดอร์ผล (เช่น outputs/)")
    ap.add_argument("--count", required=True, type=int, help="จำนวนภาพที่จะ GET")
    ap.add_argument("--api-mode", choices=["train", "test"], default="train",
                    help="train = ไม่จำกัด · test = กินโควตา ต้องใส่เอง")
    ap.add_argument("--skip-mqtt", action="store_true", help="ไม่ต่อ/ไม่ส่ง MQTT (ทดสอบ HTTP)")
    ap.add_argument("--save-input", type=Path, default=None,
                    help="เก็บไบต์ภาพที่ GET มาลงโฟลเดอร์นี้ (หลัง POST/MQTT) · ค่าเริ่ม: ไม่เก็บ")
    ap.add_argument("--timeout", type=float, default=30, help="วินาที ต่อคำขอ HTTP / รอ ack MQTT")
    ap.add_argument("--infer-extra", default="", help='argument เพิ่มให้ infer ในเครื่องหมายคำพูด เช่น "--conf 0.3"')
    a = ap.parse_args()
    if a.count < 1:
        raise SystemExit("Error: --count ต้อง ≥ 1")
    cfg = load_config(a.config, a.skip_mqtt)
    a.out = a.out.resolve()
    if a.save_input:
        a.save_input = a.save_input.resolve()
        if a.save_input == a.out or a.out in a.save_input.parents:
            raise SystemExit("Error: --save-input ต้องอยู่นอก --out (--out ถูกล้างทุกรอบ)")
        a.save_input.mkdir(parents=True, exist_ok=True)
    ok = run(cfg, a)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
