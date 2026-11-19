#!/usr/bin/env python3
"""client.py — Day 3: รับภาพทาง HTTP → ทาย → publish MQTT ทีละภาพทันที → POST results.json + zip ทาง HTTP

ผู้จัด (19 พ.ย.): latency Day 3 = นับตั้งแต่ HTTP ส่งภาพ จน MQTT กลับมา · คะแนนทั้งหมดส่งผ่าน HTTP (ไม่ส่ง = 0)
→ โหมด stream (ค่าเริ่ม):
  0. โหลดโมเดล + อุ่นเครื่อง + ต่อ MQTT ให้พร้อม **ก่อน** GET (ไม่นับเวลานี้เข้า latency)
  1. GET ภาพ (zip / ภาพเดี่ยว / JSON รายการ URL — ถ้าเป็นรายการ จะทายภาพที่โหลดเสร็จก่อนเลย)
  2. แต่ละภาพ: ทาย → publish MQTT ทันที → เขียน mqtt/<stem>.json → seg_mask/vis เขียนในเธรดเบื้องหลัง
  3. ครบทุกภาพ: predictions.json (รูปแบบเดียวกับ infer.py) → ตรวจ counts ตรงกัน → POST results.json + team_XX_Day3.zip
     พร้อมกับรอยืนยัน MQTT (ส่งซ้ำข้อความที่ไม่ผ่าน)
  ใช้ฟังก์ชันจาก infer.py ตัวเดียวกัน → seg_mask/predictions.json/mqtt เหมือนรัน infer.py เองทุกไบต์ (ยกเว้น latency_ms)
โหมด batch (สำรอง): GET ทั้งหมด → รัน infer.py แบบ subprocess → publish + POST ขนานกัน

ค่าที่ผู้จัดประกาศ (URL, ชื่อช่อง form, broker, topic, user/password) อยู่ในไฟล์ --config (JSON)
ไฟล์ config มีรหัสผ่าน → ห้าม commit (repo เป็น public) · รหัสผ่านใส่ทาง env MQTT_PASSWORD / HTTP_TOKEN ได้
ดู client_config.example.json

ใช้:
  python client.py --config client_config.json --infer infer.py --weights weights/ --out outputs/ --repo .
  python client.py ... --local-input test_images/      # ข้าม GET (ทดสอบ)
  python client.py ... --mode batch                    # สำรอง: เรียก infer.py ทั้งชุดแล้วค่อยส่ง
  python client.py ... --mode batch --skip-infer       # ส่งผลที่มีอยู่แล้วใน --out ซ้ำ
"""
import argparse
import importlib.util
import io
import json
import os
import shlex
import subprocess
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests

EXT = {".jpg", ".jpeg", ".png"}
ZIP_SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "input", ".idea", ".vscode"}
ZIP_SKIP_FILES = {".DS_Store"}
T0 = time.perf_counter()


def log(tag, msg):
    print(f"[{time.strftime('%H:%M:%S')} +{time.perf_counter() - T0:6.2f}s] [{tag}] {msg}", flush=True)


def retry(tag, n, fn):
    for i in range(1, n + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            log(tag, f"ครั้งที่ {i}/{n} ล้ม: {e}")
            if i == n:
                raise
            time.sleep(min(2 * i, 5))


def headers(cfg):
    h = dict(cfg.get("http_headers") or {})
    tok = os.environ.get("HTTP_TOKEN") or cfg.get("http_token")
    if tok:
        h.setdefault("Authorization", f"Bearer {tok}")
    return h


def _get(url, hdr, timeout):
    r = requests.get(url, headers=hdr, timeout=timeout)
    r.raise_for_status()
    return r


# ---------- 1. รับภาพ ----------
def fetch_iter(cfg, dest, timeout, retries):
    """GET get_url แล้ว yield path ภาพทีละภาพเมื่อพร้อม · รองรับ zip / ภาพเดี่ยว / JSON รายการ
    (list ของ URL/ชื่อไฟล์ หรือ dict ที่มี key images/files/urls/data) — รูปแบบจริงตามเอกสารผู้จัด"""
    dest.mkdir(parents=True, exist_ok=True)
    hdr, url = headers(cfg), cfg["get_url"]
    r = retry("GET", retries, lambda: _get(url, hdr, timeout))
    ctype = r.headers.get("Content-Type", "").lower()
    log("GET", f"{r.status_code} {ctype or '-'} {len(r.content) / 1e6:.2f} MB")
    if "zip" in ctype or r.content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            names = []
            for m in z.infolist():
                name = Path(m.filename).name
                if not m.is_dir() and Path(name).suffix.lower() in EXT and not name.startswith("."):
                    (dest / name).write_bytes(z.read(m))
                    names.append(name)
        for name in sorted(names, key=str.casefold):
            yield dest / name
        return
    if ctype.startswith("image/"):
        name = Path(urlparse(url).path).name or "image.jpg"
        (dest / name).write_bytes(r.content)
        yield dest / name
        return
    data = r.json()
    if isinstance(data, dict):
        data = next((data[k] for k in ("images", "files", "urls", "data") if k in data), None)
    if not isinstance(data, list):
        raise SystemExit(f"Error: ไม่รู้จักรูปแบบคำตอบ GET: {r.text[:300]}")
    for item in data:
        u = item if isinstance(item, str) else (item.get("url") or item.get("image_url") or item.get("name"))
        name = Path(urlparse(u).path).name if isinstance(item, str) else (item.get("name") or Path(urlparse(u).path).name)
        full = urljoin(url if url.endswith("/") else url + "/", u)
        ri = retry("GET", retries, lambda: _get(full, hdr, timeout))
        (dest / name).write_bytes(ri.content)
        yield dest / name


def local_iter(folder):
    for p in sorted((p for p in folder.iterdir() if p.suffix.lower() in EXT), key=lambda p: p.name.casefold()):
        yield p


# ---------- ตรวจ outputs ----------
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
    return msgs


# ---------- HTTP POST ----------
def make_zip(repo, out, team):
    """team_XX_Day3.zip ของ repo (รวม outputs/) ไม่เอา .git / __pycache__ / ภาพ input / config ที่มีรหัส"""
    zpath = out.parent / f"team_{team}_Day3.zip"
    root = f"team_{team}_day3"
    zpath.unlink(missing_ok=True)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(repo.rglob("*")):
            rel = p.relative_to(repo)
            if p.is_dir() or p == zpath or p.name in ZIP_SKIP_FILES or p.suffix == ".zip":
                continue
            if any(part in ZIP_SKIP_DIRS for part in rel.parts) or (
                    p.name.startswith("client_config") and p.name != "client_config.example.json"):
                continue
            z.write(p, f"{root}/{rel}")
    return zpath


def task_http(cfg, a, result):
    try:
        zpath = make_zip(a.repo, a.out, cfg["team"])
        log("HTTP", f"zip {zpath.name} {zpath.stat().st_size / 1e6:.1f} MB")
        res_bytes = (a.out / "predictions.json").read_bytes()

        def post():
            files = {cfg.get("post_results_field", "results"): ("results.json", res_bytes, "application/json"),
                     cfg.get("post_zip_field", "zip"): (zpath.name, zpath.read_bytes(), "application/zip")}
            r = requests.post(cfg["post_url"], files=files, data=cfg.get("post_extra_fields") or {},
                              headers=headers(cfg), timeout=a.timeout)
            log("HTTP", f"POST {r.status_code}: {r.text[:300]}")
            r.raise_for_status()
            return r
        retry("HTTP", a.retries, post)
        result["http"] = True
    except Exception as e:  # noqa: BLE001
        log("HTTP", f"ล้ม: {e}")
        result["http"] = False


# ---------- MQTT ----------
class Mqtt:
    def __init__(self, cfg, a):
        import paho.mqtt.client as mqtt
        self.cfg, self.a, self.infos = cfg, a, {}
        cid = f"team{cfg['team']}-{os.getpid()}"
        try:
            self.c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=cid)
        except AttributeError:  # paho 1.x
            self.c = mqtt.Client(client_id=cid)
        if cfg.get("mqtt_user"):
            self.c.username_pw_set(cfg["mqtt_user"], os.environ.get("MQTT_PASSWORD") or cfg.get("mqtt_password") or None)
        if cfg.get("mqtt_tls"):
            self.c.tls_set()
        self.c.loop_start()

    def connect(self):
        def once():
            self.c.connect(self.cfg["mqtt_host"], int(self.cfg.get("mqtt_port", 1883)), keepalive=30)
            end = time.time() + 10
            while not self.c.is_connected() and time.time() < end:
                time.sleep(0.05)
            if not self.c.is_connected():
                raise RuntimeError("broker ไม่ตอบรับใน 10 s (host/port/user/password/TLS ผิด?)")
            log("MQTT", f"เชื่อมต่อ {self.cfg['mqtt_host']}:{self.cfg.get('mqtt_port', 1883)} แล้ว")
        retry("MQTT", self.a.retries, once)

    def publish(self, name, payload):
        """ส่งทันที ไม่รอยืนยัน (ยืนยันทีหลังใน finish)"""
        try:
            self.infos[name] = (payload, self.c.publish(self.cfg["mqtt_topic"], payload, qos=int(self.cfg.get("mqtt_qos", 1))))
        except Exception as e:  # noqa: BLE001
            log("MQTT", f"{name} publish ล้ม: {e}")
            self.infos[name] = (payload, None)

    def finish(self, result):
        ok = 0
        for name, (payload, info) in self.infos.items():
            def wait(info=info, payload=payload):
                if info is None or not self._wait(info):
                    info = self.c.publish(self.cfg["mqtt_topic"], payload, qos=int(self.cfg.get("mqtt_qos", 1)))
                    if not self._wait(info):
                        raise RuntimeError(f"ไม่ได้รับยืนยัน rc={info.rc}")
            try:
                retry("MQTT", self.a.retries, wait)
                ok += 1
            except Exception:  # noqa: BLE001
                log("MQTT", f"{name} ✗")
        self.c.loop_stop()
        self.c.disconnect()
        log("MQTT", f"ยืนยันแล้ว {ok}/{len(self.infos)}")
        result["mqtt"] = bool(self.infos) and ok == len(self.infos)

    def _wait(self, info):
        try:
            info.wait_for_publish(timeout=self.a.timeout)
        except Exception:  # noqa: BLE001
            return False
        return info.is_published()


# ---------- โหมด stream ----------
def load_infer(path):
    spec = importlib.util.spec_from_file_location("infer_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_stream(cfg, a):
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
    if a.only in ("both", "mqtt"):
        try:
            mq = Mqtt(cfg, a)
            mq.connect()
        except Exception as e:  # noqa: BLE001
            log("MQTT", f"ต่อไม่ได้ ({e}) — ทำ HTTP ต่อ")
            mq = None
    log("READY", f"โหลดโมเดล imgsz {imgsz} + MQTT {'พร้อม' if mq else 'ไม่พร้อม'} · เริ่มรับภาพ")

    src = local_iter(a.local_input) if a.local_input else fetch_iter(
        cfg, (a.work or a.out.parent / "input").resolve(), a.timeout, a.retries)
    pool = ThreadPoolExecutor(max_workers=1)
    pending, preds, seen = [], [], set()
    for f in src:
        t0 = time.perf_counter()
        if f.stem.casefold() in seen:
            log("IMG", f"ข้าม {f.name}: ชื่อซ้ำ")
            continue
        try:
            with Image.open(f) as im:
                rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
        except Exception as e:  # noqa: BLE001
            log("IMG", f"อ่าน {f.name} ไม่ได้: {e} — ข้าม")
            continue
        seen.add(f.stem.casefold())
        inst, objects = inf.predict(net, imgsz, rgb, ia)
        msg = inf.mqtt_message(f.name, objects, time.perf_counter() - t0)
        payload = json.dumps(msg, ensure_ascii=False)
        if mq:
            mq.publish(f.name, payload)
        (a.out / "mqtt" / f"{f.stem}.json").write_text(payload, encoding="utf-8")
        pending.append(pool.submit(inf.write_outputs, a.out, f.stem, rgb, inst, ia.vis_scale))
        preds.append({"image_name": f.name, "objects": objects})
        log("IMG", f"{f.name}: {len(objects)} เม็ด · ทาย+publish {time.perf_counter() - t0:.2f} s")
    for p in pending:
        p.result()
    pool.shutdown(wait=True)
    if not preds:
        raise SystemExit("Error: ไม่ได้ภาพเลย")
    preds.sort(key=lambda p: p["image_name"].casefold())  # ลำดับเดียวกับ infer.py
    inf.write_predictions(a.out / "predictions.json", preds)
    check_outputs(a.out, len(preds))

    result, ths = {}, []
    if a.only in ("both", "http"):
        ths.append(threading.Thread(target=task_http, args=(cfg, a, result)))
    if mq:
        ths.append(threading.Thread(target=mq.finish, args=(result,)))
    elif a.only in ("both", "mqtt"):
        result["mqtt"] = False
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    return result


# ---------- โหมด batch (สำรอง) ----------
def run_batch(cfg, a):
    if a.local_input:
        inp = a.local_input
        n = sum(1 for _ in local_iter(inp))
        log("GET", f"ข้าม · ใช้ {n} ภาพจาก {inp}")
    else:
        inp = (a.work or a.out.parent / "input").resolve()
        n = sum(1 for _ in fetch_iter(cfg, inp, a.timeout, a.retries))
    if not a.skip_infer:
        cmd = [sys.executable, str(a.infer), "--input", str(inp), "--weights", str(a.weights), "--out", str(a.out)]
        cmd += shlex.split(a.infer_extra)
        log("INFER", " ".join(cmd))
        if subprocess.run(cmd).returncode != 0:
            raise SystemExit("Error: infer.py ล้ม")
    msgs = check_outputs(a.out, n)
    result, ths = {}, []
    if a.only in ("both", "http"):
        ths.append(threading.Thread(target=task_http, args=(cfg, a, result)))
    if a.only in ("both", "mqtt"):
        def mqtt_all():
            try:
                mq = Mqtt(cfg, a)
                mq.connect()
                for m in msgs:
                    mq.publish(m.name, m.read_text(encoding="utf-8"))
                mq.finish(result)
            except Exception as e:  # noqa: BLE001
                log("MQTT", f"ล้ม: {e}")
                result["mqtt"] = False
        ths.append(threading.Thread(target=mqtt_all))
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--infer", required=True, type=Path, help="infer.py ที่ใช้")
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="โฟลเดอร์ผล (เช่น outputs/)")
    ap.add_argument("--repo", required=True, type=Path, help="โฟลเดอร์ repo ที่จะ zip")
    ap.add_argument("--mode", choices=["stream", "batch"], default="stream")
    ap.add_argument("--work", type=Path, default=None, help="ที่เก็บภาพที่ GET มา (ค่าเริ่ม <out>/../input)")
    ap.add_argument("--local-input", type=Path, default=None, help="ข้าม GET ใช้ภาพในโฟลเดอร์นี้")
    ap.add_argument("--skip-infer", action="store_true", help="(batch) ไม่รัน infer ส่งผลที่มีใน --out")
    ap.add_argument("--infer-extra", default="", help='argument เพิ่มให้ infer ในเครื่องหมายคำพูด เช่น "--write-mqtt 1"')
    ap.add_argument("--only", choices=["both", "http", "mqtt"], default="both")
    ap.add_argument("--timeout", type=float, default=60)
    ap.add_argument("--retries", type=int, default=3)
    a = ap.parse_args()

    cfg = json.loads(a.config.read_text(encoding="utf-8"))
    a.repo, a.out = a.repo.resolve(), a.out.resolve()
    result = run_stream(cfg, a) if a.mode == "stream" else run_batch(cfg, a)
    log("DONE", f"{result}")
    sys.exit(0 if result and all(result.values()) else 1)


if __name__ == "__main__":
    main()
