#!/usr/bin/env python3
"""client.py — Day 3: รับภาพทาง HTTP → infer.py → ส่งผล HTTP (results.json + zip) และ MQTT พร้อมกัน

ขั้นตอน (สไลด์ Day 3 หน้า 9–10):
  1. GET ภาพจากเซิร์ฟเวอร์กรรมการ → <work>/input/   (ข้ามได้ด้วย --local-input สำหรับทดสอบ/สำรอง)
  2. รัน infer.py ด้วย argument 3 ตัว (+ --write-mqtt 1 ถ้าใช้ตัว toolkit) → <out>/{seg_mask, predictions.json, mqtt, vis}
  3. Task A (เธรด 1): POST results.json (= predictions.json) + team_XX_Day3.zip
     Task B (เธรด 2): publish <out>/mqtt/*.json ภาพละ 1 ข้อความ (qos 1 รอยืนยัน)
  ทั้งสอง task ไม่รอกัน · ลองใหม่ --retries ครั้ง · exit 1 ถ้า task ใดล้ม

ค่าที่ผู้จัดประกาศ 12:00 (URL, ชื่อช่อง form, broker, topic, user/password) อยู่ในไฟล์ --config (JSON)
ไฟล์ config มีรหัสผ่าน → ห้าม commit (repo เป็น public) · รหัสผ่านใส่ทาง env MQTT_PASSWORD / HTTP_TOKEN ได้

ตัวอย่าง config (ค่า <...> = รอเอกสารผู้จัด):
{
  "team": "36",
  "get_url": "<URL ที่ GET ภาพ>",
  "post_url": "<URL ที่ POST ผล>",
  "post_results_field": "results", "post_zip_field": "zip",
  "post_extra_fields": {"team": "36"},
  "http_headers": {},
  "mqtt_host": "<broker>", "mqtt_port": 1883, "mqtt_user": "<user>", "mqtt_password": "",
  "mqtt_topic": "<topic>", "mqtt_tls": false
}

ใช้:
  python client.py --config client_config.json --infer infer.py --weights weights/ --out outputs/ --repo .
  python client.py ... --local-input test_images/      # ข้าม GET (ทดสอบ)
  python client.py ... --skip-infer                    # ส่งผลที่มีอยู่แล้วใน --out ซ้ำ
"""
import argparse
import io
import json
import os
import shlex
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests

EXT = {".jpg", ".jpeg", ".png"}
ZIP_SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "input", ".idea", ".vscode"}
ZIP_SKIP_FILES = {".DS_Store"}


def log(tag, msg):
    print(f"[{time.strftime('%H:%M:%S')}] [{tag}] {msg}", flush=True)


def retry(tag, n, fn):
    for i in range(1, n + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            log(tag, f"ครั้งที่ {i}/{n} ล้ม: {e}")
            if i == n:
                raise
            time.sleep(min(2 * i, 5))


# ---------- 1. รับภาพ ----------
def fetch_images(cfg, dest, timeout, retries):
    """GET get_url · รองรับ 3 แบบ (รูปแบบจริงรอเอกสาร 12:00):
    zip ของภาพ · ภาพเดี่ยว · JSON รายการ (list ของ URL/ชื่อไฟล์ หรือ dict ที่มี key images/files/urls)"""
    dest.mkdir(parents=True, exist_ok=True)
    hdr = headers(cfg)
    url = cfg["get_url"]
    r = retry("GET", retries, lambda: _get(url, hdr, timeout))
    ctype = r.headers.get("Content-Type", "").lower()
    if "zip" in ctype or r.content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            for m in z.infolist():
                name = Path(m.filename).name
                if not m.is_dir() and Path(name).suffix.lower() in EXT and not name.startswith("."):
                    (dest / name).write_bytes(z.read(m))
    elif ctype.startswith("image/"):
        name = Path(urlparse(url).path).name or "image.jpg"
        (dest / name).write_bytes(r.content)
    else:
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
    files = sorted(p for p in dest.iterdir() if p.suffix.lower() in EXT)
    if not files:
        raise SystemExit(f"Error: GET ได้ภาพ 0 ภาพ ({url})")
    log("GET", f"ได้ {len(files)} ภาพ → {dest}")
    return files


def _get(url, hdr, timeout):
    r = requests.get(url, headers=hdr, timeout=timeout)
    r.raise_for_status()
    return r


def headers(cfg):
    h = dict(cfg.get("http_headers") or {})
    tok = os.environ.get("HTTP_TOKEN") or cfg.get("http_token")
    if tok:
        h.setdefault("Authorization", f"Bearer {tok}")
    return h


# ---------- 2. infer ----------
def run_infer(a, inp):
    cmd = [sys.executable, str(a.infer), "--input", str(inp), "--weights", str(a.weights), "--out", str(a.out)]
    cmd += shlex.split(a.infer_extra)
    log("INFER", " ".join(cmd))
    t0 = time.perf_counter()
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        raise SystemExit(f"Error: infer.py exit {rc}")
    log("INFER", f"เสร็จ {time.perf_counter() - t0:.1f} s")


def check_outputs(out, n_images):
    preds = json.loads((out / "predictions.json").read_text(encoding="utf-8"))
    msgs = sorted((out / "mqtt").glob("*.json"))
    if len(preds) != n_images or len(msgs) != n_images:
        raise SystemExit(f"Error: ภาพ {n_images} แต่ predictions {len(preds)} / mqtt {len(msgs)} "
                         "(infer.py ต้องเขียน mqtt/ — ตัว toolkit ใส่ --infer-extra '--write-mqtt 1')")
    by_name = {p["image_name"]: p for p in preds}
    for m in msgs:
        d = json.loads(m.read_text(encoding="utf-8"))
        objs = by_name[d["image"]]["objects"]
        cnt = {}
        for o in objs:
            cnt[o["class"]] = cnt.get(o["class"], 0) + 1
        if any(d["counts"][k] != cnt.get(k, 0) for k in d["counts"]) or len(d["counts"]) != 7:
            raise SystemExit(f"Error: counts ใน {m.name} ไม่ตรง predictions.json")
    log("CHECK", f"predictions {len(preds)} ภาพ = mqtt {len(msgs)} ไฟล์ · counts ตรงกันทุกภาพ")
    return msgs


# ---------- 3A. HTTP POST ----------
def make_zip(repo, out, team):
    """team_XX_Day3.zip ของ repo (รวม outputs/) ไม่เอา .git / __pycache__ / ภาพ input"""
    zpath = out.parent / f"team_{team}_Day3.zip"
    root = f"team_{team}_day3"
    zpath.unlink(missing_ok=True)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(repo.rglob("*")):
            rel = p.relative_to(repo)
            if p.is_dir() or p == zpath or p.name in ZIP_SKIP_FILES or p.suffix == ".zip":
                continue
            if any(part in ZIP_SKIP_DIRS for part in rel.parts) or (p.name.startswith("client_config") and p.name != "client_config.example.json"):
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


# ---------- 3B. MQTT ----------
def task_mqtt(cfg, msgs, a, result):
    try:
        import paho.mqtt.client as mqtt
        try:
            c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"team{cfg['team']}-{os.getpid()}")
        except AttributeError:  # paho 1.x
            c = mqtt.Client(client_id=f"team{cfg['team']}-{os.getpid()}")
        user = cfg.get("mqtt_user")
        if user:
            c.username_pw_set(user, os.environ.get("MQTT_PASSWORD") or cfg.get("mqtt_password") or None)
        if cfg.get("mqtt_tls"):
            c.tls_set()
        c.loop_start()

        def connect():
            c.connect(cfg["mqtt_host"], int(cfg.get("mqtt_port", 1883)), keepalive=30)
            end = time.time() + 10
            while not c.is_connected() and time.time() < end:
                time.sleep(0.1)
            if not c.is_connected():
                raise RuntimeError("broker ไม่ตอบรับใน 10 s (host/port/user/password/TLS ผิด?)")
            log("MQTT", f"เชื่อมต่อ {cfg['mqtt_host']}:{cfg.get('mqtt_port', 1883)} แล้ว")
        retry("MQTT", a.retries, connect)
        ok = 0
        for m in msgs:
            payload = m.read_text(encoding="utf-8")

            def pub():
                info = c.publish(cfg["mqtt_topic"], payload, qos=int(cfg.get("mqtt_qos", 1)))
                info.wait_for_publish(timeout=a.timeout)
                if not info.is_published():
                    raise RuntimeError(f"publish ไม่สำเร็จ rc={info.rc}")
            try:
                retry("MQTT", a.retries, pub)
                ok += 1
                log("MQTT", f"{m.name} ✓ {payload}")
            except Exception:  # noqa: BLE001
                log("MQTT", f"{m.name} ✗")
        c.loop_stop()
        c.disconnect()
        log("MQTT", f"ส่งได้ {ok}/{len(msgs)}")
        result["mqtt"] = ok == len(msgs)
    except Exception as e:  # noqa: BLE001
        log("MQTT", f"ล้ม: {e}")
        result["mqtt"] = False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--infer", required=True, type=Path, help="infer.py ที่ใช้")
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="โฟลเดอร์ผล (เช่น outputs/)")
    ap.add_argument("--repo", required=True, type=Path, help="โฟลเดอร์ repo ที่จะ zip")
    ap.add_argument("--work", type=Path, default=None, help="ที่เก็บภาพที่ GET มา (ค่าเริ่ม <out>/../input)")
    ap.add_argument("--local-input", type=Path, default=None, help="ข้าม GET ใช้ภาพในโฟลเดอร์นี้")
    ap.add_argument("--skip-infer", action="store_true", help="ไม่รัน infer ส่งผลที่มีใน --out")
    ap.add_argument("--infer-extra", default="", help='argument เพิ่มให้ infer.py ในเครื่องหมายคำพูด เช่น "--write-mqtt 1"')
    ap.add_argument("--only", choices=["both", "http", "mqtt"], default="both")
    ap.add_argument("--timeout", type=float, default=60)
    ap.add_argument("--retries", type=int, default=3)
    a = ap.parse_args()

    cfg = json.loads(a.config.read_text(encoding="utf-8"))
    a.repo, a.out = a.repo.resolve(), a.out.resolve()
    t0 = time.perf_counter()
    if a.local_input:
        files = sorted(p for p in a.local_input.iterdir() if p.suffix.lower() in EXT)
        inp = a.local_input
        log("GET", f"ข้าม · ใช้ {len(files)} ภาพจาก {inp}")
    else:
        inp = (a.work or a.out.parent / "input").resolve()
        files = fetch_images(cfg, inp, a.timeout, a.retries)
    if not a.skip_infer:
        run_infer(a, inp)
    msgs = check_outputs(a.out, len(files))

    result, ths = {}, []
    if a.only in ("both", "http"):
        ths.append(threading.Thread(target=task_http, args=(cfg, a, result)))
    if a.only in ("both", "mqtt"):
        ths.append(threading.Thread(target=task_mqtt, args=(cfg, msgs, a, result)))
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    log("DONE", f"{result} · รวม {time.perf_counter() - t0:.1f} s")
    sys.exit(0 if result and all(result.values()) else 1)


if __name__ == "__main__":
    main()
