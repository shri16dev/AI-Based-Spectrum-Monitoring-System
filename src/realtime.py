"""
realtime.py — RF signal generator that streams predictions to Flask backend.

Changes from standalone version:
  - POSTs each chunk to Flask /predict instead of classifying locally
  - Flask handles the model, SSE broadcast, and stats
  - Fallback message if Flask is unreachable

Usage (start app.py FIRST, then this):
  python3 realtime.py
  python3 realtime.py --mode adversarial --speed fast
  python3 realtime.py --mode sweep
  python3 realtime.py --mode static --class noise
  python3 realtime.py --url http://192.168.1.10:5000
"""

import numpy as np
import time
import argparse
import sys

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="RF signal streamer -> Flask")
parser.add_argument("--mode",  choices=["random", "static", "sweep", "adversarial"],
                    default="random")
parser.add_argument("--class", dest="fixed_class",
                    choices=["signal", "noise", "multi"], default="signal")
parser.add_argument("--speed", choices=["slow", "normal", "fast"], default="normal")
parser.add_argument("--url",   default="http://localhost:5000")
parser.add_argument("--max",   type=int, default=0)
args = parser.parse_args()

SPEED_MAP   = {"slow": 2.0, "normal": 0.5, "fast": 0.05}
SLEEP_TIME  = SPEED_MAP[args.speed]
CHUNK_SIZE  = 1024
STEP        = 512
CLASSES     = ["signal", "noise", "multi"]
PREDICT_URL = args.url.rstrip("/") + "/predict"
RNG         = np.random.default_rng()

# ── Generators ────────────────────────────────────────────────────────────────
def _time():
    return np.linspace(0, 1, CHUNK_SIZE, endpoint=False)

def _awgn(sig, snr_db):
    pwr = np.mean(sig**2) + 1e-12
    n   = pwr / (10 ** (snr_db / 10))
    return sig + RNG.normal(0, np.sqrt(n), len(sig))

def _pink():
    w = RNG.standard_normal(CHUNK_SIZE)
    F = np.fft.rfft(w)
    f = np.fft.rfftfreq(CHUNK_SIZE); f[0] = 1e-6
    p = np.fft.irfft(F / np.sqrt(f), n=CHUNK_SIZE)
    return p / (np.std(p) + 1e-12) * RNG.uniform(0.5, 1.5)

def sim_signal(hard=False):
    t   = _time()
    snr = RNG.uniform(2, 6) if hard else RNG.uniform(5, 15)
    f   = RNG.integers(40, 130)
    amp = RNG.uniform(0.3, 1.8)
    s   = amp * np.sin(2 * np.pi * f * t)
    if RNG.random() < (0.5 if hard else 0.2):
        s += (amp * 0.15) * np.sin(2 * np.pi * f * 2 * t)
    return _awgn(s, snr)

def sim_noise(hard=False):
    base = _pink()
    if hard or RNG.random() < 0.4:
        t    = _time()
        base += RNG.uniform(0.05, 0.3) * np.sin(
                    2 * np.pi * RNG.integers(40, 200) * t)
    return base

def sim_multi(hard=False):
    t   = _time()
    snr = RNG.uniform(3, 7) if hard else RNG.uniform(6, 14)
    if hard or RNG.random() < 0.4:
        f1 = RNG.integers(50, 100); f2 = f1 + RNG.integers(5, 20)
    else:
        f1 = RNG.integers(40, 90);  f2 = RNG.integers(100, 200)
    a1, a2 = RNG.uniform(0.2, 1.6), RNG.uniform(0.2, 1.6)
    return _awgn(a1*np.sin(2*np.pi*f1*t) + a2*np.sin(2*np.pi*f2*t), snr)

GENERATORS = {"signal": sim_signal, "noise": sim_noise, "multi": sim_multi}

# ── Sweep state ───────────────────────────────────────────────────────────────
sweep_idx, sweep_count = 0, 0

def next_chunk():
    global sweep_idx, sweep_count
    hard = (args.mode == "adversarial")
    if   args.mode == "static": cls = args.fixed_class
    elif args.mode == "sweep":
        cls = CLASSES[sweep_idx]
        sweep_count += 1
        if sweep_count >= 5:
            sweep_count = 0; sweep_idx = (sweep_idx + 1) % len(CLASSES)
    else:
        cls = RNG.choice(CLASSES)
    return cls, GENERATORS[cls](hard=hard)

# ── Pre-flight check ──────────────────────────────────────────────────────────
if not HAS_REQUESTS:
    print("[ERROR] pip install requests"); sys.exit(1)

try:
    requests.get(args.url + "/api/stats", timeout=3)
    print(f"[OK] Flask reachable at {args.url}")
except Exception:
    print(f"[ERROR] Cannot reach Flask at {args.url}")
    print("        Run:  python3 app.py")
    sys.exit(1)

# ── Main loop ─────────────────────────────────────────────────────────────────
buffer = np.array([])
sent   = 0
C = {"signal":"\033[92m","noise":"\033[93m","multi":"\033[94m",
     "ok":"\033[92m","err":"\033[91m","dim":"\033[90m","r":"\033[0m"}
def col(t, k): return f"{C.get(k,'')}{t}{C['r']}"

print(f"\n  Streaming -> {PREDICT_URL}")
print(f"  Mode: {args.mode.upper()}  |  Speed: {args.speed}  |  Ctrl+C to stop\n")

try:
    while True:
        true_cls, new_data = next_chunk()
        buffer = np.concatenate((buffer, new_data))

        while len(buffer) >= CHUNK_SIZE:
            chunk  = buffer[:CHUNK_SIZE]
            buffer = buffer[STEP:]
            try:
                resp = requests.post(PREDICT_URL,
                                     json={"chunk": chunk.tolist(),
                                           "true_class": true_cls},
                                     timeout=5)
                d    = resp.json()
                sent += 1
                pred = d.get("predicted", "?")
                conf = d.get("confidence")
                ok   = pred == true_cls
                mark = col("v", "ok") if ok else col("x", "err")
                cstr = f"{conf*100:5.1f}%" if conf else "  n/a "
                print(f"  [{sent:>4}]  "
                      f"true={col(f'{true_cls:<8}', true_cls)}  "
                      f"pred={col(f'{pred:<8}', pred)}  "
                      f"conf={cstr}  {mark}")
            except requests.exceptions.ConnectionError:
                print(col("  [WARN] Flask disconnected...", "dim"))
                time.sleep(2)
            except Exception as e:
                print(col(f"  [ERR] {e}", "err"))

            if args.max and sent >= args.max:
                raise KeyboardInterrupt

        if SLEEP_TIME > 0:
            time.sleep(SLEEP_TIME)

except KeyboardInterrupt:
    print(f"\n  Sent {sent} chunks to Flask. Bye.")
    sys.exit(0)