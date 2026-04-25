"""
app.py — Flask backend for RF classifier web dashboard.

Endpoints:
  GET  /                  → serve dashboard HTML
  GET  /stream            → SSE stream of live predictions
  POST /predict           → accept a chunk from realtime.py, classify, broadcast
  GET  /api/stats         → JSON snapshot of current stats
  GET  /api/history       → last N predictions as JSON
  POST /api/reset         → reset all stats

Run:
  pip install flask flask-cors numpy scikit-learn joblib
  python3 app.py
"""
# Replace these lines near the top of app.py:
import sys
import os

# ADD THIS — point to src/ so features.py can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src')) 


import json
import time
import queue
import threading
import numpy as np
import joblib

from collections import Counter, defaultdict, deque
from datetime import datetime
from flask import Flask, Response, request, jsonify, render_template_string
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(__file__))
from features import extract_features, FEATURE_NAMES

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.pkl')
HISTORY_MAX = 200          # keep last 200 predictions in memory
CLASSES     = ["signal", "noise", "multi"]

app = Flask(__name__)
CORS(app)

# ── Load model ────────────────────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] Model not found at '{MODEL_PATH}'. Run train_model.py first.")
    sys.exit(1)

model     = joblib.load(MODEL_PATH)
has_proba = hasattr(model, "predict_proba")
print(f"[OK] Model loaded — confidence scores: {has_proba}")

# ── Shared state (thread-safe) ────────────────────────────────────────────────
lock        = threading.Lock()
history     = deque(maxlen=HISTORY_MAX)   # list of prediction dicts
cm          = defaultdict(lambda: defaultdict(int))   # confusion matrix
class_true  = Counter()
class_pred  = Counter()
total_count = 0
correct_count = 0
snr_history = deque(maxlen=HISTORY_MAX)
conf_history = deque(maxlen=HISTORY_MAX)

# SSE subscriber queues
subscribers = []
subs_lock   = threading.Lock()

def broadcast(data: dict):
    """Push a JSON event to all SSE subscribers."""
    payload = f"data: {json.dumps(data)}\n\n"
    dead = []
    with subs_lock:
        for q in subscribers:
            try:
                q.put_nowait(payload)
            except queue.Full:
                dead.append(q)
        for q in dead:
            subscribers.remove(q)

# ── Helper: estimate SNR ──────────────────────────────────────────────────────
def estimate_snr(x):
    fft_mag = np.abs(np.fft.rfft(x))
    peak    = np.max(fft_mag)
    floor   = np.median(fft_mag) + 1e-12
    return float(10 * np.log10((peak / floor) ** 2))

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    with open(os.path.join(os.path.dirname(__file__), "templates", "index.html")) as f:
        return f.read()

@app.route("/stream")
def stream():
    """SSE endpoint — each subscriber gets its own queue."""
    q = queue.Queue(maxsize=50)
    with subs_lock:
        subscribers.append(q)

    def generate():
        # Send current stats immediately on connect
        yield f"data: {json.dumps({'type': 'connected', 'total': total_count})}\n\n"
        try:
            while True:
                try:
                    msg = q.get(timeout=30)
                    yield msg
                except queue.Empty:
                    yield ": keepalive\n\n"
        except GeneratorExit:
            with subs_lock:
                if q in subscribers:
                    subscribers.remove(q)

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})

@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept raw signal chunk from realtime.py.
    Body JSON: { "chunk": [...1024 floats...], "true_class": "signal"|"noise"|"multi" }
    """
    global total_count, correct_count

    data = request.get_json(force=True)
    chunk      = np.array(data["chunk"], dtype=np.float64)
    true_cls   = data.get("true_class", "unknown")

    feats  = extract_features(chunk)
    pred   = model.predict([feats])[0]
    proba  = model.predict_proba([feats])[0].tolist() if has_proba else []
    conf   = max(proba) if proba else None
    snr    = estimate_snr(chunk)
    is_ok  = (pred == true_cls)
    ts     = datetime.now().isoformat()

    with lock:
        total_count  += 1
        if is_ok: correct_count += 1
        cm[true_cls][pred] += 1
        class_true[true_cls] += 1
        class_pred[pred]     += 1
        snr_history.append(snr)
        if conf is not None: conf_history.append(conf)

        record = {
            "type"       : "prediction",
            "id"         : total_count,
            "timestamp"  : ts,
            "true_class" : true_cls,
            "predicted"  : pred,
            "correct"    : is_ok,
            "confidence" : conf,
            "snr_db"     : snr,
            "proba"      : dict(zip(CLASSES, proba)) if proba else {},
            "features"   : dict(zip(FEATURE_NAMES, feats)),
            "accuracy"   : correct_count / total_count,
        }
        history.append(record)

    broadcast(record)

    return jsonify({"status": "ok", "predicted": pred, "confidence": conf})

@app.route("/api/stats")
def api_stats():
    with lock:
        acc = correct_count / total_count if total_count else 0
        snr_avg = float(np.mean(snr_history)) if snr_history else 0
        conf_avg = float(np.mean(conf_history)) if conf_history else 0
        low_conf = sum(1 for v in conf_history if v < 0.6)

        return jsonify({
            "total"        : total_count,
            "correct"      : correct_count,
            "accuracy"     : acc,
            "class_true"   : dict(class_true),
            "class_pred"   : dict(class_pred),
            "confusion"    : {tc: dict(row) for tc, row in cm.items()},
            "snr_avg"      : snr_avg,
            "conf_avg"     : conf_avg,
            "low_conf"     : low_conf,
            "subscribers"  : len(subscribers),
        })

@app.route("/api/history")
def api_history():
    n = int(request.args.get("n", 50))
    with lock:
        return jsonify(list(history)[-n:])

@app.route("/api/reset", methods=["POST"])
def api_reset():
    global total_count, correct_count
    with lock:
        total_count = correct_count = 0
        history.clear(); cm.clear()
        class_true.clear(); class_pred.clear()
        snr_history.clear(); conf_history.clear()
    broadcast({"type": "reset"})
    return jsonify({"status": "reset"})

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(os.path.join(os.path.dirname(__file__), "templates"), exist_ok=True)
    print("\n  RF Classifier Web Dashboard")
    print("  ──────────────────────────────")
    print("  Dashboard : http://localhost:5000")
    print("  SSE stream: http://localhost:5000/stream")
    print("  API stats : http://localhost:5000/api/stats")
    print("  Start realtime.py separately to push predictions.\n")
    app.run(debug=False, threaded=True, host="0.0.0.0", port=5000)