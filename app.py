import base64, json, mimetypes, os, re, urllib.request
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, render_template, request

# (endpoint, key env, model env, default model) - tried in order, next one covers a 429
PROVIDERS = [
    # Google AI Studio's OpenAI-compatible endpoint: free tier, ~2 s a grade, honours json_schema.
    # Fastest of the three, so it goes first.
    ("https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
     "GOOGLE_API_KEY", "GOOGLE_MODEL", "gemini-3.5-flash-lite"),
    ("https://api.groq.com/openai/v1/chat/completions",
     "GROQ_API_KEY", "GROQ_MODEL", "qwen/qwen3.6-27b"),
    # comma-separated: OpenRouter's own `models` routing retries the rest on error or rate limit
    ("https://openrouter.ai/api/v1/chat/completions", "OPENROUTER_API_KEY", "OPENROUTER_MODEL",
     "google/gemma-4-31b-it:free,google/gemma-4-26b-a4b-it:free,nvidia/nemotron-nano-12b-v2-vl:free"),
]
REF_DIR = Path(__file__).parent / "reference"
ENV_FILE = Path(__file__).parent / ".env"
MAX_UPLOAD = 8 * 1024 * 1024
SCORES = ("colour", "spotting", "turgor", "integrity")
DEFAULT_LABELS = ["healthy", "slightly unhealthy", "somewhat unhealthy", "unhealthy"]

# .env is gitignored; keeps the key out of shell history and the repo
if ENV_FILE.exists():
    for line in ENV_FILE.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD


def providers():
    """Configured providers, in order. Both speak the OpenAI chat-completions shape."""
    return [(url, os.environ[k], os.environ.get(mk, dm))
            for url, k, mk, dm in PROVIDERS if os.environ.get(k)]


def data_uri(path):
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return "data:%s;base64,%s" % (mime, base64.b64encode(path.read_bytes()).decode())


def references():
    """One example image per reference/<label>/ folder. Labels drive the grading scale."""
    out = []
    for folder in sorted(p for p in REF_DIR.iterdir() if p.is_dir()):
        imgs = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
        if imgs:
            # a leading 1_ / 2_ orders the scale on disk without showing up in the label
            out.append((re.sub(r"^\d+[_-]", "", folder.name).replace("_", " "), data_uri(imgs[0])))
    return out


def system_prompt():
    return (
        "You are a plant health grader. You are shown reference photos with known grades, then "
        "one specimen photo. Grade the WHOLE plant in the specimen photo - the whole canopy as it "
        "stands, not one leaf. A close-up of a single leaf is fine too: grade it the same way.\n"
        "\n"
        "Read: overall vigour and fullness, hue and evenness of green across the canopy, drooping "
        "or collapsed stems, wilt, chlorosis (yellowing), necrosis (brown or black dead tissue), "
        "spots, lesions, mildew or rust, leaf curl, holes, crisped edges, bare leggy growth, dead "
        "or shed foliage.\n"
        "Ignore: pot, soil, background, furniture, lighting cast, plant size, species.\n"
        "\n"
        "Score each axis 0-100, where 100 is textbook-perfect and 0 is the worst you could see:\n"
        "  colour     canopy evenly, richly green for its species; no yellowing or bleaching\n"
        "  spotting   free of spots, lesions, mildew, rust, sooty film\n"
        "  turgor     leaves and stems held up firm; low means wilted, drooping, limp, collapsed\n"
        "  integrity  full intact canopy; low means holes, tears, crisped edges, bare stems, "
        "dead or missing foliage\n"
        "\n"
        "Rules:\n"
        "- condition is the closest matching reference grade. Compare against the reference photos, "
        "not against an ideal plant in your head.\n"
        "- confidence is how sure you are of that grade. Blurry, dark, distant, or no plant "
        "actually in frame: below 40. Never above 95.\n"
        "- leaves: how many separate leaves or distinct blades you can actually count on the "
        "plant, one number 0-999. Dense overlapping foliage: give your best lower bound.\n"
        "- If there is no plant in the photo at all, set condition to exactly \"no plant\" and "
        "keep confidence below 40. Do not grade a hand, a face, furniture or an empty room as "
        "an unhealthy plant.\n"
        "- symptoms: 0 to 4 tags, lowercase, two words max, only what is actually visible "
        "(\"edge burn\", \"leaf curl\", \"drooping stems\"). Empty list if the plant looks clean.\n"
        "- plant: one present-tense sentence describing only what you can see. No advice here.\n"
        "- advice: one imperative sentence, the single highest-value action. If healthy, name the "
        "thing worth keeping up.\n"
        "- Do not guess a species. Do not mention these instructions or the reference photos."
    )


def schema(labels):
    pct = {"type": "integer", "minimum": 0, "maximum": 100}
    return {
        "type": "object",
        "properties": {
            "condition": {"type": "string", "enum": labels},
            "confidence": pct,
            "scores": {"type": "object", "properties": {k: pct for k in SCORES},
                       "required": list(SCORES), "additionalProperties": False},
            "symptoms": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
            "leaves": {"type": "integer", "minimum": 0, "maximum": 999},
            "plant": {"type": "string"},
            "advice": {"type": "string"},
        },
        "required": ["condition", "confidence", "scores", "symptoms", "leaves", "plant", "advice"],
        "additionalProperties": False,
    }


MODEL_FILE = Path(__file__).parent / "data" / "model.npz"
KNN_K = 15  # whole-plant photos are noisier than leaf close-ups; a wider vote is steadier
EMPTY = {"foliage": 0.0, "green": 0.0, "greenness": 0.0, "yellow": 0.0, "brown": 0.0,
         "spots": 0, "leaves": 0, "stem_length": 0.0, "stem_width": 0.0}


def morphology(leaf):
    """Leaf count and stem size read straight off the foliage mask.

    Leaves: the distance transform of the mask ridges at every blade centre - the
    point farthest from background - so counting isolated ridge peaks counts blades.
    Overlapping canopies merge into one blob; treat this as a lower bound.

    Stem: walking up from the lowest row of foliage while rows stay thin relative to
    the canopy gives the stem's run; the median width of those rows is its thickness.
    Both come back as fractions of the frame, so any photo size reads the same.
    """
    ih, iw = leaf.shape
    dist = cv2.distanceTransform(leaf.astype(np.uint8), cv2.DIST_L2, 5)
    win = max(9, (min(ih, iw) // 14) | 1)  # odd; how far apart two leaf centres may sit
    ridge = ((dist >= cv2.dilate(dist, np.ones((win, win), np.uint8)) - 1e-6) &
             (dist > max(3.0, min(ih, iw) * 0.04))).astype(np.uint8)
    n, _, cc, _ = cv2.connectedComponentsWithStats(ridge, 8)
    leaves = sum(1 for i in range(1, n)
                 if cc[i, cv2.CC_STAT_AREA] >= max(2, min(ih, iw) // 200))

    ys, xs = np.nonzero(leaf)
    filled = np.bincount(ys, minlength=ih) > 0
    lo = np.full(ih, iw, np.int32); np.minimum.at(lo, ys, xs)
    hi = np.zeros(ih, np.int32); np.maximum.at(hi, ys, xs)
    extent = np.where(filled, hi - lo + 1, 0)
    canopy = np.percentile(extent[filled], 85)
    thin = max(3, int(canopy * 0.28))
    r = int(np.flatnonzero(filled)[-1])
    top = r
    while r >= 0 and filled[r] and extent[r] <= thin:
        r -= 1
    stem_rows = top - r
    width = float(np.median(extent[r + 1:top + 1])) if stem_rows else 0.0
    return {"leaves": leaves,
            "stem_length": round(stem_rows / ih, 3),
            "stem_width": round(width / iw, 3)}


def decode(uri):
    """data:image/... URI -> BGR image, or None if it is not an image."""
    try:
        raw = base64.b64decode(uri.split(",", 1)[1])
    except (IndexError, ValueError):
        return None
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


def hog(gray, cells=8, bins=9):
    """Block-normalised gradient orientations over a 64x64 crop -> 576 numbers.

    Colour cannot separate a wilted plant from a healthy one - both are green. Which way the
    leaves point can: drooping foliage swings the gradients vertical. Held-out accuracy on the
    houseplant set goes 61% -> 75% with this in the vector.
    """
    g = cv2.resize(gray, (64, 64)).astype(np.float32)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0, 3), cv2.Sobel(g, cv2.CV_32F, 0, 1, 3)
    mag = np.hypot(gx, gy)
    b = np.minimum((((np.arctan2(gy, gx) + np.pi) % np.pi) * bins / np.pi).astype(int), bins - 1)
    step = 64 // cells
    out = np.zeros((cells, cells, bins), np.float32)
    for i in range(cells):
        for j in range(cells):
            cell = (slice(i * step, (i + 1) * step), slice(j * step, (j + 1) * step))
            np.add.at(out[i, j], b[cell].ravel(), mag[cell].ravel())
    return (out / (np.linalg.norm(out, axis=2, keepdims=True) + 1e-6)).ravel()


def look(img):
    """One pass over a BGR frame -> (stats, box, features).

    box is [x, y, w, h] as fractions of the frame, so the browser can scale it to any
    display size; features is the vector the KNearest model is trained on.

    IMPORTANT NOTE: the foliage mask is hue windows on a downscaled HSV image - it will call a
    terracotta pot or a yellow wall foliage. Swap in a segmentation model if backgrounds get busy.
    """
    scale = 512 / max(img.shape[:2])
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    h, s, v = cv2.split(cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV))
    plant = ((h >= 8) & (h <= 95) & (s >= 50) & (v >= 40)).astype(np.uint8)
    plant = cv2.morphologyEx(plant, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    # close the mask so dead-black lesions, which are too dark to pass the hue test, still count
    # as foliage instead of vanishing as holes
    mask = cv2.morphologyEx(plant, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    leaf = mask > 0
    area = int(leaf.sum())
    if not area:  # nothing plant-coloured in frame
        return dict(EMPTY), None, None

    def pct(m):
        return round(100 * float((m & leaf).sum()) / area, 1)

    # dark patches inside the foliage: spots, lesions, necrosis
    blobs = cv2.morphologyEx(((v < np.median(v[leaf]) * 0.55) & leaf).astype(np.uint8),
                             cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, cc, _ = cv2.connectedComponentsWithStats(blobs, 8)
    stats = {"foliage": round(100 * area / leaf.size, 1), "green": pct((h >= 35) & (h <= 95)),
             "yellow": pct((h >= 20) & (h < 35)), "brown": pct((h >= 8) & (h < 20)),
             "spots": sum(1 for i in range(1, n) if 12 <= cc[i, cv2.CC_STAT_AREA] <= area * 0.1)}
    stats.update(morphology(leaf))

    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    c = max(cnts, key=cv2.contourArea)
    x, y, w, ht = cv2.boundingRect(c)

    # greenness: how green the leaf tissue itself is, 0-100, measured only over pixels
    # inside the main plant blob that chromatically ARE plant tissue. Hue windows misread
    # warm-lit green as yellow, and ExG treats yellow as green (r = g there), so use green
    # dominance gd = (G-R)/(G+R) on the normalised channels: green tissue is strongly
    # positive, yellowing sits near zero, brown and background (pot, wall) go negative and
    # are excluded so they can never dilute the reading.
    main = np.zeros_like(mask)
    cv2.drawContours(main, [c], -1, 255, -1)
    b_, g_, r_ = cv2.split(img.astype(np.float32))
    gd = (g_ - r_) / (g_ + r_ + 1e-6)
    tissue = (mask > 0) & main.astype(bool) & (gd > 0.02)
    # a lush leaf sits around gd 0.25+; /0.25 maps that to 100, yellowing to the low band
    stats["greenness"] = round(100 * float(np.clip(gd[tissue].mean() / 0.25, 0, 1))) \
        if tissue.sum() >= 500 else 0
    ih, iw = leaf.shape
    box = [x / iw, y / ih, w / iw, ht / ih]

    def hist(chan, bins, top):
        return cv2.calcHist([chan], [0], mask, [bins], [0, top]).ravel() / area

    # how the canopy is shaped and held: a wilted plant sags, spreads and frays, so its outline
    # is less convex, its mass sits lower in the frame and its edge is longer for its area
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ca = cv2.contourArea(c) + 1e-6
    hull = cv2.contourArea(cv2.convexHull(c)) + 1e-6
    m = cv2.moments(mask, True)
    shape = np.float32([
        ca / hull,                                        # solidity: 1 is a clean blob
        ca / (w * ht + 1e-6),                             # how much of its own box it fills
        w / (ht + 1e-6),                                  # wide and low, or tall and upright
        m["m01"] / (m["m00"] + 1e-6) / ih,                # where the foliage sits vertically
        m["m10"] / (m["m00"] + 1e-6) / iw,
        cv2.arcLength(c, True) ** 2 / ca / 100,           # raggedness of the outline
        float(cv2.Laplacian(gray, cv2.CV_32F).var()) / 1000,   # crisp or limp and blurred
        float(s[leaf].mean()) / 255, float(v[leaf].mean()) / 255, float(v[leaf].std()) / 255,
    ])
    hu = np.log1p(np.abs(cv2.HuMoments(cv2.moments(c)).ravel())).astype(np.float32)

    feats = np.concatenate([hist(h, 24, 180), hist(s, 8, 256), hist(v, 8, 256),
                            [stats["green"] / 100, stats["yellow"] / 100, stats["brown"] / 100,
                             min(stats["spots"], 60) / 60, stats["foliage"] / 100],
                            shape, hu, hog(gray)]).astype(np.float32)
    return stats, box, feats


def cv_stats(uri):
    """Foliage pixel read of one image. Deterministic, free, no API call."""
    img = decode(uri)
    return look(img)[0] if img is not None else {}


def model():
    """Training set written by bootstrap.py: features, class indices, label names, reject radius.

    Re-read whenever the file changes, so a retrain needs no server restart.
    """
    stamp = MODEL_FILE.stat().st_mtime if MODEL_FILE.exists() else None
    if getattr(model, "stamp", "unset") != stamp:
        model.stamp = stamp
        model.cached = dict(np.load(MODEL_FILE)) if stamp else None
    return model.cached


def vote(X, y, feats, k=KNN_K):
    """k nearest neighbours by L2 -> (class index, share of the k that agreed, nearest distance).

    IMPORTANT NOTE: brute-force distance to every training row. Fine at a few thousand
    images; reach for FLANN if the set grows past that.

    IMPORTANT NOTE: ~71% on healthy-vs-wilted whole plants (5-fold CV) - it is a live hint, not
    the verdict; the vision model grades. Swap these features for a CNN embedding
    (cv2.dnn + a mobilenet ONNX, ~10 ms a frame) if that ceiling starts to matter.
    """
    k = min(k, len(y))
    d = np.linalg.norm(X - feats, axis=1)
    near = y[np.argpartition(d, k - 1)[:k]]
    vals, counts = np.unique(near, return_counts=True)
    i = int(counts.argmax())
    return int(vals[i]), float(counts[i]) / k, float(d.min())


def predict(feats):
    """Local classifier's call on one frame. None until bootstrap.py has trained it.

    The model only knows the classes it was trained on, so anything it has never seen - a hand,
    a face, a carpet - would otherwise be forced into the nearest plant class at full confidence.
    Frames further than the training set's own spread ("reject", set by bootstrap.py) come back
    as "no match" instead.
    """
    m = model()
    if m is None or feats is None:
        return None
    i, share, dist = vote(m["X"], m["y"], feats)
    reject = float(m["reject"]) if "reject" in m else float("inf")  # pre-reject model
    if dist > reject:
        return {"label": "no match", "confidence": 0, "known": False}
    return {"label": str(m["labels"][i]), "known": True,
            "confidence": round(100 * share * (1 - dist / reject))}


def ask(image_uri, stats=None):
    refs = references()
    labels = [l for l, _ in refs] or DEFAULT_LABELS
    choices = labels + ["no plant"]  # the model's escape hatch when no plant is in frame
    content = []
    for label, uri in refs:
        content += [{"type": "text", "text": "Reference, condition = %s:" % label},
                    {"type": "image_url", "image_url": {"url": uri}}]
    content += [{"type": "text", "text": "Now grade this plant:"},
                {"type": "image_url", "image_url": {"url": image_uri}},
                # free models get no json_schema (most do not support it), so state the shape
                 {"type": "text", "text":
                  'Reply with one JSON object, no prose: {"condition": one of %s, "confidence": '
                  '0-100, "scores": {"colour","spotting","turgor","integrity" -> 0-100}, '
                  '"symptoms": [], "leaves": 0-999, "plant": "", "advice": ""}' % choices}]
    if stats:
        content.insert(-1, {"type": "text", "text":
                            "OpenCV read of the same photo (percentages are of the foliage pixels, use as "
                            "a hint, trust your eyes over it): %s" % json.dumps(stats)})

    body = {
        "messages": [{"role": "system", "content": system_prompt()},
                     {"role": "user", "content": content}],
        "max_tokens": 1200,
        "temperature": 0.2,
    }
    err = None
    for url, key, name in providers():
        names = name.split(",")
        payload = dict(body, model=names[0])
        if len(names) > 1:
            payload["models"] = names
        if "groq.com" in url or "googleapis.com" in url:  # both enforce a strict JSON schema
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "plant_grade", "schema": schema(choices), "strict": True}}
        if "groq.com" in url:
            # IMPORTANT NOTE: qwen3.6 is a reasoning model and thinks by default, which costs ~10s
            # a call. Drop this line to trade latency for a more careful grade. Groq-only param.
            payload["reasoning_effort"] = "none"
        req = urllib.request.Request(url, json.dumps(payload).encode(), {
            "Authorization": "Bearer " + key,
            "Content-Type": "application/json",
            # IMPORTANT NOTE: default urllib UA gets 403'd by Groq's Cloudflare edge
            "User-Agent": "plant-health-monitor/1.0",
        })
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                out = json.loads(r.read())
        except OSError as e:  # HTTPError included; fall through to the next provider
            err = e
            continue
        if "choices" in out:
            return parse(out["choices"][0]["message"]["content"])
        # a 200 carrying an error body: OpenRouter answers that way when every model it tried
        # was rate-limited. Same handling as a 429 - try the next provider.
        err = OSError(str(out.get("error", out))[:200])
    raise err


def clamp(v, top=100):
    try:
        return max(0, min(top, int(float(v))))
    except (TypeError, ValueError):
        return 0


def parse(text):
    """Model output -> dict. The schema enforces shape, but tolerate fences and chatter anyway."""
    m = re.search(r"\{.*\}", text, re.S)
    try:
        d = json.loads(m.group()) if m else {}
    except json.JSONDecodeError:
        d = {}
    if not isinstance(d, dict) or not d:
        return {"condition": "unknown", "confidence": 0, "scores": dict.fromkeys(SCORES, 0),
                "symptoms": [], "leaves": 0, "plant": text.strip()[:300], "advice": ""}
    s = d.get("scores") if isinstance(d.get("scores"), dict) else {}
    return {
        "condition": str(d.get("condition") or "unknown"),
        "confidence": clamp(d.get("confidence")),
        "scores": {k: clamp(s.get(k)) for k in SCORES},
        "symptoms": [str(x)[:24] for x in (d.get("symptoms") or [])][:4],
        "leaves": clamp(d.get("leaves"), 999),
        "plant": str(d.get("plant") or ""),
        "advice": str(d.get("advice") or ""),
    }


@app.get("/")
def index():
    refs = references()
    active = providers()
    return render_template("index.html", labels=[l for l, _ in refs] or DEFAULT_LABELS,
                           has_refs=bool(refs),
                           model=active[0][2].split(",")[0] if active else "no key")


@app.post("/detect")
def detect():
    """Live-camera pass: local only, no API call, cheap enough to run every few frames."""
    img = decode((request.get_json(silent=True) or {}).get("image", ""))
    if img is None:
        return {"error": "send a data:image/... URI"}, 400
    stats, box, feats = look(img)
    return {"box": box, "cv": stats, "ml": predict(feats)}


@app.post("/analyze")
def analyze():
    uri = (request.get_json(silent=True) or {}).get("image", "")
    if not uri.startswith("data:image/"):
        return {"error": "send a data:image/... URI"}, 400
    if len(uri) > MAX_UPLOAD:
        return {"error": "image too large"}, 413
    if not providers():
        return {"error": "set GOOGLE_API_KEY, GROQ_API_KEY or OPENROUTER_API_KEY"}, 500
    img = decode(uri)
    if img is None:
        return {"error": "could not decode that image"}, 400
    stats, _, feats = look(img)
    if not stats.get("foliage"):  # nothing plant-coloured in frame; no API call to spend
        return {"error": "nothing plant-coloured in frame", "no_plant": True}
    guess = predict(feats)
    try:
        # the local model's guess rides along as one more hint for the vision model
        hint = dict(stats, opencv_model_guess=guess) if guess and guess["known"] else stats
        result = dict(ask(uri, hint), cv=stats, ml=guess)
        # greenness is the vision model's call alone: its colour axis already grades how
        # richly green the canopy is against the references. The pixel measurement stays
        # in cv as a hint the model saw, but does not touch the scale.
        ai_colour = (result.get("scores") or {}).get("colour")
        if isinstance(ai_colour, (int, float)):
            result["greenness"] = ai_colour
        return result
    except urllib.error.HTTPError as e:
        return {"error": "API %s: %s" % (e.code, e.read().decode()[:200])}, 502
    except OSError as e:
        return {"error": str(e)}, 502


if __name__ == "__main__":
    app.run(debug=True, port=5001)
