"""Self-check: run `python3 test_app.py`. No API key needed."""
import base64, io, os, urllib.error, urllib.request
os.environ.setdefault("GROQ_API_KEY", "test")
import app

# parse() tolerates fences, chatter, and garbage
d = app.parse('```json\n{"condition":"healthy","confidence":91,"plant":"green","advice":"water"}\n```')
assert d["condition"] == "healthy" and d["confidence"] == 91, d
assert app.parse("sure! {\"condition\":\"unhealthy\"}")["condition"] == "unhealthy"
assert app.parse("{broken")["condition"] == "unknown"
# "no plant" is a first-class condition the vision model can return
assert app.parse("{\"condition\":\"no plant\"}")["condition"] == "no plant"
assert app.parse("no json here")["plant"] == "no json here"
assert app.parse("{broken")["scores"] == dict.fromkeys(app.SCORES, 0)

# scores/symptoms are clamped and capped whatever the model returns
d = app.parse('{"condition":"x","confidence":"140","scores":{"colour":-5,"spotting":"70"},'
              '"symptoms":["a","b","c","d","e"]}')
assert d["confidence"] == 100 and d["scores"]["colour"] == 0 and d["scores"]["spotting"] == 70, d
assert d["scores"]["turgor"] == 0 and len(d["symptoms"]) == 4, d

# schema enum tracks the reference folders, so a new folder needs no code change
assert app.schema(["a", "b"])["properties"]["condition"]["enum"] == ["a", "b"]

# references() picks one image per label folder, and drops any ordering prefix from the label
png = base64.b64decode(b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVR4nGNiAAAABgADNjd8qAAAAABJRU5ErkJggg==")
tmp = app.REF_DIR / "9_test_label"
tmp.mkdir(exist_ok=True)
(tmp / "_t.png").write_bytes(png)
try:
    refs = dict(app.references())
    assert "test label" in refs, sorted(refs)
    assert refs["test label"].startswith("data:image/png;base64,")
finally:
    (tmp / "_t.png").unlink()
    tmp.rmdir()

# cv_stats(): a flat green frame reads as almost all foliage, all green, no spots
import numpy as np, cv2


def uri_of(bgr):
    return "data:image/png;base64," + base64.b64encode(cv2.imencode(".png", bgr)[1]).decode()


green = np.zeros((120, 120, 3), np.uint8)
green[:] = (40, 180, 60)
st = app.cv_stats(uri_of(green))
assert st["foliage"] > 95 and st["green"] > 95 and st["spots"] == 0, st

# a dark blotch shows up as a spot, and a black frame has no foliage at all
spotted = green.copy()
cv2.circle(spotted, (60, 60), 8, (10, 20, 10), -1)
assert app.cv_stats(uri_of(spotted))["spots"] == 1, app.cv_stats(uri_of(spotted))
assert app.cv_stats(uri_of(np.zeros((60, 60, 3), np.uint8)))["foliage"] == 0
assert app.cv_stats("data:image/png;base64,not-base64!") == {}

# look(): the foliage box covers nearly the whole green frame, features are one fixed-length vector
_, box, feats = app.look(green)
assert box and box[2] > 0.9 and box[3] > 0.9, box
assert len(feats) == 638, len(feats)
assert app.look(np.zeros((60, 60, 3), np.uint8))[1:] == (None, None)

# vote(): nearest neighbours win, and a unanimous k means full confidence
X, y = np.float32([[0, 0], [0, .1], [5, 5]]), np.int32([0, 0, 1])
assert app.vote(X, y, np.float32([0, .05]), 2)[:2] == (0, 1.0)
assert app.vote(X, y, np.float32([5, 4.9]), 1)[:2] == (1, 1.0)
assert app.vote(X, y, np.float32([0, 0]), 3)[:2] == (0, 2 / 3)
assert round(app.vote(X, y, np.float32([0, 1.1]), 1)[2], 2) == 1.0  # nearest distance
assert app.predict(None) is None

# predict(): a frame further out than the training set's own spread is "no match", not a forced
# guess at full confidence
real_model = app.model
app.model = lambda: {"X": X, "y": y, "reject": np.float32(0.5), "labels": np.array(["near", "far"])}
try:
    # 2 of the 3 neighbours agree (67%), scaled down by how far out the frame sits
    assert app.predict(np.float32([0, .05])) == {"label": "near", "known": True, "confidence": 60}
    assert app.predict(np.float32([90, 90])) == {"label": "no match", "known": False,
                                                 "confidence": 0}
    # a model saved before the reject radius existed still answers instead of blowing up
    app.model = lambda: {"X": X, "y": y, "labels": np.array(["near", "far"])}
    assert app.predict(np.float32([90, 90]))["known"] is True
finally:
    app.model = real_model

# endpoint rejects non-images before spending an API call
c = app.app.test_client()
assert c.post("/analyze", json={"image": "hello"}).status_code == 400
# a frame with nothing plant-coloured in it is refused before the API call too
r = c.post("/analyze", json={"image": uri_of(np.zeros((60, 60, 3), np.uint8))})
assert r.status_code == 200 and r.get_json()["no_plant"], r.get_json()
# /detect is local-only: a box and pixel stats, no API key needed
r = c.post("/detect", json={"image": uri_of(green)})
assert r.status_code == 200 and r.get_json()["box"] and r.get_json()["cv"]["green"] > 95, r.get_json()
assert c.post("/detect", json={"image": "nope"}).status_code == 400

# page renders, with whatever scale reference/ currently holds
r = c.get("/")
assert r.status_code == 200 and "Reference scale" in r.get_data(as_text=True)
# providers(): only keys that are set, in order, model env overrides the default
for k in ("OPENROUTER_API_KEY", "GOOGLE_API_KEY"):
    os.environ.pop(k, None)
GROQ, OPENROUTER = app.PROVIDERS[1], app.PROVIDERS[2]
assert [u for u, _, _ in app.providers()] == [GROQ[0]], app.providers()
os.environ["OPENROUTER_API_KEY"] = "test2"
os.environ["OPENROUTER_MODEL"] = "some/vision-model"
assert [(k, m) for _, k, m in app.providers()] == [("test", GROQ[3]),
                                                   ("test2", "some/vision-model")], app.providers()

# a rate-limited first provider falls through to the second
calls = []


def fake_urlopen(req, timeout=None):
    calls.append(req.full_url)
    if "groq.com" in req.full_url:
        raise urllib.error.HTTPError(req.full_url, 429, "rate limited", {}, None)
    return io.BytesIO(b'{"choices":[{"message":{"content":"{\\"condition\\":\\"healthy\\"}"}}]}')


real, urllib.request.urlopen = urllib.request.urlopen, fake_urlopen
try:
    assert app.ask("data:image/png;base64,x", {"green": 12})["condition"] == "healthy"
    assert len(calls) == 2 and "openrouter" in calls[1], calls
finally:
    urllib.request.urlopen = real

# a 200 that carries an error body instead of choices is a failed provider, not a KeyError
def all_error(req, timeout=None):
    return io.BytesIO(b'{"error":{"message":"rate limited"}}')


real, urllib.request.urlopen = urllib.request.urlopen, all_error
try:
    try:
        app.ask("data:image/png;base64,x")
        raise AssertionError("expected the last provider's error to be raised")
    except OSError as e:
        assert "rate limited" in str(e), e
finally:
    urllib.request.urlopen = real

print("ok")
