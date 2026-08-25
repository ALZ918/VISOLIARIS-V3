"""One-shot project setup: run `python3 bootstrap.py` and answer three questions.

Asks for a Google AI Studio key, an optional OpenRouter key, and whether to download the
dataset (~290 MB), then extracts, sorts and trains. Say no to the download to reuse
whatever is already in data/.

Writes .env with the keys you type, plus data/metadata.csv and data/model.npz (all
gitignored), and fills reference/<n_label>/ with a few sample images per class. Class
folders are ordered healthy-first, then alphabetically; rename them if that is not the severity order you want.
"""
import csv, os, random, re, shutil, subprocess, sys, zipfile
from pathlib import Path

import cv2
import numpy as np

import app

# whole potted plants, healthy vs wilted - not leaf close-ups, so a phone photo of a plant
# on a shelf lands inside the training distribution instead of coming back "no match"
URL = ("https://www.kaggle.com/api/v1/datasets/download/"
       "russellchan/healthy-and-wilted-houseplant-images")
ROOT = Path(__file__).parent
DATA, REF = ROOT / "data", ROOT / "reference"
ZIP, RAW = DATA / "houseplants.zip", DATA / "houseplants"
ENV = ROOT / ".env"
META, SAMPLES, IMG = DATA / "metadata.csv", 3, (".jpg", ".jpeg", ".png", ".webp")
REF_MAX = 640  # longest side of the copies dropped into reference/


def ask(prompt, current=""):
    """Blank answer keeps whatever is already in .env, so re-running is safe."""
    shown = " [keep existing]" if current else ""
    return input("%s%s: " % (prompt, shown)).strip() or current


def setup():
    """Ask for the API keys and write them to .env (gitignored)."""
    # importing app already pulled any existing .env into os.environ
    keys = {"GOOGLE_API_KEY": ask("Google AI Studio API key (aistudio.google.com/apikey)",
                                  os.environ.get("GOOGLE_API_KEY", "")),
            "OPENROUTER_API_KEY": ask("OpenRouter API key (optional, press enter to skip)",
                                      os.environ.get("OPENROUTER_API_KEY", ""))}
    if not keys["GOOGLE_API_KEY"] and not keys["OPENROUTER_API_KEY"]:
        sys.exit("need at least one key - grading calls a vision model")
    # merge into whatever else is already in .env instead of clobbering it
    env = dict(l.split("=", 1) for l in ENV.read_text().splitlines()
               if "=" in l and not l.startswith("#")) if ENV.exists() else {}
    env.update({k: v for k, v in keys.items() if v})
    ENV.write_text("".join("%s=%s\n" % kv for kv in env.items()))
    os.environ.update(env)
    print("wrote .env\n")


def images(folder):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG)


def fetch():
    DATA.mkdir(exist_ok=True)
    if not ZIP.exists():
        # curl, not urllib: it resumes a half-finished download and draws its own progress
        subprocess.run(["curl", "-L", "-C", "-", "-o", str(ZIP), URL], check=True)


def extract():
    """Unzip once. Runs even when you skip the download, so a hand-placed zip still works."""
    if not RAW.exists():
        print("extracting…")
        with zipfile.ZipFile(ZIP) as z:
            z.extractall(RAW)


def classes():
    """Every folder in the extracted tree that holds a decent pile of images = one class."""
    found = {d.name: d for d in RAW.rglob("*")
             if d.is_dir() and "__MACOSX" not in d.parts and len(images(d)) >= 10}
    if not found:
        sys.exit("no class folders found under %s" % RAW)
    # healthy first, the rest alphabetically - rename the reference folders to reorder the scale
    return sorted(found.items(), key=lambda kv: ("healthy" not in kv[0].lower(), kv[0].lower()))


def build_reference(found, rows, X, y):
    """Copy the most typical images of each class into reference/ as the few-shot anchors.

    Typical = closest to the class mean in feature space. Taking the first few files instead
    picks up whatever the dataset happens to have mislabelled, and one bad anchor skews every
    grade the vision model gives.
    """
    for i, (name, _) in enumerate(found):
        label = re.sub(r"\W+", "_", name.strip().lower()).strip("_")
        dest = REF / ("%d_%s" % (i + 1, label))
        shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True)
        mine = np.flatnonzero(y == i)
        d = np.linalg.norm(X[mine] - X[mine].mean(0), axis=1)
        for j in mine[np.argsort(d)[:SAMPLES]]:
            src = ROOT / rows[j]["path"]
            # downscaled copies: the originals are ~3 MB phone shots, and every reference image
            # is base64'd into each prompt
            img = cv2.imread(str(src))
            if img is None:
                continue
            s = REF_MAX / max(img.shape[:2])
            if s < 1:
                img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(dest / (src.stem.replace(" ", "_") + ".jpg")), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 80])
        print("reference/%s  <- %s" % (dest.name, name))


def scan(found):
    """One pass over every image: a metadata row and a feature vector each."""
    rows, X, y = [], [], []
    for i, (name, folder) in enumerate(found):
        for path in images(folder):
            img = cv2.imread(str(path))
            if img is None:
                continue
            stats, _, feats = app.look(img)
            if feats is None:  # nothing plant-coloured; useless as a training sample
                continue
            rows.append(dict(stats, path=str(path.relative_to(ROOT)), label=name, class_index=i))
            X.append(feats)
            y.append(i)
        print("  %-20s %d images" % (name, y.count(i)))
    return rows, np.array(X, np.float32), np.array(y, np.int32)


def train(X, y, found):
    """No training step to speak of - k-NN is the training set. Report held-out accuracy, save."""
    idx = list(range(len(y)))
    random.Random(0).shuffle(idx)
    cut = int(len(idx) * 0.8)
    Xt, yt, Xv, yv = X[idx[:cut]], y[idx[:cut]], X[idx[cut:]], y[idx[cut:]]
    pred = np.array([app.vote(Xt, yt, f)[0] for f in Xv])
    print("\nheld-out accuracy: %.1f%% on %d images" % (100 * float((pred == yv).mean()), len(yv)))
    for i, (name, _) in enumerate(found):
        m = yv == i
        if m.any():
            print("  %-20s %.1f%%" % (name, 100 * float((pred[m] == i).mean())))
    # how far apart the training images themselves sit, so anything further out can be called
    # "no match" instead of being forced into the nearest class (see app.predict)
    nn = np.array([np.partition(np.linalg.norm(X - f, axis=1), 1)[1] for f in X])
    reject = float(np.percentile(nn, 99))
    print("reject distance: %.3f (median neighbour %.3f)" % (reject, float(np.median(nn))))
    # ship every image: k-NN gets strictly better with more of them, and the file is small
    np.savez(app.MODEL_FILE, X=X, y=y, reject=reject, labels=np.array([n for n, _ in found]))


if __name__ == "__main__":
    setup()
    if not input("Download the dataset? ~290 MB (Y/n): ").strip().lower().startswith("n"):
        fetch()
    extract()
    found = classes()
    print("\n%d classes: %s\n" % (len(found), ", ".join(n for n, _ in found)))
    print("\nreading images…")
    rows, X, y = scan(found)
    build_reference(found, rows, X, y)
    with META.open("w", newline="") as f:
        w = csv.DictWriter(f, ["path", "label", "class_index", "foliage", "green", "yellow",
                               "brown", "spots"])
        w.writeheader()
        w.writerows(rows)
    train(X, y, found)
    print("\nwrote %s and %s — now run: python3 app.py" % (META.name, app.MODEL_FILE.name))
