#!/usr/bin/env python3
"""MoodMusicX: read the mood on a face and suggest songs for it.

    python moodmusic.py photo.jpg                # mood + 10 songs that match it
    python moodmusic.py photo.jpg --lift         # a playlist that starts at your mood and lifts it
    python moodmusic.py --webcam                 # live, press space to get a playlist, q to quit
    python moodmusic.py --mood sad --lift        # skip the camera

Pipeline: OpenCV face detection -> ViT facial-expression model
(trpakov/vit-face-expression, FER2013 emotions) -> a point on Russell's
valence/arousal circumplex -> nearest tracks by Spotify valence/energy
(maharshipandya/spotify-tracks-dataset).
"""
import argparse
import csv
import math
import random
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent
TRACKS = ROOT / "data" / "spotify_tracks.csv"
TRACKS_URL = "https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/main/dataset.csv"
EMOTION_MODEL = "trpakov/vit-face-expression"

# Russell's circumplex model: each emotion as (valence, arousal), both in 0..1
CIRCUMPLEX = {
    "happy": (0.90, 0.70),
    "surprise": (0.70, 0.85),
    "neutral": (0.50, 0.40),
    "sad": (0.15, 0.25),
    "fear": (0.20, 0.80),
    "angry": (0.15, 0.85),
    "disgust": (0.15, 0.60),
}
# where the "lift" playlist should end up: content and fairly upbeat
TARGET = (0.80, 0.65)


def mood_point(emotions):
    """Probability-weighted position of a face on the valence/arousal plane."""
    total = sum(emotions.values()) or 1.0
    v = sum(p * CIRCUMPLEX[e][0] for e, p in emotions.items() if e in CIRCUMPLEX) / total
    a = sum(p * CIRCUMPLEX[e][1] for e, p in emotions.items() if e in CIRCUMPLEX) / total
    return v, a


# ---------------------------------------------------------------- vision

def find_cascade(cv2, name="haarcascade_frontalface_default.xml"):
    """pip's opencv exposes cv2.data; distro builds keep the XML under share/opencv4."""
    if hasattr(cv2, "data"):
        return Path(cv2.data.haarcascades) / name
    here = Path(cv2.__file__).resolve()
    for root in [*here.parents, Path(sys.prefix)]:
        for candidate in root.glob(f"share/opencv*/haarcascades/{name}"):
            return candidate
    raise SystemExit(f"could not find {name}; install opencv-python")


class MoodReader:
    def __init__(self):
        import cv2
        from transformers import pipeline

        self.cv2 = cv2
        self.faces = cv2.CascadeClassifier(str(find_cascade(cv2)))
        self.classifier = pipeline("image-classification", model=EMOTION_MODEL, top_k=None)

    def read_frame(self, bgr):
        """Returns ({emotion: probability}, face box) for the largest face, or (None, None)."""
        from PIL import Image

        gray = self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2GRAY)
        found = self.faces.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if len(found) == 0:
            return None, None
        x, y, w, h = max(found, key=lambda f: f[2] * f[3])
        pad = int(0.15 * w)  # FER crops include a bit of forehead and chin
        crop = bgr[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]
        image = Image.fromarray(self.cv2.cvtColor(crop, self.cv2.COLOR_BGR2RGB))
        scores = {r["label"].lower(): r["score"] for r in self.classifier(image)}
        return scores, (x, y, w, h)

    def read_file(self, path):
        bgr = self.cv2.imread(str(path))
        if bgr is None:
            raise SystemExit(f"could not read image {path}")
        return self.read_frame(bgr)


# ---------------------------------------------------------------- music

def load_tracks(min_popularity=45):
    if not TRACKS.exists():
        TRACKS.parent.mkdir(exist_ok=True)
        print(f"downloading track features from {TRACKS_URL}", file=sys.stderr)
        urllib.request.urlretrieve(TRACKS_URL, TRACKS)
    tracks, seen = [], set()
    with TRACKS.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                pop = int(row["popularity"])
                if pop < min_popularity:
                    continue
                key = (row["track_name"].lower(), row["artists"].lower())
                if key in seen:  # the dataset lists a track once per genre
                    continue
                seen.add(key)
                tracks.append({
                    "name": row["track_name"], "artists": row["artists"].replace(";", ", "),
                    "genre": row["track_genre"], "popularity": pop,
                    "valence": float(row["valence"]), "energy": float(row["energy"]),
                    "tempo": float(row["tempo"]), "id": row["track_id"],
                })
            except (KeyError, ValueError):
                continue
    return tracks


def nearest(tracks, point, n, exclude=(), genres=None, rng=None):
    v, a = point
    pool = [t for t in tracks if t["id"] not in exclude and (not genres or t["genre"] in genres)]
    # small popularity bonus so suggestions are recognizable, then a random pick among the closest
    scored = sorted(pool, key=lambda t: math.hypot(t["valence"] - v, t["energy"] - a) - 0.0005 * t["popularity"])
    shortlist = scored[:max(n * 4, 20)]
    rng = rng or random.Random()
    return sorted(rng.sample(shortlist, min(n, len(shortlist))), key=lambda t: math.hypot(t["valence"] - v, t["energy"] - a))


def playlist(tracks, start, n, lift, genres=None, seed=None):
    rng = random.Random(seed)
    if not lift:
        return nearest(tracks, start, n, genres=genres, rng=rng)
    # iso principle: meet the listener where they are, then move step by step to the target
    songs, used = [], set()
    for i in range(n):
        t = i / max(1, n - 1)
        point = (start[0] + (TARGET[0] - start[0]) * t, start[1] + (TARGET[1] - start[1]) * t)
        pick = nearest(tracks, point, 1, exclude=used, genres=genres, rng=rng)
        if pick:
            songs.append(pick[0])
            used.add(pick[0]["id"])
    return songs


def describe(point):
    v, a = point
    tone = "positive" if v > 0.6 else "negative" if v < 0.4 else "neutral"
    energy = "high-energy" if a > 0.6 else "calm" if a < 0.4 else "moderate-energy"
    return f"{tone}, {energy} (valence {v:.2f}, arousal {a:.2f})"


def print_playlist(songs):
    for i, s in enumerate(songs, 1):
        print(f"{i:2d}. {s['name'][:40]:40s}  {s['artists'][:28]:28s}  {s['genre']:12s}  "
              f"valence {s['valence']:.2f}  energy {s['energy']:.2f}  https://open.spotify.com/track/{s['id']}")


def report(emotions, args, tracks):
    top = sorted(emotions.items(), key=lambda kv: -kv[1])
    print("mood: " + ", ".join(f"{e} {p:.0%}" for e, p in top[:3]))
    point = mood_point(emotions)
    print(f"      {describe(point)}\n")
    print_playlist(playlist(tracks, point, args.n, args.lift, set(args.genre) if args.genre else None, args.seed))


def webcam(args, tracks):
    import cv2

    reader = MoodReader()
    cam = cv2.VideoCapture(args.camera)
    emotions = None
    print("space = playlist for the current mood, q = quit")
    while True:
        ok, frame = cam.read()
        if not ok:
            raise SystemExit("camera not available")
        emotions, box = reader.read_frame(frame)
        if box is not None:
            x, y, w, h = box
            label = max(emotions, key=emotions.get)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 200, 120), 2)
            cv2.putText(frame, f"{label} {emotions[label]:.0%}", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 200, 120), 2)
        cv2.imshow("MoodMusicX", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" ") and emotions:
            print()
            report(emotions, args, tracks)
    cam.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", nargs="?")
    parser.add_argument("--webcam", action="store_true")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--mood", choices=sorted(CIRCUMPLEX), help="use this mood instead of reading a face")
    parser.add_argument("--lift", action="store_true", help="playlist that moves from your mood toward a positive one")
    parser.add_argument("-n", type=int, default=10)
    parser.add_argument("--genre", action="append", help="restrict to a Spotify genre (repeatable), e.g. --genre indie")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    tracks = load_tracks()
    if args.webcam:
        webcam(args, tracks)
    elif args.mood:
        report({args.mood: 1.0}, args, tracks)
    elif args.image:
        emotions, _ = MoodReader().read_file(args.image)
        if emotions is None:
            raise SystemExit("no face found in the image")
        report(emotions, args, tracks)
    else:
        parser.error("give an image, --webcam or --mood")


if __name__ == "__main__":
    main()
