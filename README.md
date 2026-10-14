# MoodMusicX
Facial recognition to detect mood and suggest songs accordingly

```
camera / photo ─► OpenCV face detection ─► ViT expression model ─► valence/arousal ─► nearest songs
```

- **Face:** OpenCV Haar cascade, largest face, slightly padded crop
- **Mood:** [`trpakov/vit-face-expression`](https://huggingface.co/trpakov/vit-face-expression) (Apache-2.0, trained on FER2013): probabilities for angry, disgust, fear, happy, neutral, sad, surprise
- **Mood → music:** each emotion has a position on Russell's circumplex (valence × arousal); the face's position is the probability-weighted average. Songs are matched on Spotify's `valence` and `energy` features from the [spotify-tracks-dataset](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset) (BSD, ~114k tracks, downloaded on first run)
- **`--lift`:** the *iso principle* from music therapy: the playlist starts where your mood is and moves step by step toward a content, upbeat target, instead of jumping straight to happy songs

```sh
nix-shell                          # or: pip install -r requirements.txt
python moodmusic.py photo.jpg
python moodmusic.py photo.jpg --lift -n 12 --genre indie --genre acoustic
python moodmusic.py --webcam       # space = playlist for the current face, q = quit
python moodmusic.py --mood sad --lift
python -m pytest tests
```

Limits: FER2013 is posed, low-resolution and skewed toward some demographics, so expression ≠ feeling; treat the mood as a hint. Spotify's audio-features API is closed to new apps, so the catalog is a static snapshot.
