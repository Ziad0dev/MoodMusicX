import math

from moodmusic import CIRCUMPLEX, TARGET, mood_point, playlist


def grid_tracks():
    return [{"id": f"{v}-{e}", "name": f"t{v}{e}", "artists": "a", "genre": "pop" if v % 2 else "jazz",
             "popularity": 60, "valence": v / 10, "energy": e / 10, "tempo": 120.0}
            for v in range(11) for e in range(11)]


def test_mood_point_is_weighted_average():
    assert mood_point({"happy": 1.0}) == CIRCUMPLEX["happy"]
    v, a = mood_point({"happy": 0.5, "sad": 0.5})
    assert math.isclose(v, (0.9 + 0.15) / 2) and math.isclose(a, (0.7 + 0.25) / 2)


def test_matching_playlist_stays_near_the_mood():
    songs = playlist(grid_tracks(), CIRCUMPLEX["sad"], 5, lift=False, seed=0)
    assert all(s["valence"] <= 0.4 and s["energy"] <= 0.5 for s in songs)


def test_lift_playlist_moves_toward_target():
    songs = playlist(grid_tracks(), CIRCUMPLEX["sad"], 8, lift=True, seed=0)
    assert len({s["id"] for s in songs}) == 8
    assert songs[0]["valence"] < 0.35
    assert abs(songs[-1]["valence"] - TARGET[0]) <= 0.15


def test_genre_filter():
    songs = playlist(grid_tracks(), (0.5, 0.5), 5, lift=False, genres={"jazz"}, seed=0)
    assert {s["genre"] for s in songs} == {"jazz"}
