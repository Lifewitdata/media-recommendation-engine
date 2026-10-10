"""Generate synthetic datasets for media-recommendation-engine (seed-42).

Outputs (under data/):
  users.csv        - 5,000 users: user_id, age_band, region, device
  content.csv      - 1,500 catalog items: content_id, genre, tags, language, duration_min
  interactions.csv - ~200k events: user_id, content_id, event, event_ts
                     event in ('impression','play','completion','like')

Interaction funnel is affinity-driven: each user has a genre-affinity vector,
so play/completion/like probabilities rise with affinity -> real signal for
the recommender to learn. Honest synthetic data for a portfolio project.
"""
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
OUT = Path(__file__).resolve().parent
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------- users
N_USERS = 5_000
users = pd.DataFrame({
    "user_id": np.arange(1, N_USERS + 1),
    "age_band": rng.choice(["18-24", "25-34", "35-44", "45+"], N_USERS,
                           p=[0.28, 0.34, 0.22, 0.16]),
    "region": rng.choice(["Delhi", "Mumbai", "Bengaluru", "Chennai",
                          "Kolkata", "Hyderabad", "Jaipur", "Pune"], N_USERS),
    "device": rng.choice(["mobile", "tablet", "desktop"], N_USERS,
                         p=[0.62, 0.21, 0.17]),
})

# --------------------------------------------------------------- content
N_CONTENT = 1_500
GENRES = ["drama", "comedy", "news", "music", "documentary", "sports"]
TAGS = {
    "drama": ["crime", "thriller", "romance", "family", "mystery"],
    "comedy": ["standup", "sitcom", "sketch", "parody", "roast"],
    "news": ["politics", "daily", "business", "world", "weather"],
    "music": ["pop", "classical", "hiphop", "devotional", "rock"],
    "documentary": ["nature", "history", "science", "biography", "travel"],
    "sports": ["cricket", "football", "kabaddi", "tennis", "highlights"],
}
DUR = {"drama": (35, 60), "comedy": (20, 45), "news": (8, 20),
       "music": (3, 8), "documentary": (40, 70), "sports": (10, 30)}

genre_of = rng.choice(GENRES, N_CONTENT, p=[0.26, 0.22, 0.14, 0.16, 0.10, 0.12])
content = pd.DataFrame({
    "content_id": np.arange(1, N_CONTENT + 1),
    "genre": genre_of,
    "tags": [",".join(rng.choice(TAGS[g], size=2, replace=False)) for g in genre_of],
    "language": rng.choice(["hindi", "tamil", "english", "telugu", "bengali"],
                           N_CONTENT, p=[0.38, 0.16, 0.18, 0.14, 0.14]),
    "duration_min": [int(rng.integers(*DUR[g])) for g in genre_of],
})
# skewed popularity: a few hits, long tail
content["popularity_w"] = rng.lognormal(0.0, 1.0, N_CONTENT)

genre_idx = {g: i for i, g in enumerate(GENRES)}
content_by_genre = {g: content.loc[content.genre == g, "content_id"].to_numpy()
                    for g in GENRES}
pop_by_genre = {}
for g in GENRES:
    w = content.loc[content.genre == g, "popularity_w"].to_numpy()
    pop_by_genre[g] = w / w.sum()
dur_lookup = content.set_index("content_id")["duration_min"].to_dict()

# ---------------------------------------------------------- interactions
START = np.datetime64("2026-07-01")
END = np.datetime64("2026-09-30")
window_sec = int((END - START) / np.timedelta64(1, "s"))

affinity = rng.dirichlet(np.ones(len(GENRES)) * 2.0, size=N_USERS)  # user x genre

rows = []
for u in range(1, N_USERS + 1):
    n_impr = int(np.clip(rng.poisson(22), 5, 90))
    g_idx = rng.choice(len(GENRES), size=n_impr, p=affinity[u - 1])
    for gi in g_idx:
        g = GENRES[gi]
        c = int(rng.choice(content_by_genre[g], p=pop_by_genre[g]))
        a = affinity[u - 1, gi]
        t0 = START + np.timedelta64(int(rng.integers(0, window_sec)), "s")
        rows.append((u, c, "impression", t0))
        if rng.random() < 0.30 + 0.45 * a:                       # play
            t1 = t0 + np.timedelta64(int(rng.integers(60, 300)), "s")
            rows.append((u, c, "play", t1))
            if rng.random() < 0.30 + 0.45 * a:                   # completion
                t2 = t1 + np.timedelta64(int(dur_lookup[c] * 60 * rng.uniform(0.6, 1.0)), "s")
                rows.append((u, c, "completion", t2))
                if rng.random() < 0.10 + 0.30 * a:               # like
                    t3 = t2 + np.timedelta64(int(rng.integers(10, 120)), "s")
                    rows.append((u, c, "like", t3))

interactions = pd.DataFrame(rows, columns=["user_id", "content_id", "event", "event_ts"])
interactions = interactions.sort_values("event_ts").reset_index(drop=True)
content = content.drop(columns=["popularity_w"])

# ------------------------------------------------------------------ save
users.to_csv(OUT / "users.csv", index=False)
content.to_csv(OUT / "content.csv", index=False)
interactions.to_csv(OUT / "interactions.csv", index=False)

# -------------------------------------------------------------- validate
print("users:", users.shape, "| unique user_id:", users.user_id.nunique(),
      "| nulls:", int(users.isna().sum().sum()))
print("content:", content.shape, "| unique content_id:", content.content_id.nunique(),
      "| nulls:", int(content.isna().sum().sum()))
print("interactions:", interactions.shape, "| nulls:", int(interactions.isna().sum().sum()))
print("FK users ok:", interactions.user_id.isin(users.user_id).all(),
      "| FK content ok:", interactions.content_id.isin(content.content_id).all())
print("event mix:\n", interactions.event.value_counts().to_string())
print("play rate: %.3f | completion|play: %.3f | like|completion: %.3f" % (
    (interactions.event == "play").sum() / (interactions.event == "impression").sum(),
    (interactions.event == "completion").sum() / max((interactions.event == "play").sum(), 1),
    (interactions.event == "like").sum() / max((interactions.event == "completion").sum(), 1)))
print("date range:", interactions.event_ts.min(), "->", interactions.event_ts.max())
print("top genre by completions:\n",
      interactions[interactions.event == "completion"].merge(content[["content_id", "genre"]], on="content_id").genre.value_counts().head(3).to_string())
