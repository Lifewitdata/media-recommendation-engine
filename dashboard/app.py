import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

st.set_page_config(page_title="Media Recommender", layout="wide")
st.title("Media Content Recommendation Engine")

DATA = Path(__file__).resolve().parent.parent / "data"

@st.cache_data
def load():
    users = pd.read_csv(DATA / "users.csv")
    content = pd.read_csv(DATA / "content.csv")
    inter = pd.read_csv(DATA / "interactions.csv", parse_dates=["event_ts"])
    return users, content, inter

users, content, inter = load()
REL = {"completion", "like"}
W = {"impression": 1, "play": 2, "completion": 3, "like": 5}

@st.cache_data
def build_models():
    cut = inter["event_ts"].quantile(0.8)
    tr = inter[inter["event_ts"] <= cut].copy()
    tr["w"] = tr["event"].map(W)
    R = tr.pivot_table(index="user_id", columns="content_id", values="w",
                       aggfunc="sum", fill_value=0)
    R = R.reindex(columns=content["content_id"].to_numpy(), fill_value=0)
    sim = cosine_similarity(R.to_numpy().T)
    pop = tr[tr["event"] == "completion"]["content_id"].value_counts()
    content["text"] = (content["genre"] + " " + content["tags"].str.replace(",", " ")
                       + " " + content["language"])
    X = TfidfVectorizer().fit_transform(content["text"])
    return tr, R, sim, pop, X

tr, R, item_sim, pop_rank, X = build_models()
uidx = {u: i for i, u in enumerate(R.index)}
content_ids = content["content_id"].to_numpy()
cpos = {c: i for i, c in enumerate(content_ids)}
pop_vec = np.array([pop_rank.get(c, 0) for c in content_ids], dtype=float)

def norm(v):
    v = np.asarray(v, dtype=float)
    return (v - v.min()) / (v.max() - v.min() + 1e-9)

def _top(s, k, seen):
    s = np.asarray(s, dtype=float)
    order = np.argsort(s)[::-1]
    recs = [int(content_ids[j]) for j in order if int(content_ids[j]) not in seen][:k]
    return recs, np.round(s[[cpos[c] for c in recs]], 3)

def recommend(user_id, model, k):
    seen = set(tr[tr["user_id"] == user_id]["content_id"])
    cold = user_id not in uidx
    if model == "Popularity" or (cold and model != "Hybrid"):
        return _top(norm(pop_vec), k, seen)
    if model == "Item-item CF":
        return _top(norm(item_sim.T @ R.to_numpy()[uidx[user_id]]), k, seen)
    rows = tr[tr["user_id"] == user_id]
    if rows.empty:
        return _top(norm(pop_vec), k, seen)
    vecs = X[[cpos[c] for c in rows["content_id"]]]
    w = rows["w"].to_numpy()
    prof = np.asarray(vecs.T @ (w / w.sum())).reshape(1, -1)
    s_cb = norm(linear_kernel(prof, X).ravel())
    if model == "Content-based":
        return _top(s_cb, k, seen)
    s_cf = norm(item_sim.T @ R.to_numpy()[uidx[user_id]]) if not cold else 0
    s = 0.5 * s_cf + 0.3 * s_cb + 0.2 * norm(pop_vec)
    return _top(s, k, seen)

with st.sidebar:
    st.header("Controls")
    uid = st.selectbox("User", sorted(users["user_id"].unique())[:200])
    model = st.radio("Model", ["Hybrid", "Item-item CF", "Content-based", "Popularity"])
    k = st.slider("Top-N", 5, 20, 10)

uinfo = users[users["user_id"] == uid].iloc[0]
st.subheader(f"User {uid} — {uinfo['age_band']}, {uinfo['region']}, {uinfo['device']}")
hist = (tr[(tr["user_id"] == uid) & (tr["event"].isin(REL))]
        .merge(content[["content_id", "genre"]], on="content_id"))
if not hist.empty:
    st.write("Top genres by completions/likes:",
             ", ".join(hist["genre"].value_counts().head(3).index.tolist()))

recs, scores = recommend(uid, model, k)
out = (pd.DataFrame({"rank": range(1, len(recs) + 1), "content_id": recs,
                     "score": np.round(scores, 3)})
       .merge(content[["content_id", "genre", "tags", "language", "duration_min"]],
              on="content_id"))
st.subheader(f"Top-{k} recommendations ({model})")
st.dataframe(out, use_container_width=True)
