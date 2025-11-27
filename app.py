# app.py

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, Any, List
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib

from sentence_transformers import SentenceTransformer

# ---------- Setup ----------

app = FastAPI(title="Journal Mind Map API (MVP)")

DATA_DIR = Path("data")

DF_CLUSTERS_PATH = DATA_DIR / "cluster_table.parquet"
DF_CHUNKS_PATH = DATA_DIR / "chunks_with_clusters.parquet"
PCA_PATH = DATA_DIR / "pca_model.joblib"
CENTROIDS_PATH = DATA_DIR / "cluster_centroids.npy"
CLUSTER_IDS_PATH = DATA_DIR / "cluster_ids_ordered.npy"  # optional

# Load precomputed artifacts
df_clusters = pd.read_parquet(DF_CLUSTERS_PATH)
df_chunks = pd.read_parquet(DF_CHUNKS_PATH)
pca = joblib.load(PCA_PATH)
centroids = np.load(CENTROIDS_PATH)

# If you saved ordered_ids, use them; otherwise, take from df_clusters
if CLUSTER_IDS_PATH.exists():
    cluster_ids_ordered = np.load(CLUSTER_IDS_PATH).tolist()
else:
    cluster_ids_ordered = df_clusters["cluster_id"].tolist()

# Ensure ordering consistency between centroids and df_clusters
# (Assuming they were saved in the same order originally)
# Otherwise, you'd need to reorder accordingly.

# Choose cluster column name from your pipeline
CLUSTER_COL = "cluster_kmeans"  # or "cluster_hdbscan"

# Load embedding model (same one used in your notebooks)
embed_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


# ---------- Request models ----------

class AnalyzeRequest(BaseModel):
    text: str


# ---------- Endpoints ----------

@app.get("/mindmap")
def get_mindmap() -> Dict[str, List[Dict[str, Any]]]:
    """
    Minimal payload for drawing the emotional mind map.
    """
    nodes: List[Dict[str, Any]] = []

    for _, row in df_clusters.iterrows():
        nodes.append(
            {
                "cluster_id": int(row["cluster_id"]),
                "valence": float(row["valence"]),
                "arousal": float(row["arousal"]),
                "size": int(row["size"]),
                "emotion_label": row.get("emotion_label", ""),
                "label": row.get("medoid_text_short", ""),
            }
        )

    return {"nodes": nodes}


@app.get("/clusters/{cluster_id}/chunks")
def get_cluster_chunks(
    cluster_id: int,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> Dict[str, Any]:
    """
    Return chunks belonging to a cluster.
    Frontend calls this when a node is clicked.
    """
    mask = df_chunks[CLUSTER_COL] == cluster_id
    subset = df_chunks.loc[mask]

    if subset.empty:
        raise HTTPException(status_code=404, detail="Cluster not found or empty")

    total = len(subset)
    page = subset.iloc[offset : offset + limit]

    items: List[Dict[str, Any]] = []
    for _, row in page.iterrows():
        items.append(
            {
                "entry_id": int(row["entry_id"]),
                "chunk_text": row["chunk_text"],
                "timestamp": str(row["timestamp"]) if "timestamp" in row and row["timestamp"] is not None else None,
            }
        )

    return {
        "cluster_id": cluster_id,
        "total": total,
        "limit": limit,
        "offset": offset,
        "items": items,
    }


@app.post("/analyze")
def analyze_text(payload: AnalyzeRequest) -> Dict[str, Any]:
    """
    Accepts a piece of text, embeds it, projects it via PCA,
    and assigns it to the nearest existing cluster.
    """

    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    # 1) Embed
    emb = embed_model.encode([text], convert_to_numpy=True)  # shape: (1, dim)

    # 2) Project with same PCA as training
    emb_pca = pca.transform(emb)  # shape: (1, pca_dim)

    # 3) Compute similarity to all centroids
    sims = cosine_similarity(emb_pca, centroids).ravel()  # shape: (num_clusters,)
    best_idx = int(sims.argmax())
    best_sim = float(sims[best_idx])
    best_cluster_id = int(cluster_ids_ordered[best_idx])

    # 4) Look up cluster info
    row = df_clusters.loc[df_clusters["cluster_id"] == best_cluster_id]
    if row.empty:
        raise HTTPException(status_code=500, detail="Best cluster not found in table")

    row = row.iloc[0]

    return {
        "cluster_id": best_cluster_id,
        "similarity": best_sim,
        "valence": float(row["valence"]),
        "arousal": float(row["arousal"]),
        "emotion_label": row.get("emotion_label", ""),
        "medoid_text": row.get("medoid_text_full", ""),
        "cluster_size": int(row["size"]),
    }
