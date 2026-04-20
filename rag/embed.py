from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class EmbeddingArtifacts:
    model_id: str
    vectorizer: TfidfVectorizer
    matrix: np.ndarray


def fit_tfidf_embeddings(texts: list[str]) -> EmbeddingArtifacts:
    if not texts:
        raise ValueError("Cannot embed an empty text list")

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts).astype("float32").toarray()
    return EmbeddingArtifacts(model_id="tfidf-ngram-1-2", vectorizer=vectorizer, matrix=matrix)


def embed_query(vectorizer: TfidfVectorizer, query: str) -> np.ndarray:
    vector = vectorizer.transform([query]).astype("float32").toarray()[0]
    return vector

