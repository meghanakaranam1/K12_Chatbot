# src/data/processor.py
"""
Data processing and embeddings management
"""
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Optional, Dict
import sys, os

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
from data.time_processor import time_processor

class DataProcessor:
    """Handles data loading, processing, and embeddings"""

    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.embedder: Optional[SentenceTransformer] = None
        self.embeddings: Optional[np.ndarray] = None
        self.index: Optional[faiss.IndexFlatIP] = None  # cosine sim
        self.topic_texts: Optional[List[str]] = None
        self.topic_vectors: Optional[np.ndarray] = None  # normalized
        self.duration_cache: Optional[Dict[int, Optional[int]]] = None

    def load_dataframe(self, path: str) -> pd.DataFrame:
        """Load CSV with robust encoding detection"""
        encodings = ["utf-8", "utf-8-sig", "latin1", "ISO-8859-1", "cp1252", "utf-16"]
        for enc in encodings:
            try:
                df = pd.read_csv(path, encoding=enc, dtype=str)
                df.fillna("", inplace=True)
                return df
            except Exception:
                continue
        raise RuntimeError(f"Could not read CSV from {path}")

    def safe_get(self, row: pd.Series, col: str, default: str = "") -> str:
        """Safely get value from pandas row"""
        return row[col] if col in row and pd.notna(row[col]) else default

    def build_combined_field(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build combined searchable field from multiple columns"""
        existing_cols = [c for c in config.search_columns if c in df.columns]
        df["combined"] = df[existing_cols].agg(" | ".join, axis=1)
        return df

    def initialize(self, data_path: str):
        """Initialize embeddings + FAISS index"""
        print(f"🔍 [DEBUG] data_processor: Initializing with {data_path}")
        self.df = self.load_dataframe(data_path)
        print(f"✅ Loaded {len(self.df)} rows")
        self.df = self.build_combined_field(self.df)

        print("🔍 Loading embedder...")
        self.embedder = SentenceTransformer(config.embed_model)

        print("🔍 Creating embeddings (may take 1–2 min first time)...")
        self.embeddings = self.embedder.encode(
            self.df["combined"].tolist(),
            batch_size=128,
            show_progress_bar=True,
            normalize_embeddings=True
        ).astype("float32")
        print(f"✅ Embeddings shape: {self.embeddings.shape}")

        print("🔍 Building FAISS index (cosine similarity)...")
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)
        print("✅ FAISS index ready")

        # NEW: Precompute topic texts and embeddings for fast topic inference
        print("🔍 [DEBUG] data_processor: Building topic texts for inference")
        titles, small = [], []
        for i in range(len(self.df)):
            row = self.get_row(i)
            title = self.safe_get(row, "Strategic Action")
            desc = self.safe_get(row, "Short Description")
            if not title and not desc:
                titles.append("")
            txt = f"{title} {desc}".strip()
            small.append(txt if txt else title)
        self.topic_texts = small

        # Embed topic texts ONCE for topic inference (reuses the same model)
        print("🔍 [DEBUG] data_processor: Creating topic embeddings")
        self.topic_vectors = self.embedder.encode(
            self.topic_texts, show_progress_bar=False,
            batch_size=128, normalize_embeddings=True
        )
        print(f"✅ Topic vectors shape: {self.topic_vectors.shape}")

        # Build duration cache for fast time filtering
        print("🔍 [DEBUG] data_processor: Building duration cache")
        self.duration_cache = time_processor.build_duration_cache(self.df)
        # Keep a copy in the time processor for fast lookup by other modules
        time_processor.set_duration_cache(self.duration_cache)
        print(f"✅ Duration cache built with {len([k for k, v in self.duration_cache.items() if v is not None])} activities with known durations")

    def search(self, query: str, top_k: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """Semantic search over activities"""
        if self.index is None:
            raise RuntimeError("FAISS index not initialized. Call initialize() first.")

        if top_k is None:
            top_k = config.top_k

        print(f"🔍 Searching: {query[:60]}")
        q_emb = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        distances, indices = self.index.search(q_emb, top_k)
        print(f"✅ Returned {len(indices[0])} results (top_k={top_k})")
        return distances, indices

    def get_row(self, index: int) -> pd.Series:
        return self.df.iloc[index]

    def get_rows(self, indices: List[int]) -> List[pd.Series]:
        return [self.df.iloc[i] for i in indices if 0 <= i < len(self.df)]

# Global instance
data_processor = DataProcessor()
