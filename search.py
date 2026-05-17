import json
import logging
import numpy as np
from fastembed import TextEmbedding
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class SemanticSearch:
    def __init__(self, catalog_file="data/shl_product_catalog_new.json", embeddings_file="data/embeddings.npy"):
        self.catalog_minified = []

        # Load catalog
        try:
            with open(catalog_file, 'r', encoding='utf-8') as f:
                full_catalog = json.load(f, strict=False)
                for item in full_catalog:
                    self.catalog_minified.append({
                        "name": item.get("name", ""),
                        "link": item.get("link", ""),
                        "description": item.get("description", ""),
                        "keys": item.get("keys", []),
                        "job_levels": item.get("job_levels", []),
                        "duration": item.get("duration", ""),
                        "remote": item.get("remote", "")
                    })
            logger.info(f"Loaded {len(self.catalog_minified)} assessments into catalog.")
        except Exception as e:
            logger.error(f"Error loading catalog: {e}")

        # Load pre-computed embeddings
        self.catalog_embeddings = None
        try:
            self.catalog_embeddings = np.load(embeddings_file)
            logger.info(f"Loaded embeddings from {embeddings_file} with shape {self.catalog_embeddings.shape}")
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")

        # Use fastembed (ONNX-based, no PyTorch required, same model as embeddings.npy)
        logger.info("Initializing fastembed model 'BAAI/bge-small-en-v1.5'...")
        self.embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        logger.info("fastembed model initialized successfully.")

    def search(self, query: str, top_k: int = 30) -> list:
        if not query.strip():
            query = "assessment"

        if self.catalog_embeddings is not None:
            # Generate query embedding via fastembed (returns a generator)
            query_vec = np.array(list(self.embedding_model.embed([query])))
            similarities = cosine_similarity(query_vec, self.catalog_embeddings).flatten()
            top_indices = similarities.argsort()[-top_k:][::-1]
            return [self.catalog_minified[i] for i in top_indices]
        else:
            logger.warning("Embeddings not available. Returning first items as fallback.")
            return self.catalog_minified[:top_k]
