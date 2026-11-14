"""Local embedding adapter for google/embeddinggemma-2 (Sentence Transformers).

No mock or random fallback: if the model cannot be loaded, an exception is raised.
"""
from __future__ import annotations

import math
from typing import Protocol


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class MyEmbedder:
    """EmbeddingGemma 2, text-only, run locally.

    - Documents use the model's "document" prompt ("title: none | text: ...").
    - Queries use the "query" prompt ("task: search result | query: ...").
    - Vision/audio encoders are not loaded (text-only, ~270M params).
    - float32 (the model card warns float16 produces NaN); L2-normalized output,
      so cosine distance in pgvector equals 1 - dot product.
    """

    DOC_PROMPT = "document"
    QUERY_PROMPT = "query"

    def __init__(self, model_name: str, revision: str | None = None,
                 device: str | None = None, batch_size: int = 16) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.revision = revision
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(
            model_name,
            revision=revision,
            device=self.device,
            model_kwargs={"dtype": torch.float32},
            config_kwargs={"vision_config": None, "audio_config": None},
        )
        prompts = getattr(self.model, "prompts", {}) or {}
        for name in (self.DOC_PROMPT, self.QUERY_PROMPT):
            if name not in prompts:
                raise RuntimeError(f"Model {model_name} has no prompt named {name!r}")
        self.prompts = {name: prompts[name] for name in (self.DOC_PROMPT, self.QUERY_PROMPT)}
        probe = self._encode(["kiểm tra"], self.QUERY_PROMPT)
        self.dimensions = len(probe[0])

    def _encode(self, texts: list[str], prompt_name: str) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            prompt_name=prompt_name,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        result = [[float(x) for x in row] for row in vectors]
        for row in result:
            if any(math.isnan(x) or math.isinf(x) for x in row):
                raise RuntimeError("Embedding contains NaN/Inf; check dtype/device")
        return result

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._encode(texts, self.DOC_PROMPT)

    def embed_query(self, question: str) -> list[float]:
        return self._encode([question], self.QUERY_PROMPT)[0]

    def count_tokens(self, texts: list[str]) -> list[int]:
        """Token count of each text with the model tokenizer (without prompt)."""
        tokenizer = self.model.tokenizer
        tokenizer = getattr(tokenizer, "tokenizer", tokenizer)  # processor -> tokenizer
        return [len(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in texts]

    def describe(self) -> dict:
        import sentence_transformers
        import torch
        import transformers

        return {
            "provider": "local (Sentence Transformers)",
            "model": self.model_name,
            "revision": self.revision,
            "dimensions": self.dimensions,
            "device": self.device,
            "dtype": "float32",
            "normalized": True,
            "document_prompt": self.prompts[self.DOC_PROMPT],
            "query_prompt": self.prompts[self.QUERY_PROMPT],
            "max_seq_length": getattr(self.model, "max_seq_length", None),
            "versions": {
                "sentence_transformers": sentence_transformers.__version__,
                "transformers": transformers.__version__,
                "torch": torch.__version__,
            },
            "cost_usd": 0.0,
        }
