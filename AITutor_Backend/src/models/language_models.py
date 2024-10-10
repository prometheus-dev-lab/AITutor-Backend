import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple
from tqdm.auto import tqdm
from math import ceil
from collections import OrderedDict
import numpy as np


class LRCache:
    """Lowest-Ranked Cache"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: str) -> Tuple[torch.Tensor, float]:
        if key not in self.cache:
            return None
        return self.cache[key]

    def put(self, key: str, value: Tuple[torch.Tensor, float]):
        if key in self.cache:
            del self.cache[key]
        elif len(self.cache) >= self.capacity:
            # Remove the item with the lowest score
            lowest_score_key = min(self.cache, key=lambda k: self.cache[k][1])
            del self.cache[lowest_score_key]
        self.cache[key] = value

    def update_score(self, key: str, new_score: float):
        if key in self.cache:
            embedding, _ = self.cache[key]
            self.cache[key] = (embedding, new_score)


class Ranker:
    def __init__(
        self,
        model_name="microsoft/codebert-base",
        batch_size=8,
        device=None,
        cache_size=1000,
    ):
        self.batch_size = batch_size
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device).half()  # Convert model to FP16
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.cache = LRCache(cache_size)

    @torch.inference_mode()
    def encode(self, texts: List[str]) -> torch.Tensor:
        embeddings = []
        for batch in tqdm(
            self._chunks(texts, self.batch_size),
            desc="Encoding",
            total=ceil(len(texts) / self.batch_size),
        ):
            tokens = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.tokenizer.model_max_length,
            ).to(self.device)

            output = self.model(**tokens)
            batch_embeddings = (
                output.last_hidden_state[:, 0, :].cpu().half()
            )  # Use [CLS] token embedding
            embeddings.append(batch_embeddings)

        return torch.cat(embeddings, dim=0)

    @torch.inference_mode()
    def rank(self, query: str, texts: List[str]) -> List[float]:
        query_embedding = self.encode([query])[0]

        scores = []
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            cached_result = self.cache.get(text)
            if cached_result is not None:
                embedding, score = cached_result
                scores.append(score)
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        if uncached_texts:
            uncached_embeddings = self.encode(uncached_texts)

            for i, embedding in zip(uncached_indices, uncached_embeddings):
                score = self.cosine_similarity(query_embedding, embedding)
                scores.insert(i, score)
                self.cache.put(texts[i], (embedding, score))

        # Update scores for cached items
        for i, text in enumerate(texts):
            if text in self.cache.cache:
                self.cache.update_score(text, scores[i])

        return scores

    @staticmethod
    def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(
            torch.nn.functional.cosine_similarity(
                a.unsqueeze(0), b.unsqueeze(0), dim=1
            ).item()
        )

    @staticmethod
    def _chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i : i + n]
