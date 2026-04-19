import hashlib
import json
import os
import pickle
import time
from dataclasses import dataclass
from heapq import nlargest
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


TokenKey = str

# Bump this whenever the HyDE prompt format changes so cached generations
# are rebuilt against the latest prompt.
_HYDE_PROMPT_VERSION = "v1-legal"

_HYDE_SYSTEM = """\
你是一位专业的中国法律文本生成助手。
你的任务是：根据用户的法律咨询问题，生成 {n} 段「假想相关法律条文文本」。

每段格式要求（严格遵守）：
第一行：可能适用的中国法律/法规/司法解释的全称（例如《中华人民共和国民法典》），不要写条号
第二行起：以法条条文风格描述相关法律规定（150-300字），涵盖构成要件、法律后果、适用范围等
最后一行：【关键词】法律术语1、法律术语2、…（8-15个术语）

严格禁止：
- 禁止写出任何具体条号（如"第X条"、"第一百零七条"等），一律用"依据相关规定"代替
- 禁止编造当事人姓名、具体案号、虚构金额
- {n} 段之间用单独一行"---"分隔"""

_HYDE_USER = """\
法律咨询问题：{question}

请从不同法律视角生成 {n} 段假想法律条文文本（严格按上述格式，用"---"分隔）："""


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


def _minmax_norm(scores: Dict[int, float]) -> Dict[int, float]:
    if not scores:
        return {}
    vals = np.fromiter(scores.values(), dtype=np.float32)
    vmin = float(vals.min())
    vmax = float(vals.max())
    if vmax - vmin < 1e-12:
        return {k: 0.0 for k in scores.keys()}
    return {k: (float(v) - vmin) / (vmax - vmin) for k, v in scores.items()}


def _rrf_fuse(
    dense_scores: Dict[int, float],
    sparse_scores: Dict[int, float],
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
    rrf_k: int = 60,
) -> Dict[int, float]:
    fused: Dict[int, float] = {}

    def _add(scores: Dict[int, float], weight: float):
        if not scores or weight == 0:
            return
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for rank, (doc_id, _) in enumerate(ranked, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + float(weight) / float(rrf_k + rank)

    _add(dense_scores, dense_weight)
    _add(sparse_scores, sparse_weight)
    return fused


def _as_token_weight_dict(x) -> Dict[TokenKey, float]:
    if x is None:
        return {}
    if isinstance(x, dict):
        out: Dict[TokenKey, float] = {}
        for key, value in x.items():
            try:
                out[str(key)] = float(value)
            except Exception:
                continue
        return out
    if isinstance(x, (list, tuple)):
        out: Dict[TokenKey, float] = {}
        for item in x:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            try:
                out[str(item[0])] = float(item[1])
            except Exception:
                continue
        return out
    return {}


def _extract_sparse_list(embs: dict) -> Optional[list]:
    for key in ("sparse_vecs", "lexical_weights"):
        if key in embs and embs.get(key) is not None:
            return embs.get(key)
    return None


def _build_hyde_prompts(question: str, num_variants: int) -> Tuple[str, str]:
    return (
        _HYDE_SYSTEM.format(n=num_variants),
        _HYDE_USER.format(question=question, n=num_variants),
    )


def _parse_hyde_output(text: str, num_variants: int) -> List[str]:
    parts = [part.strip() for part in text.split("---")]
    variants = [part for part in parts if len(part) >= 40]
    return variants[:num_variants]


def _cross_query_rrf(
    ranked_lists: List[List[Tuple[int, float]]],
    weights: List[float],
    rrf_k: int = 60,
) -> List[Tuple[int, float]]:
    fused: Dict[int, float] = {}
    for ranked, weight in zip(ranked_lists, weights):
        for rank, (doc_id, _) in enumerate(ranked, start=1):
            key = int(doc_id)
            fused[key] = fused.get(key, 0.0) + float(weight) / (float(rrf_k) + float(rank))
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


def _cross_query_minmax(
    ranked_lists: List[List[Tuple[int, float]]],
    weights: List[float],
) -> List[Tuple[int, float]]:
    fused: Dict[int, float] = {}
    for ranked, weight in zip(ranked_lists, weights):
        if not ranked:
            continue
        scores_dict = {int(doc_id): float(score) for doc_id, score in ranked}
        vals = np.fromiter(scores_dict.values(), dtype=np.float32)
        vmin, vmax = float(vals.min()), float(vals.max())
        denom = vmax - vmin
        for doc_id, score in scores_dict.items():
            normed = (float(score) - vmin) / denom if denom > 1e-12 else 0.0
            fused[doc_id] = fused.get(doc_id, 0.0) + float(weight) * normed
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


def _cross_query_fuse(
    ranked_lists: List[List[Tuple[int, float]]],
    weights: List[float],
    cross_fusion: str = "rrf",
    cross_rrf_k: int = 60,
) -> List[Tuple[int, float]]:
    method = (cross_fusion or "rrf").lower()
    if method == "rrf":
        return _cross_query_rrf(ranked_lists, weights, rrf_k=cross_rrf_k)
    if method == "minmax":
        return _cross_query_minmax(ranked_lists, weights)
    raise ValueError(
        f"Unsupported cross_fusion: {cross_fusion}. Use 'rrf' or 'minmax'."
    )


class HyDEGenerator:
    def __init__(
        self,
        llm_config: dict,
        num_variants: int = 2,
        temperature: float = 0.5,
        max_retries: int = 3,
        cache_dir: str = "data/retrieval/hyde_cache",
    ):
        self.llm_config = llm_config
        self.num_variants = num_variants
        self.temperature = temperature
        self.max_retries = max_retries
        self._model_name: str = llm_config.get("model_name", "")
        self._client = None
        self._client_type: Optional[str] = None

        os.makedirs(cache_dir, exist_ok=True)
        self._cache_path = os.path.join(cache_dir, "hyde_cache.jsonl")
        self._mem_cache: Dict[str, List[str]] = {}
        self._load_disk_cache()

    def _load_disk_cache(self):
        if not os.path.exists(self._cache_path):
            return
        try:
            with open(self._cache_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    self._mem_cache[entry["key"]] = entry["variants"]
        except Exception as exc:
            print(f"[HyDEGenerator] Failed to load disk cache: {exc}")

    def _persist(self, key: str, variants: List[str]):
        self._mem_cache[key] = variants
        try:
            with open(self._cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "variants": variants}, ensure_ascii=False) + "\n")
        except Exception as exc:
            print(f"[HyDEGenerator] Failed to write disk cache: {exc}")

    def _cache_key(self, query: str) -> str:
        raw = f"{_HYDE_PROMPT_VERSION}|{self._model_name}|{self.num_variants}|{query}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def _lazy_init_client(self):
        if self._client is not None:
            return
        model_type = self.llm_config.get("model_type", "openai")
        if model_type == "zhipu":
            from zhipuai import ZhipuAI  # type: ignore

            self._client = ZhipuAI(api_key=self.llm_config.get("api_key", ""))
            self._client_type = "zhipu"
            return

        import httpx
        from openai import OpenAI  # type: ignore

        api_base = self.llm_config.get("api_base", "")
        api_key = self.llm_config.get("api_key", "")
        if api_base and self._model_name.startswith("gpt"):
            self._client = OpenAI(
                base_url=api_base,
                api_key=api_key,
                http_client=httpx.Client(base_url=api_base, follow_redirects=True),
            )
        else:
            self._client = OpenAI(base_url=api_base, api_key=api_key)
        self._client_type = "openai"

    def generate(self, query: str) -> List[str]:
        key = self._cache_key(query)
        if key in self._mem_cache:
            return self._mem_cache[key]

        self._lazy_init_client()
        system_prompt, user_prompt = _build_hyde_prompts(query, self.num_variants)

        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self._model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self.temperature,
                    stream=False,
                )
                raw = response.choices[0].message.content.strip()
                variants = _parse_hyde_output(raw, self.num_variants)
                if variants:
                    self._persist(key, variants)
                    return variants
            except Exception as exc:
                print(
                    f"[HyDEGenerator] Attempt {attempt + 1}/{self.max_retries} failed: {exc}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)

        print(
            f"[HyDEGenerator] All retries failed for query='{query[:60]}...'; "
            "falling back to the original query only."
        )
        return []


class BGEM3HybridRetriever:
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        use_fp16: bool = True,
        cache_dir: str = "data/retrieval/bge_m3",
        normalize_dense: bool = True,
    ):
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.cache_dir = cache_dir
        self.normalize_dense = normalize_dense

        self._model = None
        self._faiss = None
        self._index = None
        self._inv_index: Optional[Dict[int, List[Tuple[int, float]]]] = None

        os.makedirs(self.cache_dir, exist_ok=True)

    def _lazy_import(self):
        if self._faiss is None:
            try:
                import faiss  # type: ignore
            except Exception as exc:
                raise ImportError(
                    "Missing dependency faiss. Please install a CPU or GPU build first."
                ) from exc
            self._faiss = faiss

        if self._model is None:
            try:
                import torch
                from FlagEmbedding import BGEM3FlagModel  # type: ignore
            except Exception as exc:
                raise ImportError(
                    "Missing dependency FlagEmbedding for BGE-M3 hybrid retrieval.\n"
                    "Please install it with: pip install FlagEmbedding"
                ) from exc

            if torch.cuda.is_available():
                try:
                    torch.zeros(1).cuda()
                    device = "cuda"
                except Exception:
                    print(
                        "Warning: CUDA device detected but not compatible with current "
                        "PyTorch. Falling back to CPU."
                    )
                    device = "cpu"
            else:
                device = "cpu"

            self._model = BGEM3FlagModel(
                self.model_name,
                use_fp16=(self.use_fp16 and device == "cuda"),
                device=device,
            )

    @property
    def law_faiss_path(self) -> str:
        return os.path.join(self.cache_dir, "law_dense.faiss")

    @property
    def inv_index_path(self) -> str:
        return os.path.join(self.cache_dir, "law_inv_index.pkl")

    @property
    def meta_path(self) -> str:
        return os.path.join(self.cache_dir, "meta.json")

    def build_or_load(self, law_texts: List[str], batch_size: int = 8, max_length: int = 8192):
        self._lazy_import()

        if (
            os.path.exists(self.law_faiss_path)
            and os.path.exists(self.inv_index_path)
            and os.path.exists(self.meta_path)
        ):
            try:
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                meta = {}

            try:
                index = self._faiss.read_index(self.law_faiss_path)
                with open(self.inv_index_path, "rb") as f:
                    inv = pickle.load(f)
            except Exception:
                index = None
                inv = None

            num_docs_ok = meta.get("num_docs") == len(law_texts)
            normalize_ok = meta.get("normalize_dense", True) == self.normalize_dense
            ntotal_ok = getattr(index, "ntotal", None) == len(law_texts)
            inv_ok = isinstance(inv, dict) and (len(inv) > 0 or len(law_texts) < 50)
            if num_docs_ok and normalize_ok and ntotal_ok and inv_ok:
                self._index = index
                self._inv_index = inv
                return

        if not law_texts:
            print("Warning: law_texts is empty, skipping build_or_load.")
            return

        print(f"Building BGE-M3 index for {len(law_texts)} texts...")
        embs = self._model.encode(
            law_texts,
            batch_size=batch_size,
            max_length=max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        dense = np.asarray(embs.get("dense_vecs"), dtype=np.float32)
        if self.normalize_dense:
            dense = _l2_normalize(dense)

        dim = int(dense.shape[1])
        index = self._faiss.IndexFlatIP(dim)
        index.add(dense)
        self._faiss.write_index(index, self.law_faiss_path)
        self._index = index

        sparse_list = _extract_sparse_list(embs) or []
        inv_index: Dict[TokenKey, List[Tuple[int, float]]] = {}
        for doc_id, sparse_vector in enumerate(sparse_list):
            sparse_dict = _as_token_weight_dict(sparse_vector)
            for token, weight in sparse_dict.items():
                inv_index.setdefault(token, []).append((doc_id, float(weight)))

        if len(law_texts) > 50 and len(inv_index) == 0:
            raise RuntimeError(
                "BGE-M3 sparse inverted index was not built.\n"
                "Please verify that FlagEmbedding returns 'lexical_weights' or "
                "'sparse_vecs' when return_sparse=True."
            )

        with open(self.inv_index_path, "wb") as f:
            pickle.dump(inv_index, f, protocol=pickle.HIGHEST_PROTOCOL)
        self._inv_index = inv_index

        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_name": self.model_name,
                    "num_docs": len(law_texts),
                    "dim": dim,
                    "normalize_dense": self.normalize_dense,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    def _dense_search(self, query_dense: np.ndarray, top_k: int) -> Dict[int, float]:
        query_dense = np.asarray(query_dense, dtype=np.float32)
        if query_dense.ndim == 1:
            query_dense = query_dense[None, :]
        if self.normalize_dense:
            query_dense = _l2_normalize(query_dense)
        distances, indices = self._index.search(query_dense.astype("float32"), top_k)
        scores = {}
        for idx, score in zip(indices[0].tolist(), distances[0].tolist()):
            scores[int(idx)] = float(score)
        return scores

    def _sparse_search(self, query_sparse: Dict[TokenKey, float], top_k: int) -> Dict[int, float]:
        inv = self._inv_index or {}
        acc: Dict[int, float] = {}
        for token, query_weight in query_sparse.items():
            postings = inv.get(str(token))
            if not postings:
                continue
            for doc_id, doc_weight in postings:
                acc[doc_id] = acc.get(doc_id, 0.0) + float(query_weight) * float(doc_weight)
        if not acc:
            return {}
        top = nlargest(top_k, acc.items(), key=lambda x: x[1])
        return {int(doc_id): float(score) for doc_id, score in top}

    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        dense_top_k: int = 50,
        sparse_top_k: int = 50,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        max_length: int = 8192,
        fusion: str = "rrf",
        rrf_k: int = 60,
    ) -> List[Tuple[int, float]]:
        self._lazy_import()
        if self._index is None or self._inv_index is None:
            raise RuntimeError("Index is not initialized. Call build_or_load() first.")

        embs = self._model.encode(
            [query],
            batch_size=1,
            max_length=max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        query_dense = np.asarray(embs.get("dense_vecs")[0], dtype=np.float32)
        sparse_list = _extract_sparse_list(embs) or []
        query_sparse = _as_token_weight_dict(sparse_list[0] if sparse_list else {})

        dense_scores = self._dense_search(query_dense, dense_top_k)
        sparse_scores = self._sparse_search(query_sparse, sparse_top_k)

        fusion_method = (fusion or "rrf").lower()
        if fusion_method == "minmax":
            dense_norm = _minmax_norm(dense_scores)
            sparse_norm = _minmax_norm(sparse_scores)
            doc_ids = set(dense_norm.keys()) | set(sparse_norm.keys())
            fused = {
                doc_id: dense_weight * dense_norm.get(doc_id, 0.0)
                + sparse_weight * sparse_norm.get(doc_id, 0.0)
                for doc_id in doc_ids
            }
        elif fusion_method == "rrf":
            fused = _rrf_fuse(
                dense_scores=dense_scores,
                sparse_scores=sparse_scores,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
                rrf_k=rrf_k,
            )
        else:
            raise ValueError(f"Unsupported fusion: {fusion}. Use 'rrf' or 'minmax'.")

        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        return [(int(doc_id), float(score)) for doc_id, score in ranked[:top_k]]

    def hybrid_search_batch(
        self,
        queries: List[str],
        top_k: int = 10,
        dense_top_k: int = 50,
        sparse_top_k: int = 50,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        max_length: int = 8192,
        batch_size: int = 32,
        fusion: str = "rrf",
        rrf_k: int = 60,
    ) -> List[List[Tuple[int, float]]]:
        from tqdm import tqdm

        self._lazy_import()
        if self._index is None or self._inv_index is None:
            raise RuntimeError("Index is not initialized. Call build_or_load() first.")

        embs = self._model.encode(
            queries,
            batch_size=batch_size,
            max_length=max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense_vecs = np.asarray(embs.get("dense_vecs"), dtype=np.float32)
        if self.normalize_dense:
            dense_vecs = _l2_normalize(dense_vecs)
        sparse_list = _extract_sparse_list(embs) or []

        distances, indices = self._index.search(dense_vecs.astype("float32"), dense_top_k)

        fusion_method = (fusion or "rrf").lower()
        results: List[List[Tuple[int, float]]] = []
        for i in tqdm(range(len(queries)), desc="Sparse search + fusion", leave=False):
            dense_scores = {
                int(indices[i][j]): float(distances[i][j])
                for j in range(dense_top_k)
                if int(indices[i][j]) >= 0
            }
            query_sparse = _as_token_weight_dict(sparse_list[i] if i < len(sparse_list) else {})
            sparse_scores = self._sparse_search(query_sparse, sparse_top_k)

            if fusion_method == "minmax":
                dense_norm = _minmax_norm(dense_scores)
                sparse_norm = _minmax_norm(sparse_scores)
                doc_ids = set(dense_norm.keys()) | set(sparse_norm.keys())
                fused = {
                    doc_id: dense_weight * dense_norm.get(doc_id, 0.0)
                    + sparse_weight * sparse_norm.get(doc_id, 0.0)
                    for doc_id in doc_ids
                }
            elif fusion_method == "rrf":
                fused = _rrf_fuse(
                    dense_scores=dense_scores,
                    sparse_scores=sparse_scores,
                    dense_weight=dense_weight,
                    sparse_weight=sparse_weight,
                    rrf_k=rrf_k,
                )
            else:
                raise ValueError(f"Unsupported fusion: {fusion}. Use 'rrf' or 'minmax'.")

            ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
            results.append([(int(doc_id), float(score)) for doc_id, score in ranked[:top_k]])

        return results


def build_progressive_results_path(enable_hyde: bool, enable_reranker: bool) -> str:
    if enable_hyde and enable_reranker:
        filename = "retrieval_bge-m3_hybrid_hyde_rerank.jsonl"
    elif enable_hyde:
        filename = "retrieval_bge-m3_hybrid_hyde.jsonl"
    elif enable_reranker:
        filename = "retrieval_bge-m3_hybrid_rerank.jsonl"
    else:
        filename = "retrieval_bge-m3_hybrid.jsonl"
    return f"data/retrieval/res/{filename}"


def build_progressive_run_config(
    enable_hyde: bool,
    enable_reranker: bool,
    overrides: Optional[Dict] = None,
) -> Dict:
    """
    Build a progressive retrieval config aligned with the original four entry
    scripts. The caller can still override any field.
    """
    from src.config.config import Config

    config: Dict = {
        "model_type": "BGE-M3-Progressive",
        "model_name": "BAAI/bge-m3",
        "question_file_path": "data/rewrite_question.jsonl",
        "law_path": "data/law_library.jsonl",
        "top_k": 10,
        "dense_top_k": 200,
        "sparse_top_k": 200,
        "dense_weight": 0.8,
        "sparse_weight": 0.2,
        "fusion": "minmax",
        "rrf_k": 60,
        "max_length": 1024,
        "enable_hyde": enable_hyde,
        "enable_reranker": enable_reranker,
    }

    if enable_hyde and enable_reranker:
        config.update(
            {
                "batch_size": 16,
                "normalize_dense": True,
                "cache_dir": "data/retrieval/bge_m3_hyde_rerank",
                "hyde_cache_dir": "data/retrieval/hyde_cache",
                "llm_config": Config._default_configs["qwen"],
                "hyde_num_variants": 2,
                "hyde_temperature": 0.3,
                "hyde_max_retries": 3,
                "per_query_top_k": 100,
                "query_weight": 1.0,
                "hyde_weight": 0.75,
                "cross_fusion": "minmax",
                "cross_rrf_k": 60,
                "reranker_model_name": "BAAI/bge-reranker-v2-m3",
                "rerank_top_k": 200,
                "rerank_batch_size": 32,
                "rerank_query_source": "first_variant",
            }
        )
    elif enable_hyde:
        config.update(
            {
                "batch_size": 16,
                "normalize_dense": False,
                "cache_dir": "data/retrieval/bge_m3",
                "hyde_cache_dir": "data/retrieval/hyde_cache",
                "llm_config": Config._default_configs["qwen"],
                "hyde_num_variants": 2,
                "hyde_temperature": 0.3,
                "hyde_max_retries": 3,
                "per_query_top_k": 100,
                "query_weight": 1.0,
                "hyde_weight": 0.75,
                "cross_fusion": "minmax",
                "cross_rrf_k": 60,
            }
        )
    elif enable_reranker:
        config.update(
            {
                "batch_size": 6,
                "normalize_dense": True,
                "cache_dir": "data/retrieval/bge_m3_rerank",
                "reranker_model_name": "BAAI/bge-reranker-v2-m3",
                "rerank_top_k": 200,
                "rerank_batch_size": 128,
            }
        )
    else:
        config.update(
            {
                "batch_size": 16,
                "normalize_dense": False,
                "cache_dir": "data/retrieval/bge_m3",
            }
        )

    config["results_path"] = build_progressive_results_path(
        enable_hyde=enable_hyde,
        enable_reranker=enable_reranker,
    )

    if overrides:
        config.update(overrides)

    if "results_path" not in config or not config["results_path"]:
        config["results_path"] = build_progressive_results_path(
            enable_hyde=config.get("enable_hyde", False),
            enable_reranker=config.get("enable_reranker", False),
        )

    return config


def run_progressive_experiment(
    enable_hyde: bool,
    enable_reranker: bool,
    overrides: Optional[Dict] = None,
    metrics: Optional[Iterable[str]] = None,
    k_values: Optional[Iterable[int]] = None,
):
    from src.pipeline import EvaluatorPipeline, RetrieverPipeline

    run_config = build_progressive_run_config(
        enable_hyde=enable_hyde,
        enable_reranker=enable_reranker,
        overrides=overrides,
    )

    pipeline = RetrieverPipeline()
    pipeline.run_retriever(**run_config)

    evaluator = EvaluatorPipeline()
    evaluator.run_evaluator(
        eval_type="retrieval",
        results_path=run_config["results_path"],
        metrics=list(metrics or ["recall", "precision", "f1", "ndcg", "mrr"]),
        k_values=list(k_values or [1, 3, 5, 10]),
    )

    return run_config


@dataclass(frozen=True)
class ProgressiveComponents:
    enable_hyde: bool = False
    enable_reranker: bool = False


@dataclass
class QueryBundle:
    retrieval_queries: List[str]
    retrieval_weights: List[float]
    rerank_query: str


class HybridRetrievalComponent:
    """
    Mandatory dense + sparse hybrid retrieval component.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        use_fp16: bool = True,
        cache_dir: str = "data/retrieval/bge_m3",
        normalize_dense: bool = True,
    ):
        self._retriever = BGEM3HybridRetriever(
            model_name=model_name,
            use_fp16=use_fp16,
            cache_dir=cache_dir,
            normalize_dense=normalize_dense,
        )

    def build_or_load(
        self,
        law_texts: List[str],
        batch_size: int = 8,
        max_length: int = 8192,
    ):
        self._retriever.build_or_load(
            law_texts,
            batch_size=batch_size,
            max_length=max_length,
        )

    def search_batch(
        self,
        queries: List[str],
        top_k: int = 10,
        dense_top_k: int = 50,
        sparse_top_k: int = 50,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        max_length: int = 8192,
        batch_size: int = 32,
        fusion: str = "rrf",
        rrf_k: int = 60,
    ) -> List[List[Tuple[int, float]]]:
        return self._retriever.hybrid_search_batch(
            queries=queries,
            top_k=top_k,
            dense_top_k=dense_top_k,
            sparse_top_k=sparse_top_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            max_length=max_length,
            batch_size=batch_size,
            fusion=fusion,
            rrf_k=rrf_k,
        )


class HyDEAugmentationComponent:
    """
    Optional HyDE query augmentation component.
    """

    def __init__(
        self,
        llm_config: dict,
        hyde_num_variants: int = 2,
        hyde_temperature: float = 0.5,
        hyde_max_retries: int = 3,
        hyde_cache_dir: str = "data/retrieval/hyde_cache",
    ):
        self._generator = HyDEGenerator(
            llm_config=llm_config,
            num_variants=hyde_num_variants,
            temperature=hyde_temperature,
            max_retries=hyde_max_retries,
            cache_dir=hyde_cache_dir,
        )

    def generate(self, query: str) -> List[str]:
        return self._generator.generate(query)


class RerankerComponent:
    """
    Optional cross-encoder reranking component.
    """

    def __init__(
        self,
        reranker_model_name: str = "BAAI/bge-reranker-v2-m3",
        use_fp16: bool = True,
    ):
        self.reranker_model_name = reranker_model_name
        self.use_fp16 = use_fp16
        self._reranker = None

    def _lazy_load(self):
        if self._reranker is not None:
            return

        try:
            import torch
            from FlagEmbedding import FlagReranker  # type: ignore
        except Exception as e:
            raise ImportError(
                "Missing dependency FlagEmbedding for BGE reranker.\n"
                "Please install it with: pip install FlagEmbedding"
            ) from e

        if torch.cuda.is_available():
            try:
                torch.zeros(1).cuda()
                device = "cuda"
            except Exception:
                device = "cpu"
        else:
            device = "cpu"

        self._reranker = FlagReranker(
            self.reranker_model_name,
            use_fp16=(self.use_fp16 and device == "cuda"),
            device=device,
        )

    def score_pairs(
        self,
        pairs: Sequence[Sequence[str]],
        batch_size: int = 32,
    ) -> List[float]:
        if not pairs:
            return []

        self._lazy_load()
        scores = self._reranker.compute_score(list(pairs), batch_size=batch_size)
        if not isinstance(scores, list):
            scores = [scores]
        return [float(score) for score in scores]


class BGEM3ProgressiveRetriever:
    """
    Progressive retrieval framework:
    user query
      -> optional HyDE augmentation
      -> mandatory hybrid retrieval
      -> optional reranker

    The four legacy strategies can all be represented by one orchestrator:
    - Hybrid only
    - HyDE + Hybrid
    - Hybrid + Reranker
    - HyDE + Hybrid + Reranker
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        reranker_model_name: str = "BAAI/bge-reranker-v2-m3",
        use_fp16: bool = True,
        cache_dir: str = "data/retrieval/bge_m3",
        normalize_dense: bool = True,
        components: Optional[ProgressiveComponents] = None,
        llm_config: Optional[dict] = None,
        hyde_num_variants: int = 2,
        hyde_temperature: float = 0.5,
        hyde_max_retries: int = 3,
        hyde_cache_dir: str = "data/retrieval/hyde_cache",
        query_weight: float = 1.0,
        hyde_weight: float = 0.5,
        cross_fusion: str = "rrf",
        cross_rrf_k: int = 60,
        rerank_query_source: str = "query",
    ):
        self.components = components or ProgressiveComponents()

        if self.components.enable_hyde and llm_config is None:
            raise ValueError(
                "HyDE component requires llm_config. "
                "Please pass a config dict with fields such as "
                "model_type / model_name / api_key."
            )

        if rerank_query_source not in ("query", "first_variant"):
            raise ValueError(
                "rerank_query_source only supports 'query' or 'first_variant', "
                f"got: {rerank_query_source!r}"
            )

        self.query_weight = query_weight
        self.hyde_weight = hyde_weight
        self.cross_fusion = cross_fusion
        self.cross_rrf_k = cross_rrf_k
        self.rerank_query_source = rerank_query_source

        self.hybrid = HybridRetrievalComponent(
            model_name=model_name,
            use_fp16=use_fp16,
            cache_dir=cache_dir,
            normalize_dense=normalize_dense,
        )
        self.hyde = (
            HyDEAugmentationComponent(
                llm_config=llm_config,
                hyde_num_variants=hyde_num_variants,
                hyde_temperature=hyde_temperature,
                hyde_max_retries=hyde_max_retries,
                hyde_cache_dir=hyde_cache_dir,
            )
            if self.components.enable_hyde
            else None
        )
        self.reranker = (
            RerankerComponent(
                reranker_model_name=reranker_model_name,
                use_fp16=use_fp16,
            )
            if self.components.enable_reranker
            else None
        )

        self._law_texts: Optional[List[str]] = None

    def build_or_load(
        self,
        law_texts: List[str],
        batch_size: int = 8,
        max_length: int = 8192,
    ):
        self._law_texts = list(law_texts)
        self.hybrid.build_or_load(
            law_texts,
            batch_size=batch_size,
            max_length=max_length,
        )

    def _build_query_bundle(self, query: str) -> QueryBundle:
        if not self.components.enable_hyde:
            return QueryBundle(
                retrieval_queries=[query],
                retrieval_weights=[1.0],
                rerank_query=query,
            )

        hyde_variants = self.hyde.generate(query)
        rerank_query = query
        if self.rerank_query_source == "first_variant" and hyde_variants:
            rerank_query = hyde_variants[0]

        return QueryBundle(
            retrieval_queries=[query] + hyde_variants,
            retrieval_weights=[self.query_weight] + [self.hyde_weight] * len(hyde_variants),
            rerank_query=rerank_query,
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        per_query_top_k: int = 100,
        dense_top_k: int = 50,
        sparse_top_k: int = 50,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        max_length: int = 8192,
        batch_size: int = 32,
        fusion: str = "rrf",
        rrf_k: int = 60,
        rerank_top_k: int = 100,
        rerank_batch_size: int = 32,
    ) -> List[Tuple[int, float]]:
        results = self.search_batch(
            queries=[query],
            top_k=top_k,
            per_query_top_k=per_query_top_k,
            dense_top_k=dense_top_k,
            sparse_top_k=sparse_top_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            max_length=max_length,
            batch_size=batch_size,
            fusion=fusion,
            rrf_k=rrf_k,
            rerank_top_k=rerank_top_k,
            rerank_batch_size=rerank_batch_size,
        )
        return results[0] if results else []

    def search_batch(
        self,
        queries: List[str],
        top_k: int = 10,
        per_query_top_k: int = 100,
        dense_top_k: int = 50,
        sparse_top_k: int = 50,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        max_length: int = 8192,
        batch_size: int = 32,
        fusion: str = "rrf",
        rrf_k: int = 60,
        rerank_top_k: int = 100,
        rerank_batch_size: int = 32,
    ) -> List[List[Tuple[int, float]]]:
        if not queries:
            return []

        if self.components.enable_reranker and self._law_texts is None:
            raise RuntimeError(
                "Retriever is not initialized. Please call build_or_load() first."
            )

        query_bundles = [self._build_query_bundle(query) for query in queries]
        candidate_top_k = top_k
        if self.components.enable_hyde:
            candidate_top_k = per_query_top_k
        elif self.components.enable_reranker:
            candidate_top_k = rerank_top_k

        flat_queries: List[str] = []
        flat_meta: List[Tuple[int, int]] = []
        for query_idx, bundle in enumerate(query_bundles):
            for bundle_query_idx, retrieval_query in enumerate(bundle.retrieval_queries):
                flat_queries.append(retrieval_query)
                flat_meta.append((query_idx, bundle_query_idx))

        flat_ranked_lists = self.hybrid.search_batch(
            queries=flat_queries,
            top_k=candidate_top_k,
            dense_top_k=dense_top_k,
            sparse_top_k=sparse_top_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            max_length=max_length,
            batch_size=batch_size,
            fusion=fusion,
            rrf_k=rrf_k,
        )

        grouped_ranked_lists: List[List[List[Tuple[int, float]]]] = [[] for _ in queries]
        grouped_weights: List[List[float]] = [[] for _ in queries]

        for (query_idx, bundle_query_idx), ranked in zip(flat_meta, flat_ranked_lists):
            if not ranked:
                continue
            grouped_ranked_lists[query_idx].append(ranked)
            grouped_weights[query_idx].append(
                query_bundles[query_idx].retrieval_weights[bundle_query_idx]
            )

        if not self.components.enable_reranker:
            results: List[List[Tuple[int, float]]] = []
            for query_idx in range(len(queries)):
                ranked_lists = grouped_ranked_lists[query_idx]
                if not ranked_lists:
                    results.append([])
                    continue

                if self.components.enable_hyde:
                    fused = _cross_query_fuse(
                        ranked_lists,
                        grouped_weights[query_idx],
                        cross_fusion=self.cross_fusion,
                        cross_rrf_k=self.cross_rrf_k,
                    )
                else:
                    fused = ranked_lists[0]

                results.append(
                    [(int(doc_id), float(score)) for doc_id, score in fused[:top_k]]
                )
            return results

        all_pairs: List[List[str]] = []
        candidate_doc_ids_per_query: List[List[int]] = []

        for query_idx, bundle in enumerate(query_bundles):
            ranked_lists = grouped_ranked_lists[query_idx]
            if not ranked_lists:
                candidate_doc_ids_per_query.append([])
                continue

            if self.components.enable_hyde:
                fused = _cross_query_fuse(
                    ranked_lists,
                    grouped_weights[query_idx],
                    cross_fusion=self.cross_fusion,
                    cross_rrf_k=self.cross_rrf_k,
                )
            else:
                fused = ranked_lists[0]

            candidate_doc_ids = [int(doc_id) for doc_id, _ in fused[:rerank_top_k]]
            candidate_doc_ids_per_query.append(candidate_doc_ids)

            for doc_id in candidate_doc_ids:
                all_pairs.append([bundle.rerank_query, self._law_texts[doc_id]])

        if not all_pairs:
            return [[] for _ in queries]

        rerank_scores = self.reranker.score_pairs(
            all_pairs,
            batch_size=rerank_batch_size,
        )

        results: List[List[Tuple[int, float]]] = []
        pair_idx = 0
        for candidate_doc_ids in candidate_doc_ids_per_query:
            if not candidate_doc_ids:
                results.append([])
                continue

            scored: List[Tuple[int, float]] = []
            for doc_id in candidate_doc_ids:
                if pair_idx >= len(rerank_scores):
                    break
                scored.append((int(doc_id), float(rerank_scores[pair_idx])))
                pair_idx += 1

            scored.sort(key=lambda x: x[1], reverse=True)
            results.append(scored[:top_k])

        return results


ProgressiveRetriever = BGEM3ProgressiveRetriever
fuse_ranked_lists = _cross_query_fuse


__all__ = [
    "BGEM3HybridRetriever",
    "BGEM3ProgressiveRetriever",
    "HybridRetrievalComponent",
    "HyDEAugmentationComponent",
    "HyDEGenerator",
    "ProgressiveRetriever",
    "ProgressiveComponents",
    "QueryBundle",
    "RerankerComponent",
    "build_progressive_results_path",
    "build_progressive_run_config",
    "fuse_ranked_lists",
    "run_progressive_experiment",
]
