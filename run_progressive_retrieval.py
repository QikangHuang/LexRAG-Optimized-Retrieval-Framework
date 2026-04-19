from src.retrieval.bge_m3_progressive_retrieval import run_progressive_experiment


# ---------------------------------------------------------------------------
# Component switches
# Hybrid retrieval is always enabled.
# You only choose whether to add HyDE before it and reranker after it.
# ---------------------------------------------------------------------------
ENABLE_HYDE = True
ENABLE_RERANKER = False


# ---------------------------------------------------------------------------
# Input / output
# ---------------------------------------------------------------------------
QUESTION_FILE_PATH = "data/rewrite_question.jsonl"
LAW_FILE_PATH = "data/law_library.jsonl"
MODEL_NAME = "BAAI/bge-m3"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"


# ---------------------------------------------------------------------------
# Retrieval parameters
# These are the most commonly tuned parameters.
# ---------------------------------------------------------------------------
TOP_K = 10
DENSE_TOP_K = 200
SPARSE_TOP_K = 200
DENSE_WEIGHT = 0.8
SPARSE_WEIGHT = 0.2
FUSION = "minmax"          # "rrf" or "minmax"
RRF_K = 60
BATCH_SIZE = None          # None -> use the legacy default for the selected mode
MAX_LENGTH = 1024
NORMALIZE_DENSE = None     # None -> use the legacy default for the selected mode
CACHE_DIR = None           # None -> use the legacy default for the selected mode


# ---------------------------------------------------------------------------
# HyDE parameters
# Only used when ENABLE_HYDE = True.
# ---------------------------------------------------------------------------
LLM_CONFIG = None          # None -> reuse the legacy HyDE default from build_progressive_run_config
HYDE_CACHE_DIR = "data/retrieval/hyde_cache"
HYDE_NUM_VARIANTS = 2
HYDE_TEMPERATURE = 0.3
HYDE_MAX_RETRIES = 3
PER_QUERY_TOP_K = 100
QUERY_WEIGHT = 1.0
HYDE_WEIGHT = 0.75
CROSS_FUSION = "minmax"    # "rrf" or "minmax"
CROSS_RRF_K = 60


# ---------------------------------------------------------------------------
# Reranker parameters
# Only used when ENABLE_RERANKER = True.
# ---------------------------------------------------------------------------
RERANK_TOP_K = 200
RERANK_BATCH_SIZE = None   # None -> use the legacy default for the selected mode
RERANK_QUERY_SOURCE = "first_variant"   # "query" or "first_variant"


# ---------------------------------------------------------------------------
# Evaluation parameters
# ---------------------------------------------------------------------------
METRICS = ["recall", "precision", "f1", "ndcg", "mrr"]
K_VALUES = [1, 3, 5, 10]


def build_overrides():
    overrides = {
        "question_file_path": QUESTION_FILE_PATH,
        "law_path": LAW_FILE_PATH,
        "model_name": MODEL_NAME,
        "top_k": TOP_K,
        "dense_top_k": DENSE_TOP_K,
        "sparse_top_k": SPARSE_TOP_K,
        "dense_weight": DENSE_WEIGHT,
        "sparse_weight": SPARSE_WEIGHT,
        "fusion": FUSION,
        "rrf_k": RRF_K,
        "max_length": MAX_LENGTH,
    }

    if BATCH_SIZE is not None:
        overrides["batch_size"] = BATCH_SIZE
    if NORMALIZE_DENSE is not None:
        overrides["normalize_dense"] = NORMALIZE_DENSE
    if CACHE_DIR is not None:
        overrides["cache_dir"] = CACHE_DIR

    if ENABLE_HYDE:
        overrides.update(
            {
                "hyde_cache_dir": HYDE_CACHE_DIR,
                "hyde_num_variants": HYDE_NUM_VARIANTS,
                "hyde_temperature": HYDE_TEMPERATURE,
                "hyde_max_retries": HYDE_MAX_RETRIES,
                "per_query_top_k": PER_QUERY_TOP_K,
                "query_weight": QUERY_WEIGHT,
                "hyde_weight": HYDE_WEIGHT,
                "cross_fusion": CROSS_FUSION,
                "cross_rrf_k": CROSS_RRF_K,
            }
        )
        if LLM_CONFIG is not None:
            overrides["llm_config"] = LLM_CONFIG

    if ENABLE_RERANKER:
        overrides.update(
            {
                "reranker_model_name": RERANKER_MODEL_NAME,
                "rerank_top_k": RERANK_TOP_K,
                "rerank_query_source": RERANK_QUERY_SOURCE,
            }
        )
        if RERANK_BATCH_SIZE is not None:
            overrides["rerank_batch_size"] = RERANK_BATCH_SIZE

    return overrides


run_progressive_experiment(
    enable_hyde=ENABLE_HYDE,
    enable_reranker=ENABLE_RERANKER,
    overrides=build_overrides(),
    metrics=METRICS,
    k_values=K_VALUES,
)
