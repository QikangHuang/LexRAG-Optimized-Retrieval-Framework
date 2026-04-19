import faiss
import json
from tqdm import tqdm
import numpy as np
from pathlib import Path
import os
from .dense_retriever import DenseRetriever
from .bge_m3_progressive_retrieval import (
    BGEM3ProgressiveRetriever,
    ProgressiveComponents,
    build_progressive_results_path,
)

class Pipeline:
    def __init__(self, config=None):
        self.openai_config = config or {}
        self.init_dir()

    def _create_lexical_retriever(self, bm25_backend=None):
        # Keep pyserini/Java as an optional dependency for BM25/QLD only.
        from .lexical_matching import LexicalRetriever

        return LexicalRetriever(bm25_backend=bm25_backend)

    def _create_bge_m3_colbert_retriever(self, **kwargs):
        try:
            from .bge_m3_colbert_retriever import BGEM3ColBERTRetriever
        except ImportError as exc:
            raise ImportError(
                "BGE-M3 ColBERT retrieval is unavailable because "
                "src/retrieval/bge_m3_colbert_retriever.py could not be imported."
            ) from exc

        return BGEM3ColBERTRetriever(**kwargs)

    def run_retriever(self, model_type, question_file_path, law_path, 
           bm25_backend="bm25s", faiss_type="FlatIP", model_name=None, **kwargs):
        if model_type == "bm25":
            self.pipeline_bm25(question_file_path, law_path, bm25_backend)
        elif model_type == "qld":
            self.pipeline_qld(question_file_path, law_path)
        elif model_type in {"BGE-M3", "bge-m3", "bge_m3", "bge-m3-hybrid"}:
            self.pipeline_bge_m3_hybrid(
                question_file_path=question_file_path,
                law_path=law_path,
                model_name=model_name,
                **kwargs,
            )
        elif model_type in {"BGE-M3-Rerank", "bge-m3-rerank", "bge_m3_rerank"}:
            self.pipeline_bge_m3_hybrid_rerank(
                question_file_path=question_file_path,
                law_path=law_path,
                model_name=model_name,
                **kwargs,
            )
        elif model_type in {"BGE-M3-HyDE", "bge-m3-hyde", "bge_m3_hyde"}:
            self.pipeline_bge_m3_hybrid_hyde(
                question_file_path=question_file_path,
                law_path=law_path,
                model_name=model_name,
                **kwargs,
            )
        elif model_type in {
            "BGE-M3-HyDE-Rerank", "bge-m3-hyde-rerank", "bge_m3_hyde_rerank",
        }:
            self.pipeline_bge_m3_hybrid_hyde_rerank(
                question_file_path=question_file_path,
                law_path=law_path,
                model_name=model_name,
                **kwargs,
            )
        elif model_type in {
            "BGE-M3-Progressive", "bge-m3-progressive", "bge_m3_progressive",
        }:
            self.pipeline_bge_m3_progressive(
                question_file_path=question_file_path,
                law_path=law_path,
                model_name=model_name,
                **kwargs,
            )
        elif model_type in {"BGE-M3-ColBERT", "bge-m3-colbert", "bge_m3_colbert"}:
            self.pipeline_bge_m3_colbert(
                question_file_path=question_file_path,
                law_path=law_path,
                model_name=model_name,
                **kwargs,
            )
        else:
            self.pipeline_law(law_path, model_type, faiss_type, model_name)
            self.pipeline_question(question_file_path, model_type, model_name)
            self.pipeline_search(question_file_path, law_path, model_type, faiss_type)

    def pipeline_bm25(self, question_path, law_path, backend):
        res_path = f"data/retrieval/res/retrieval_bm25_{backend}.jsonl"
        
        with open(question_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
        with open(law_path, "r", encoding="utf-8") as f:
            laws = [json.loads(line) for line in f]
        corpus = [law["name"] + law["content"] for law in laws]
        
        retriever = self._create_lexical_retriever(bm25_backend=backend)
        queries = [conv["question"]["content"] for d in data for conv in d["conversation"]]
        
        if backend == "bm25s":
            result_idx_list, scores = retriever.search(corpus, law_path, queries, k=10)
            idx = 0
            for d in data:
                for conv in d["conversation"]:
                    tmp_laws = []
                    for result_idx, score in zip(result_idx_list[idx][0], scores[idx][0]):
                        tmp_laws.append({
                            "article": laws[result_idx],
                            "score": float(score)
                        })
                    conv["question"]["recall"] = tmp_laws
                    idx += 1
                    
        elif backend == "pyserini":
            results, scores = retriever.search(corpus, law_path, queries, k=10)
            idx = 0
            for d in data:
                for conv in d["conversation"]:
                    tmp_laws = []
                    for doc_id, score in zip(results[idx], scores[idx]):
                        tmp_laws.append({
                            "article": laws[int(doc_id)],
                            "score": float(score)
                        })
                    conv["question"]["recall"] = tmp_laws
                    idx += 1

        with open(res_path, "w", encoding="utf-8") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    def pipeline_qld(self, question_path, law_path):
        res_path = "data/retrieval/res/retrieval_qld.jsonl"

        with open(question_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
        with open(law_path, "r", encoding="utf-8") as f:
            laws = [json.loads(line) for line in f]
        corpus = [law["name"] + law["content"] for law in laws]
        
        retriever = self._create_lexical_retriever()
        queries = [conv["question"]["content"] for d in data for conv in d["conversation"]]

        results, scores = retriever.search(corpus, law_path, queries, k=10, method="qld")
        idx = 0
        for d in data:
            for conv in d["conversation"]:
                tmp_laws = []
                for doc_id, score in zip(results[idx], scores[idx]):
                    tmp_laws.append({
                        "article": laws[int(doc_id)],
                        "score": float(score)
                    })
                conv["question"]["recall"] = tmp_laws
                idx += 1

        with open(res_path, "w", encoding="utf-8") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    def pipeline_bge_m3_hybrid(
        self,
        question_file_path: str,
        law_path: str,
        model_name: str = None,
        top_k: int = 10,
        dense_top_k: int = 50,
        sparse_top_k: int = 50,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        batch_size: int = 8,
        max_length: int = 8192,
        cache_dir: str = "data/retrieval/bge_m3",
        use_fp16: bool = True,
        normalize_dense: bool = True,
        fusion: str = "rrf",
        rrf_k: int = 60,
    ):
        self.pipeline_bge_m3_progressive(
            question_file_path=question_file_path,
            law_path=law_path,
            model_name=model_name,
            top_k=top_k,
            dense_top_k=dense_top_k,
            sparse_top_k=sparse_top_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            batch_size=batch_size,
            max_length=max_length,
            cache_dir=cache_dir,
            use_fp16=use_fp16,
            normalize_dense=normalize_dense,
            fusion=fusion,
            rrf_k=rrf_k,
            enable_hyde=False,
            enable_reranker=False,
            results_path="data/retrieval/res/retrieval_bge-m3_hybrid.jsonl",
            log_label="[BGE-M3 Hybrid]",
        )

    def pipeline_bge_m3_hybrid_rerank(
        self,
        question_file_path: str,
        law_path: str,
        model_name: str = None,
        reranker_model_name: str = None,
        top_k: int = 10,
        dense_top_k: int = 50,
        sparse_top_k: int = 50,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        batch_size: int = 8,
        max_length: int = 8192,
        rerank_top_k: int = 100,
        rerank_batch_size: int = 32,
        cache_dir: str = "data/retrieval/bge_m3_rerank",
        use_fp16: bool = True,
        normalize_dense: bool = True,
        fusion: str = "rrf",
        rrf_k: int = 60,
    ):
        self.pipeline_bge_m3_progressive(
            question_file_path=question_file_path,
            law_path=law_path,
            model_name=model_name,
            reranker_model_name=reranker_model_name or "BAAI/bge-reranker-v2-m3",
            top_k=top_k,
            dense_top_k=dense_top_k,
            sparse_top_k=sparse_top_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            batch_size=batch_size,
            max_length=max_length,
            cache_dir=cache_dir,
            use_fp16=use_fp16,
            normalize_dense=normalize_dense,
            fusion=fusion,
            rrf_k=rrf_k,
            rerank_top_k=rerank_top_k,
            rerank_batch_size=rerank_batch_size,
            enable_hyde=False,
            enable_reranker=True,
            results_path="data/retrieval/res/retrieval_bge-m3_hybrid_rerank.jsonl",
            log_label="[BGE-M3 Hybrid + Reranker]",
        )

    def pipeline_law(self, law_path, model_type, faiss_type, model_name):
        law_index_path = f"data/retrieval/law_index_{model_type}.faiss"
        checkpoint_path = f"data/retrieval/npy/law_embeddings_{model_type}.npy"
        
        # 如果最终索引文件已存在，直接返回
        if os.path.exists(law_index_path):
            return

        with open(law_path, "r", encoding="utf-8") as f:
            laws = [json.loads(line)["name"]+json.loads(line)["content"] for line in f]

        emb_model = DenseRetriever(**self.openai_config)
        # 使用断点续传功能
        embeddings = emb_model.embed(laws, model_type, model_name, checkpoint_path=checkpoint_path)
        
        # 保存最终索引
        emb_model.save_faiss(embeddings, faiss_type, law_index_path)
        
        # 清理临时断点文件
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

    def pipeline_question(self, question_path, model_type, model_name):
        question_emb_path = f"data/retrieval/npy/retrieval_{model_type}.npy"
        checkpoint_path = f"data/retrieval/npy/question_embeddings_{model_type}.npy"
        
        # 如果最终文件已存在，直接返回
        if os.path.exists(question_emb_path):
            return

        with open(question_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
            questions = [q["question"]["content"] for d in data for q in d["conversation"]]

        emb_model = DenseRetriever(**self.openai_config)
        # 使用断点续传功能
        embeddings = emb_model.embed(questions, model_type, model_name, checkpoint_path=checkpoint_path)
        
        # 保存最终结果
        np.save(question_emb_path, embeddings)
        
        # 清理临时断点文件
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

    def pipeline_search(self, question_path, law_path, model_type, faiss_type):
        res_path = f"data/retrieval/res/retrieval_{model_type}.jsonl"
        law_index_path = f"data/retrieval/law_index_{model_type}.faiss"
        question_emb_path = f"data/retrieval/npy/retrieval_{model_type}.npy"

        index = faiss.read_index(law_index_path)
        question_embeds = np.load(question_emb_path)
        D, I = index.search(question_embeds.astype('float32'), 10)
        
        with open(law_path, "r", encoding="utf-8") as f:
            laws = [json.loads(line) for line in f]
        
        with open(question_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
        
        self.incorporate_dense_results(data, laws, D, I, res_path)

    def incorporate_dense_results(self, data, laws, D, I, res_path):
        idx = 0
        for d in data:
            for conv in d["conversation"]:
                tmp_laws = []
                for i in range(len(I[idx])):
                    tmp_laws.append({
                        "article": laws[I[idx][i]], 
                        "score": float(D[idx][i])
                    })
                conv["question"]["recall"] = tmp_laws
                idx += 1
        
        with open(res_path, "w", encoding="utf-8") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    def pipeline_bge_m3_hybrid_hyde(
        self,
        question_file_path: str,
        law_path: str,
        model_name: str = None,
        top_k: int = 10,
        per_query_top_k: int = 100,
        dense_top_k: int = 50,
        sparse_top_k: int = 50,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        batch_size: int = 8,
        max_length: int = 8192,
        cache_dir: str = "data/retrieval/bge_m3",
        hyde_cache_dir: str = "data/retrieval/hyde_cache",
        use_fp16: bool = True,
        normalize_dense: bool = True,
        fusion: str = "rrf",
        rrf_k: int = 60,
        llm_config: dict = None,
        hyde_num_variants: int = 2,
        hyde_temperature: float = 0.5,
        hyde_max_retries: int = 3,
        query_weight: float = 1.0,
        hyde_weight: float = 0.5,
        cross_fusion: str = "rrf",
        cross_rrf_k: int = 60,
    ):
        self.pipeline_bge_m3_progressive(
            question_file_path=question_file_path,
            law_path=law_path,
            model_name=model_name,
            top_k=top_k,
            per_query_top_k=per_query_top_k,
            dense_top_k=dense_top_k,
            sparse_top_k=sparse_top_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            batch_size=batch_size,
            max_length=max_length,
            cache_dir=cache_dir,
            hyde_cache_dir=hyde_cache_dir,
            use_fp16=use_fp16,
            normalize_dense=normalize_dense,
            fusion=fusion,
            rrf_k=rrf_k,
            llm_config=llm_config,
            hyde_num_variants=hyde_num_variants,
            hyde_temperature=hyde_temperature,
            hyde_max_retries=hyde_max_retries,
            query_weight=query_weight,
            hyde_weight=hyde_weight,
            cross_fusion=cross_fusion,
            cross_rrf_k=cross_rrf_k,
            enable_hyde=True,
            enable_reranker=False,
            results_path="data/retrieval/res/retrieval_bge-m3_hybrid_hyde.jsonl",
            log_label="[BGE-M3 Hybrid + HyDE]",
        )

    def pipeline_bge_m3_hybrid_hyde_rerank(
        self,
        question_file_path: str,
        law_path: str,
        model_name: str = None,
        reranker_model_name: str = "BAAI/bge-reranker-v2-m3",
        top_k: int = 10,
        per_query_top_k: int = 100,
        dense_top_k: int = 50,
        sparse_top_k: int = 50,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        batch_size: int = 8,
        max_length: int = 8192,
        cache_dir: str = "data/retrieval/bge_m3_hyde_rerank",
        hyde_cache_dir: str = "data/retrieval/hyde_cache",
        use_fp16: bool = True,
        normalize_dense: bool = True,
        fusion: str = "rrf",
        rrf_k: int = 60,
        rerank_top_k: int = 50,
        rerank_batch_size: int = 32,
        llm_config: dict = None,
        hyde_num_variants: int = 2,
        hyde_temperature: float = 0.5,
        hyde_max_retries: int = 3,
        query_weight: float = 1.0,
        hyde_weight: float = 0.5,
        cross_fusion: str = "rrf",
        cross_rrf_k: int = 60,
        rerank_query_source: str = "query",
    ):
        self.pipeline_bge_m3_progressive(
            question_file_path=question_file_path,
            law_path=law_path,
            model_name=model_name,
            reranker_model_name=reranker_model_name,
            top_k=top_k,
            per_query_top_k=per_query_top_k,
            dense_top_k=dense_top_k,
            sparse_top_k=sparse_top_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            batch_size=batch_size,
            max_length=max_length,
            cache_dir=cache_dir,
            hyde_cache_dir=hyde_cache_dir,
            use_fp16=use_fp16,
            normalize_dense=normalize_dense,
            fusion=fusion,
            rrf_k=rrf_k,
            rerank_top_k=rerank_top_k,
            rerank_batch_size=rerank_batch_size,
            llm_config=llm_config,
            hyde_num_variants=hyde_num_variants,
            hyde_temperature=hyde_temperature,
            hyde_max_retries=hyde_max_retries,
            query_weight=query_weight,
            hyde_weight=hyde_weight,
            cross_fusion=cross_fusion,
            cross_rrf_k=cross_rrf_k,
            rerank_query_source=rerank_query_source,
            enable_hyde=True,
            enable_reranker=True,
            results_path="data/retrieval/res/retrieval_bge-m3_hybrid_hyde_rerank.jsonl",
            log_label="[BGE-M3 Hybrid + HyDE + Rerank]",
        )

    def pipeline_bge_m3_progressive(
        self,
        question_file_path: str,
        law_path: str,
        model_name: str = None,
        reranker_model_name: str = "BAAI/bge-reranker-v2-m3",
        top_k: int = 10,
        per_query_top_k: int = 100,
        dense_top_k: int = 50,
        sparse_top_k: int = 50,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        batch_size: int = 8,
        max_length: int = 8192,
        cache_dir: str = "data/retrieval/bge_m3",
        hyde_cache_dir: str = "data/retrieval/hyde_cache",
        use_fp16: bool = True,
        normalize_dense: bool = True,
        fusion: str = "rrf",
        rrf_k: int = 60,
        rerank_top_k: int = 100,
        rerank_batch_size: int = 32,
        llm_config: dict = None,
        hyde_num_variants: int = 2,
        hyde_temperature: float = 0.5,
        hyde_max_retries: int = 3,
        query_weight: float = 1.0,
        hyde_weight: float = 0.5,
        cross_fusion: str = "rrf",
        cross_rrf_k: int = 60,
        rerank_query_source: str = "query",
        enable_hyde: bool = False,
        enable_reranker: bool = False,
        results_path: str = None,
        log_label: str = None,
    ):
        """
        Unified BGE-M3 retrieval entry:
        - hybrid retrieval is always enabled
        - HyDE and reranker are optional progressive components
        """
        res_path = results_path or build_progressive_results_path(
            enable_hyde=enable_hyde,
            enable_reranker=enable_reranker,
        )

        with open(law_path, "r", encoding="utf-8") as f:
            laws = [json.loads(line) for line in f]
        law_texts = [
            f"{law.get('name', '')}\n{law.get('content', '')}"
            for law in laws
            if law.get("name", "").strip() or law.get("content", "").strip()
        ]

        with open(question_file_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]

        retriever = BGEM3ProgressiveRetriever(
            model_name=model_name or "BAAI/bge-m3",
            reranker_model_name=reranker_model_name,
            use_fp16=use_fp16,
            cache_dir=cache_dir,
            normalize_dense=normalize_dense,
            components=ProgressiveComponents(
                enable_hyde=enable_hyde,
                enable_reranker=enable_reranker,
            ),
            llm_config=llm_config,
            hyde_num_variants=hyde_num_variants,
            hyde_temperature=hyde_temperature,
            hyde_max_retries=hyde_max_retries,
            hyde_cache_dir=hyde_cache_dir,
            query_weight=query_weight,
            hyde_weight=hyde_weight,
            cross_fusion=cross_fusion,
            cross_rrf_k=cross_rrf_k,
            rerank_query_source=rerank_query_source,
        )
        retriever.build_or_load(
            law_texts,
            batch_size=batch_size,
            max_length=max_length,
        )

        query_map = []
        all_queries = []
        for d_idx, d in enumerate(data):
            for conv_idx, conv in enumerate(d.get("conversation", [])):
                all_queries.append(conv["question"]["content"])
                query_map.append((d_idx, conv_idx))

        label = log_label or "[BGE-M3 Progressive]"
        print(
            f"{label} "
            f"hyde={enable_hyde}, reranker={enable_reranker}, "
            f"queries={len(all_queries)}"
        )
        all_results = retriever.search_batch(
            queries=all_queries,
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

        for (d_idx, conv_idx), ranked in zip(query_map, all_results):
            conv = data[d_idx]["conversation"][conv_idx]
            conv["question"]["recall"] = [
                {"article": laws[int(doc_id)], "score": float(score)}
                for doc_id, score in ranked
            ]

        with open(res_path, "w", encoding="utf-8") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    def pipeline_bge_m3_colbert(
        self,
        question_file_path: str,
        law_path: str,
        model_name: str = None,
        top_k: int = 10,
        dense_top_k: int = 50,
        sparse_top_k: int = 50,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        colbert_weight: float = 1.0,
        batch_size: int = 8,
        max_length: int = 8192,
        cache_dir: str = "data/retrieval/bge_m3_colbert",
        use_fp16: bool = True,
        normalize_dense: bool = True,
        normalize_colbert: bool = True,
        fusion: str = "rrf",
        rrf_k: int = 60,
    ):
        """
        BGE-M3 三路混合检索（dense + sparse + ColBERT MaxSim）：
        - dense  ：FAISS（IndexFlatIP）粗检索
        - sparse ：BGE-M3 sparse_vecs 倒排累加
        - colbert：对 dense∪sparse 候选集执行 MaxSim 细粒度评分
        - fusion ：三路 RRF 或 MinMax 加权融合
        - 结果写入：data/retrieval/res/retrieval_bge-m3_colbert.jsonl
        """
        res_path = "data/retrieval/res/retrieval_bge-m3_colbert.jsonl"

        with open(law_path, "r", encoding="utf-8") as f:
            laws = [json.loads(line) for line in f]
        law_texts = [
            f"{law.get('name', '')}\n{law.get('content', '')}"
            for law in laws
            if law.get("name", "").strip() or law.get("content", "").strip()
        ]

        with open(question_file_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]

        retriever = self._create_bge_m3_colbert_retriever(
            model_name=model_name or "BAAI/bge-m3",
            use_fp16=use_fp16,
            cache_dir=cache_dir,
            normalize_dense=normalize_dense,
            normalize_colbert=normalize_colbert,
        )
        retriever.build_or_load(law_texts, batch_size=batch_size, max_length=max_length)

        # 收集所有 queries 和位置映射
        query_map = []  # [(d_idx, conv_idx)]
        all_queries = []
        for d_idx, d in enumerate(data):
            for conv_idx, conv in enumerate(d.get("conversation", [])):
                all_queries.append(conv["question"]["content"])
                query_map.append((d_idx, conv_idx))

        # 批量三路检索（一次 GPU encode + 一次 FAISS batch search + 逐 query ColBERT MaxSim）
        print(f"[BGE-M3 ColBERT] 批量检索 {len(all_queries)} 条 queries...")
        all_results = retriever.hybrid_search_batch(
            queries=all_queries,
            top_k=top_k,
            dense_top_k=dense_top_k,
            sparse_top_k=sparse_top_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            colbert_weight=colbert_weight,
            max_length=max_length,
            batch_size=batch_size,
            fusion=fusion,
            rrf_k=rrf_k,
        )

        # 将结果写回 data 结构
        for (d_idx, conv_idx), ranked in zip(query_map, all_results):
            conv = data[d_idx]["conversation"][conv_idx]
            conv["question"]["recall"] = [
                {"article": laws[int(doc_id)], "score": float(score)}
                for doc_id, score in ranked
            ]

        with open(res_path, "w", encoding="utf-8") as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

        print(f"[BGE-M3 ColBERT] 检索完成，结果已写入 {res_path}")

    def init_dir(self):
        os.makedirs("data/retrieval/res", exist_ok=True)
        os.makedirs("data/retrieval/npy", exist_ok=True)
