from sentence_transformers import SentenceTransformer
from modelscope import snapshot_download
from openai import OpenAI
import httpx
import faiss
from tqdm import tqdm
import numpy as np
import os

class DenseRetriever:
    def __init__(self, api_key=None, base_url=None):
        self.api_key = api_key
        self.base_url = base_url
        self.model = None

    def load_model(self, model_type, model_name=None):
        if model_type == "BGE-base-zh":
            model_dir = snapshot_download(
                "AI-ModelScope/bge-base-zh-v1.5", revision="master"
            )
            self.model = SentenceTransformer(model_dir, trust_remote_code=True)
        elif model_type == "Qwen2-1.5B": #GTE model
            model_dir = snapshot_download("iic/gte_Qwen2-1.5B-instruct")
            self.model = SentenceTransformer(model_dir, trust_remote_code=True)
        elif model_type in {"BGE-M3-Dense", "bge-m3-dense", "bge_m3_dense"}:
            import torch
            from FlagEmbedding import BGEM3FlagModel

            model_ref = model_name or "BAAI/bge-m3"
            if torch.cuda.is_available():
                try:
                    torch.zeros(1).cuda()
                    device = "cuda"
                except Exception:
                    print("Warning: CUDA device detected but not compatible with current PyTorch. Falling back to CPU.")
                    device = "cpu"
            else:
                device = "cpu"

            self.model = BGEM3FlagModel(
                model_ref,
                use_fp16=(device == "cuda"),
                device=device,
            )
        elif model_type == "openai":
            self.model = None

    def _BGE_embedding(self, texts: list):
        embeddings = self.model.encode(texts)
        return embeddings

    def _Qwen2_embedding(self, texts: list):
        embeddings = self.model.encode(texts)
        return embeddings

    def _openai_embedding(self, texts: list, model_name):
        client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            http_client=httpx.Client(
                base_url=self.base_url,
                follow_redirects=True,
            ),
        )

        response = client.embeddings.create(
            input=texts,
            model=model_name,
        )
        embeddings = [data.embedding for data in response.data]
        return embeddings
        
    def embed(self, texts, model_type, model_name=None, batch_size=8, checkpoint_path=None):
        """
        向量化文本，支持断点续传
        
        Args:
            texts: 待向量化的文本列表
            model_type: 模型类型
            model_name: 模型名称（用于 openai）
            batch_size: 批次大小
            checkpoint_path: 断点文件路径（.npy格式），如果提供则支持断点续传
        """
        start_idx = 0
        embeddings = []
        
        # 如果提供了断点路径且文件存在，尝试加载已处理的部分
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                embeddings = np.load(checkpoint_path).tolist()
                start_idx = len(embeddings)
                print(f"从断点继续：已处理 {start_idx}/{len(texts)} 条数据")
            except Exception as e:
                print(f"加载断点文件失败，将从头开始: {e}")
                start_idx = 0
                embeddings = []
        
        if model_type in ["BGE-base-zh", "Qwen2-1.5B"]:
            self.load_model(model_type, model_name=model_name)
            for i in tqdm(range(start_idx, len(texts), batch_size), desc="向量化中", initial=start_idx, total=len(texts)):
                batch = texts[i : i + batch_size]
                batch_embeddings = self.model.encode(batch)
                embeddings.extend(batch_embeddings)
                
                # 保存断点
                if checkpoint_path:
                    np.save(checkpoint_path, np.array(embeddings))
            return np.array(embeddings)
        elif model_type in {"BGE-M3-Dense", "bge-m3-dense", "bge_m3_dense"}:
            self.load_model(model_type, model_name=model_name)
            for i in tqdm(range(start_idx, len(texts), batch_size), desc="向量化中", initial=start_idx, total=len(texts)):
                batch = texts[i : i + batch_size]
                batch_output = self.model.encode(
                    batch,
                    batch_size=len(batch),
                    return_dense=True,
                    return_sparse=False,
                    return_colbert_vecs=False,
                )
                batch_embeddings = batch_output["dense_vecs"]
                embeddings.extend(batch_embeddings)

                # 保存断点
                if checkpoint_path:
                    np.save(checkpoint_path, np.array(embeddings))
            return np.array(embeddings)
        elif model_type == "openai":
            for i in tqdm(range(start_idx, len(texts), batch_size), desc="向量化中", initial=start_idx, total=len(texts)):
                batch = texts[i : i + batch_size]
                # 批量调用 API，而不是逐个调用
                batch_embeddings = self._openai_embedding(batch, model_name)
                embeddings.extend(batch_embeddings)
                
                # 保存断点
                if checkpoint_path:
                    np.save(checkpoint_path, np.array(embeddings))
            return np.array(embeddings)
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @staticmethod
    def save_faiss(embeddings, faiss_type, save_path="index.faiss"):
        dim = embeddings.shape[1]
        
        if faiss_type == "FlatIP":
            index = faiss.IndexFlatIP(dim)
        elif faiss_type == "HNSW":
            index = faiss.IndexHNSWFlat(dim, 64)
        elif faiss_type == "IVF":
            nlist = min(128, int(np.sqrt(len(embeddings))))
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            index.train(embeddings.astype('float32'))
            index.nprobe = min(8, nlist//4)
        else:
            raise ValueError(f"Unsupported FAISS type: {faiss_type}")
        
        index.add(embeddings.astype('float32'))
        faiss.write_index(index, save_path)
