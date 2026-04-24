# LexRAG: Unified Progressive Retrieval Framework for Multi-turn Legal Consultation

<p align="center">
  📖 中文 | <a href="./README.md">English</a>
</p>

本仓库对应论文 *Research on the Optimization of Retrieval Strategy for Multi-round Legal Consultation System Based on RAG* 的核心实现，重点面向中文多轮法律咨询场景下的法条检索优化。

本项目基于清华大学团队提出的 LexRAG 数据集与 LexiT 工具框架开展二次研究与实现。原始 LexRAG 项目提供了中文多轮法律咨询场景、法条语料、查询重写思路以及模块化 RAG pipeline；本仓库在此基础上聚焦于 **retrieval-side optimization**，围绕 `BGE-M3` 构建统一渐进式检索框架，并比较 hybrid retrieval、HyDE query augmentation 与 Cross-Encoder reranking 在法条检索任务中的作用。

## 1. 项目来源与改动说明

本项目不是从零构建一个全新的法律 RAG benchmark，而是在 LexRAG / LexiT 的基础上进行检索方向的扩展与实验优化。原始项目可参考：`https://github.com/CSHaitao/LexRAG`。

本项目主要借鉴了以下部分：

- LexRAG 的中文多轮法律咨询任务设定。
- LexRAG 提供的多轮对话数据、法条语料库与检索评估目标。
- LexiT 的模块化 pipeline 思路，包括 processor、retriever、generator 和 evaluator 的基本组织方式。
- 查询重写作为多轮上下文处理的共享预处理步骤。

本项目在原有基础上主要做了以下修改与扩展：

- 将研究重点从完整 RAG toolkit 转向法条检索策略优化。
- 新增 `BGE-M3` dense + sparse hybrid retrieval，实现统一编码器下的双路检索。
- 新增 HyDE-based query augmentation，用法律风格假设文本缓解用户表达与法条文本之间的语义差距。
- 新增 Cross-Encoder reranking 路径，用于评估候选侧二阶段重排在该任务中的作用。
- 将 hybrid retrieval、HyDE、reranker 统一到 `BGEM3ProgressiveRetriever` 中，通过开关组合复现实验设置。
- 新增 `run_progressive_retrieval.py`，作为统一 progressive retrieval framework 的主要运行入口。

因此，这个仓库应被理解为：基于 LexRAG 数据和 LexiT 框架的检索优化实验实现，而不是原始 LexRAG / LexiT 项目的官方版本。

## 2. 研究目标与核心结论

论文关注的问题是：

- 多轮法律咨询中，当前轮问题常常依赖前文上下文，存在指代、省略和信息缺失。
- 用户自然语言表达与正式法条语言之间存在明显的术语与文风差距。
- 单一路径检索难以同时兼顾语义匹配能力和法律术语匹配能力。

围绕这些问题，论文与代码共同实现了一个渐进式检索框架：

1. 先用查询重写把多轮对话转为可检索的独立问题。
2. 以 `BGE-M3` 为统一编码器，同时执行 dense + sparse 混合检索。
3. 可选地在检索前加入 `HyDE` 生成法律风格假设文本，扩展查询表达。
4. 可选地在检索后加入 `Cross-Encoder` 对候选法条进行二阶段重排。

根据论文实验结果：

- `Hybrid Retrieval` 整体优于纯 dense 检索。
- `HyDE-Enhanced Hybrid Retrieval` 是当前实验设置下效果最好的方案。
- `Cross-Encoder Reranking` 在当前数据集和参数设置下没有带来增益。

## 3. 框架总览

统一框架对应如下流程：

```text
Multi-turn dialogue
  -> Query Rewriting
  -> Optional HyDE augmentation
  -> BGE-M3 dense + sparse hybrid retrieval
  -> Optional Cross-Encoder reranking
  -> Top-K statute results
  -> Retrieval evaluation
```

这个设计和论文第 3 章是一一对应的，只是代码里把它做成了一个统一编排器，而不是分散的实验脚本。

<p align="center">
  <img src="image/image1.png" alt="Progressive retrieval framework used in this study" width="760">
  <br>
  <em>Progressive Retrieval Framework Used in This Study</em>
</p>

## 4. 论文设计与代码映射

| 论文模块 | 代码位置 | 作用 |
| --- | --- | --- |
| Query Rewriting | `src/process/rewriter.py` | 将多轮上下文整合为独立可检索问题 |
| 统一检索框架 | `src/retrieval/bge_m3_progressive_retrieval.py` | 定义渐进式检索组件、配置和实验入口 |
| 检索运行入口 | `run_progressive_retrieval.py` | 通过开关和参数运行不同模式 |
| 统一检索管线 | `src/retrieval/run_retrieval.py` | 将 progressive retriever 接入现有 pipeline |
| 检索评估 | `src/eval/evaluator.py` | 计算 Recall / Precision / F1 / NDCG / MRR |
| 默认模型配置 | `src/config/config.py` | 配置 HyDE 或生成模块所需的 LLM 参数 |

<p align="center">
  <img src="image/image2.png" alt="Query-side and candidate-side enhancement paths over the BGE-M3-based hybrid retrieval backbone" width="760">
  <br>
  <em>Query-side and Candidate-side Enhancement Paths over the BGE-M3-based Hybrid Retrieval Backbone</em>
</p>

### 4.1 `bge_m3_progressive_retrieval.py` 做了什么

这个文件是整个项目最关键的统一实现，核心点包括：

- `BGEM3HybridRetriever`
  - 使用 `BGE-M3` 同时生成 dense 和 sparse 表示。
  - dense 分支使用 `FAISS IndexFlatIP` 检索。
  - sparse 分支使用倒排索引与词权重累加。
  - 支持 `rrf` 和 `minmax` 两种融合方式。

- `HyDEGenerator` / `HyDEAugmentationComponent`
  - 为原始 query 生成法律风格假设文本。
  - 支持缓存，避免重复生成。
  - 通过跨 query 融合把原始 query 和 HyDE 变体一起汇总。

- `RerankerComponent`
  - 使用 `BAAI/bge-reranker-v2-m3` 对候选法条二阶段重排。

- `BGEM3ProgressiveRetriever`
  - 统一组织四种模式：
    - Hybrid only
    - HyDE + Hybrid
    - Hybrid + Reranker
    - HyDE + Hybrid + Reranker

- `build_progressive_run_config()` / `run_progressive_experiment()`
  - 为不同模式生成默认参数。
  - 调用检索与评估流水线，适合直接复现实验。

## 5. 数据与输入格式

仓库中的核心数据文件：

- `data/law_library.jsonl`
  - 法条语料库。
- `data/dataset.json`
  - 多轮法律咨询数据。
- `data/rewrite_question.jsonl`
  - 经过查询重写后的检索输入。

如果你已经完成查询重写，可以直接使用 `data/rewrite_question.jsonl` 运行 progressive retrieval；如果还没有，可以先执行查询重写。

## 6. 环境准备

推荐环境：

- Python `3.9` 或 `3.10`
- 有 GPU 时优先使用 GPU 版本 `faiss` 与 `torch`

安装依赖：

```bash
pip install -r requirements.txt
```

如果你是 CPU 环境，可以把 `requirements.txt` 中的 `faiss-gpu` 替换为 `faiss-cpu`。

### 6.1 HyDE 模式额外说明

当你启用 `HyDE` 时，需要一个可用的 LLM 配置。默认情况下：

- `run_progressive_retrieval.py` 中 `LLM_CONFIG = None`
- 这会回退到 `src/config/config.py` 里的 `Config._default_configs["qwen"]`

因此，如果你要跑 HyDE，请至少完成以下其中一种配置方式：

1. 直接修改 `src/config/config.py` 里的 `qwen` 配置，填入 `api_key`。
2. 在 `run_progressive_retrieval.py` 里自定义 `LLM_CONFIG` 并覆盖默认值。

## 7. 先做查询重写

如果还没有生成 `data/rewrite_question.jsonl`，可以先运行查询重写：

```python
from src.pipeline import ProcessorPipeline

pipeline = ProcessorPipeline(model_type="qwen")
pipeline.run_processor(
    process_type="rewrite_question",
    original_data_path="data/dataset.json",
    output_path="data/rewrite_question.jsonl",
    max_retries=5,
    max_parallel=32,
    batch_size=20
)
```

这一阶段对应论文中的共享预处理步骤，用来解决多轮对话中的上下文依赖与信息省略问题。

## 8. 如何运行 Unified Progressive Retrieval Framework

### 8.1 直接运行

主入口文件是：

- `run_progressive_retrieval.py`

运行命令：

```bash
python run_progressive_retrieval.py
```

这个脚本会做两件事：

1. 调用 `run_progressive_experiment()` 执行检索。
2. 自动调用 `EvaluatorPipeline` 输出检索评估结果。

### 8.2 通过两个开关控制模式

在 `run_progressive_retrieval.py` 顶部修改：

```python
ENABLE_HYDE = True
ENABLE_RERANKER = False
```

四种组合如下：

| `ENABLE_HYDE` | `ENABLE_RERANKER` | 运行模式 | 输出结果文件 |
| --- | --- | --- | --- |
| `False` | `False` | Hybrid Retrieval | `data/retrieval/res/retrieval_bge-m3_hybrid.jsonl` |
| `True` | `False` | HyDE + Hybrid Retrieval | `data/retrieval/res/retrieval_bge-m3_hybrid_hyde.jsonl` |
| `False` | `True` | Hybrid + Reranker | `data/retrieval/res/retrieval_bge-m3_hybrid_rerank.jsonl` |
| `True` | `True` | HyDE + Hybrid + Reranker | `data/retrieval/res/retrieval_bge-m3_hybrid_hyde_rerank.jsonl` |

### 8.3 当前脚本默认含义

当前 `run_progressive_retrieval.py` 的默认设置是：

```python
ENABLE_HYDE = True
ENABLE_RERANKER = False
```

也就是默认运行论文中表现最好的主模式：**HyDE-Enhanced Hybrid Retrieval**。

### 8.4 关键参数说明

`run_progressive_retrieval.py` 中最常调的参数如下：

| 参数 | 含义 |
| --- | --- |
| `TOP_K` | 最终返回的法条数量 |
| `DENSE_TOP_K` | dense 分支候选深度 |
| `SPARSE_TOP_K` | sparse 分支候选深度 |
| `DENSE_WEIGHT` / `SPARSE_WEIGHT` | dense 与 sparse 融合权重 |
| `FUSION` | 分支内融合方式，支持 `rrf` 或 `minmax` |
| `HYDE_NUM_VARIANTS` | HyDE 生成的假设文本数量 |
| `QUERY_WEIGHT` / `HYDE_WEIGHT` | 原始 query 与 HyDE query 的跨 query 融合权重 |
| `CROSS_FUSION` | 原始 query 与 HyDE query 的融合方式 |
| `RERANK_TOP_K` | 重排阶段接收的候选数 |
| `RERANK_QUERY_SOURCE` | 重排时使用原 query 还是 HyDE 第一变体 |

### 8.5 论文对齐的默认实验设置

`src/retrieval/bge_m3_progressive_retrieval.py` 中的 `build_progressive_run_config()` 已经把论文中的默认实验配置编码进去了。特别是：

- Hybrid 模式默认使用：
  - `dense_weight=0.8`
  - `sparse_weight=0.2`
  - `fusion="minmax"`

- HyDE 模式默认使用：
  - `hyde_num_variants=2`
  - `query_weight=1.0`
  - `hyde_weight=0.75`
  - `cross_fusion="minmax"`

因此，如果你只是希望复现论文主实验，直接修改开关并运行 `run_progressive_retrieval.py` 即可。

## 9. 运行后会产出什么

### 9.1 检索结果

输出到：

- `data/retrieval/res/`

每条对话中的每一轮问题都会被写回：

- `question.recall`

其中包含检索到的法条及其分数。

### 9.2 索引与缓存

运行过程中还会生成：

- `data/retrieval/bge_m3*/`
  - dense FAISS 索引、倒排索引、meta 信息
- `data/retrieval/hyde_cache/`
  - HyDE 生成缓存

### 9.3 评估报告

评估报告默认追加写入：

- `data/retrieval/report.jsonl`

默认评估指标为：

- `recall`
- `precision`
- `f1`
- `ndcg`
- `mrr`

默认 `k` 为：

- `[1, 3, 5, 10]`

## 10. 论文实验结果摘要

下面这组结果来自论文中五种检索设置的总体比较：

**Comparison of Five Retrieval Strategies**

| Strategy | Recall@1 | Recall@3 | Recall@5 | Recall@10 | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense Retrieval (Baseline) | 0.1451 | 0.2493 | 0.3011 | 0.3757 | 0.1576 | 0.2174 | 0.2398 | 0.2651 |
| Hybrid Retrieval | 0.1480 | 0.2593 | 0.3088 | 0.3855 | 0.1607 | 0.2246 | 0.2459 | 0.2718 |
| Hybrid Retrieval + Cross-Encoder | 0.1438 | 0.2557 | 0.3071 | 0.3824 | 0.1561 | 0.2209 | 0.2428 | 0.2684 |
| HyDE-Enhanced Hybrid Retrieval | **0.1917** | **0.3161** | **0.3812** | **0.4701** | **0.2081** | **0.2777** | **0.3054** | **0.3349** |
| HyDE-Enhanced Hybrid Retrieval + Cross-Encoder | 0.1435 | 0.2570 | 0.3110 | 0.3868 | 0.1557 | 0.2215 | 0.2444 | 0.2701 |

从结果看，当前实验设置下最值得优先复现和展示的是：

- Query Rewriting + Hybrid Retrieval
- Query Rewriting + HyDE + Hybrid Retrieval

## 11. 参考引用

```bibtex
@thesis{huang2026progressivelegalretrieval,
  title={Research on the Optimization of Retrieval Strategy for Multi-round Legal Consultation System Based on RAG},
  author={Huang, Qikang},
  year={2026},
  school={Communication University of China, Hainan International College}
}
```
