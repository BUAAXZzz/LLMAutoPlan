# LLMAutoPlan
📌 Overview

This repository implements a locally deployed, retrieval-augmented large language model (RE-LLM) framework for adaptive weight optimization in Seed Implant Brachytherapy (SIBT).

The framework integrates:

- A heuristic dose-based optimization engine

- A locally deployed LLM (e.g., DeepSeek-R1-Distill-Qwen-14B)

- A structured clinical knowledge base

- A hybrid RAG retrieval pipeline

- An iterative closed-loop weight tuning workflow

🏗 System Architecture

The workflow consists of three tightly integrated modules:

1️⃣ Optimization Engine

2️⃣ LLM-based Evaluation & Weight Tuning

3️⃣ RAG Module (Hybrid Retriever)

📂 Repository Structure
```
.
├── run_autoplan_rag.py       # Iterative LLM + RAG optimization loop
├── rag_module.py             # Hybrid retrieval + RAG context provider
├── llm_qwen3.py              # LLM wrapper (not included here)
├── optim_engine/             # C++/CUDA dose optimization backend
├── knowledge_base/           # Clinical guidelines & references
├── vector_db/                # Chroma persistent embeddings
└── result_rag/               # Per-case optimization results
```

📜 Citation

If you use this work in your research, please cite:
```
@article{xiao2025rellm_sibt,
  title={An Iterative LLM Framework for SIBT Utilizing RAG-based Adaptive Weight Optimization},
  author={Xiao, Zhuo and Zhou, Fugen and Yao, Qinglong and Wang, Jingjing and Liu, Bo and Sun, Haitao and Ji, Zhe and Jiang, Yuliang and Wang, Junjie and Wu, Qiuwen},
  journal={arXiv preprint arXiv:2509.08407},
  year={2025}
}
```
