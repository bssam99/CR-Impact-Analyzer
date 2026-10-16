# CR Impact Analyzer

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-text%20splitters-1C3C3C?logo=langchain&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-vector%20store-FF6F00)
![Ollama](https://img.shields.io/badge/Ollama-local%20LLMs-000000)

A local **RAG (Retrieval-Augmented Generation) pipeline** that reads a system's technical documentation (PDFs) and generates a structured **impact analysis** for a Change Request (CR) — affected components, workflow changes, dependencies, risks, and test scope — grounded only in the retrieved documentation.

Everything runs on your machine: embeddings with **Sentence-Transformers**, a persistent **ChromaDB** vector store, and local LLMs through **Ollama**. No API keys and no data leaving your machine — after the one-time model downloads, it works fully offline. Two evaluation notebooks compare LLMs and search over the RAG settings (embedding model, chunk size/overlap, top-k, LLM).

> **Data:** no documents or CRs are included in this repo. The notebooks ship with placeholders (`CR_NUMBER` / `CR_DESCRIPTION` in Notebook 3, `EVAL_DATASET` in Notebooks 4–5) — add your own PDFs and CRs. The prompts in Notebooks 3 and 5 are worded for an order-management system (OMS); adjust the role line for your own domain.

## How it works

```
PDFs  ──►  extract text  ──►  chunk  ──►  embed  ──►  ChromaDB (persistent vector store)
                                                              │
CR description  ──►  embed  ──►  hybrid retrieval: targeted (same CR) + general (rest of the docs)
                                                              │
                                  retrieved excerpts + CR  ──►  local LLM (Ollama)
                                                              │
                                  structured impact analysis (Markdown, saved to analyses/)
```

**Hybrid retrieval.** Each chunk is tagged with a CR number taken from its PDF filename (e.g. `CR-2024-210.pdf`; `PM-…` names work too, anything else is tagged `UNKNOWN`). Notebook 3 first retrieves the top chunks tagged with the CR you're analyzing (6 by default), then adds the top chunks from the rest of the knowledge base (4 by default) — so the model sees both the CR's own text and the surrounding system documentation it affects. If the CR's own document isn't indexed, only the general chunks are used.

## Notebooks

| # | Notebook | Purpose |
|---|----------|---------|
| 1 | [`01_setup_and_config.ipynb`](notebooks/01_setup_and_config.ipynb) | Environment check — imports, Ollama connectivity, creates the `pdfs/` and `vector_store/` folders |
| 2 | [`02_build_vectordb.ipynb`](notebooks/02_build_vectordb.ipynb) | PDFs → text → chunks → embeddings → ChromaDB. Defaults: `all-mpnet-base-v2`, 800-char chunks, 150 overlap |
| 3 | [`03_analyze_cr.ipynb`](notebooks/03_analyze_cr.ipynb) | Main workflow — enter a CR, get a 7-section impact analysis (default LLM: `gemma:7b`) |
| 4 | [`04_evaluate_models.ipynb`](notebooks/04_evaluate_models.ipynb) | Runs several Ollama LLMs on your evaluation CRs and scores them (faithfulness, answer relevancy, structure, BERTScore, ROUGE-L, speed) |
| 5 | [`05_hyperparameter_evaluation.ipynb`](notebooks/05_hyperparameter_evaluation.ipynb) | Grid search over embedding model, chunk size/overlap, targeted/general top-k, and LLM; prints the best settings to copy into Notebooks 2 and 3 |

Each notebook has its own configuration cell at the top — settings aren't shared between notebooks, so keep `EMBEDDING_MODEL` the same in the notebook that builds the vector store (2) and the ones that query it (3, 4).

Retrieval was tuned over several iterations: bigger chunks to keep procedure steps intact, a stronger embedding model, and hybrid CR-targeted + general retrieval with fewer, more relevant chunks.

## Results

The charts below were produced on a document set that isn't included in this repo.

**Model comparison** — weighted score (faithfulness 30%, answer relevancy 25%, structure 20%, BERTScore 15%, ROUGE-L 10%), with average generation time next to each bar:

![Model evaluation results](results/model_evaluation.png)

`gemma:7b` came out on top, narrowly ahead of `llama3` and `mistral` at a similar speed — despite being smaller than `qwen2.5:14b` or `phi4`.

**Hyperparameter search** — average overall score for each value of each parameter, across the full grid:

![Hyperparameter evaluation results](results/hyperparameter_evaluation.png)

Key findings: `all-mpnet-base-v2` and `BAAI/bge-m3` scored highest among the embedding models, with the paraphrase-multilingual models lowest; 1200-char chunks scored slightly higher than 800 and 500, likely because they keep full procedure steps intact; and the LLM choice moved the score far more than any retrieval setting, where differences were small. Notebook 2 still defaults to 800-char chunks — to use the best configuration, copy the values Notebook 5 prints into Notebooks 2 and 3.

**Evaluation notes**
- Faithfulness, answer relevancy, and structure are lightweight keyword-overlap heuristics, not LLM-judged metrics.
- Notebook 4 uses plain top-k retrieval and a shorter 4-section prompt, not Notebook 3's hybrid retrieval and 7-section prompt.
- Notebook 5's semantic score is computed with the embedding model being tested, so the embedding comparison is indicative rather than strict.
- The default grid in Notebook 5 is large (36 vector stores and thousands of LLM calls) — trim `PARAM_GRID` before running.

## Setup

**Prerequisites**
- Python 3.11+
- [Ollama](https://ollama.com) installed and running (`ollama serve`), with the models you want pulled — Notebook 3 uses `gemma:7b` by default: `ollama pull gemma:7b`
- Internet on the first run only, to download the embedding models from Hugging Face

**Install**
```bash
git clone https://github.com/bssam99/cr-impact-analyzer.git
cd cr-impact-analyzer
pip install -r requirements.txt
```

**Run**

Start Jupyter from the repo (`jupyter lab`) and open the `notebooks/` folder. Paths are relative to the notebooks, so `pdfs/`, `vector_store/`, `analyses/`, and `evaluations/` are created inside `notebooks/`.

1. Run `01_setup_and_config.ipynb` once to check the environment and create the folders.
2. Put your PDFs in `notebooks/pdfs/`. Name each CR document after its CR number (e.g. `CR-2024-210.pdf`) so targeted retrieval can find it. Only PDFs with a text layer are read — scanned PDFs are skipped.
3. Run `02_build_vectordb.ipynb` to build the vector store. Re-run it whenever you add PDFs or change the embedding or chunk settings.
4. Run `03_analyze_cr.ipynb` for each new CR: set `CR_NUMBER` (the same tag as the filename, e.g. `CR-2024-210`) and `CR_DESCRIPTION`. A short, clean summary of the CR (2–5 sentences) retrieves better than raw text pasted from the PDF.
5. (Optional) Fill in `EVAL_DATASET` (CRs + reference answers) and run Notebook 4 and/or 5. Notebook 4's `EMBEDDING_MODEL` (`all-MiniLM-L6-v2`) must match the model the vector store was built with — rebuild it in Notebook 2 with the same model, or change the value in Notebook 4. Notebook 5 builds its own vector stores in `notebooks/eval_stores/`.

## Project structure

```
notebooks/         the five notebooks (data folders are created here at runtime)
results/           evaluation charts shown above
requirements.txt   pinned dependencies
```

## Tech stack

`Python` · `Sentence-Transformers` · `ChromaDB` · `Ollama` · `PyMuPDF` · `LangChain text splitters` · `pandas` · `matplotlib`/`seaborn` · `rouge-score` · `bert-score`

## License

[MIT](LICENSE)

## Author

**Bassam Ahmed** · [LinkedIn](https://www.linkedin.com/in/bssam-ahmed) · [GitHub](https://github.com/bssam99)
