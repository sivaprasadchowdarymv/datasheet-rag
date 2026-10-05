# Datasheet RAG Agent v4

Ask questions about electronic-component datasheets (PDF) and get answers that are
**grounded in the document**, with clickable `[REF-n]` citations showing document,
page, section and the exact source text. If the answer is not in the documents the
app says **NOT FOUND IN DOCUMENT** instead of guessing.

Successor of the single-file `amalbro_ds_1.py` (v3). Every v3 feature is kept —
see [`docs/FEATURE_CHECKLIST.md`](docs/FEATURE_CHECKLIST.md).

| Mode | Where it runs | Embeddings | Answers | Vision |
|---|---|---|---|---|
| `local` | your PC | Ollama `nomic-embed-text` | Ollama `mistral` (streamed) | Ollama `llava:7b` |
| `cloud` | Streamlit Community Cloud (free), no Ollama | in-process ONNX `BAAI/bge-small-en-v1.5` | extractive (verbatim cited excerpts) — or any open model on a server you control via `LLM_BASE_URL` | off (figure captions still searchable) |
| `local` in Docker | Oracle Always Free VM / any always-on box | Ollama | Ollama | optional |

No paid API. No API key in the code. All models are open.

---

## 1. Run on your PC (local mode)

Requirements: Python 3.12, [Ollama](https://ollama.com) installed and running.

```bash
git clone https://github.com/<you>/datasheet-rag.git
cd datasheet-rag
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
ollama pull mistral
ollama pull nomic-embed-text
ollama pull llava:7b
copy .env.example .env        # Windows   (Linux/macOS: cp .env.example .env)
streamlit run app.py
```

Open http://localhost:8501 → **Documents** tab → upload PDFs → **Ask** tab.

Note: the cross-encoder reranker (fastembed) downloads a ~90 MB ONNX model on first
use. If it cannot (offline), the app falls back to a lexical reranker and shows a
warning in the sidebar — it never crashes.

Try cloud mode on your PC (no Ollama needed): set `APP_MODE=cloud` in `.env`.

## 2. Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

The suite is fully offline (hash embedder, lexical reranker, fake LLM) and covers PDF
parsing, retrieval, NOT FOUND, citation validation, cache save/reload, multi-document
citations, XSS escaping and a Streamlit UI smoke test.

## 3. Benchmark (original v3 vs v4, real measurements)

```bash
python benchmark.py --pdf LM317.pdf LT1085.pdf --questions questions.txt \
                    --original path/to/amalbro_ds_1.py --mode local
```

`questions.txt`: one question per line, optionally `question | expected text`
(use `NOT FOUND` as expected text for questions the datasheet cannot answer).
Results are written to `benchmark_results/*.md` and `*.json`. Run it with Ollama
running, otherwise the v3 side can only measure parsing.

---

## 4. Deploy publicly — beginner steps

**Which path?** (RECOMMENDATION)

* **Path A — Streamlit Community Cloud** (default, simplest, free, no credit card):
  public URL, works with your PC off. Answers are extractive (exact cited sentences
  from the datasheet) unless you add an LLM server (Path B can be that server).
* **Path B — Oracle Cloud Always Free VM + Docker**: the full Ollama stack
  (`mistral`, `nomic-embed-text`) running 24×7 on a free ARM VM. More setup; needs
  sign-up with a card for verification; free capacity is not always available in
  every region.

FACTS behind this choice (checked October 2026; free tiers change — re-check):
Streamlit Community Cloud gives an app roughly 0.078–2 CPU cores and 690 MB–2.7 GB
RAM, sleeps after 12 h without traffic, and its disk is not permanent. That is
too small to run a 7B LLM, so the cloud profile uses small ONNX models in-process.
Hugging Face Spaces now needs a paid plan to create Docker/Gradio Spaces, so it is
not used. Oracle Always Free Ampere A1 currently allows about 2 OCPU / 12 GB RAM and
may reclaim idle instances.

### STEP 1 — Install requirements locally
See section 1 (`pip install -r requirements.txt`, `ollama pull …`).

### STEP 2 — Run locally
`streamlit run app.py` → upload a datasheet → ask "What is the maximum output current?".

### STEP 3 — Test Ollama mode
Sidebar → **Status** must show Ollama ✅ and the models. Ask a question: the answer
streams token by token and has `[REF-n]` citations; hover/tap a citation to see
page + section + source text. Ask "What is the price?" → **NOT FOUND IN DOCUMENT**.
Optionally run `pytest -q`.

### STEP 4 — Create a GitHub repository
github.com → **New repository** → name `datasheet-rag` → *Public* (or Private) →
do **not** add a README (you already have one) → Create.

### STEP 5 — Push the code
```bash
git init
git add .
git status          # make sure .env, data/, cache/ and *.pdf are NOT listed
git commit -m "Datasheet RAG Agent v4"
git branch -M main
git remote add origin https://github.com/<you>/datasheet-rag.git
git push -u origin main
```

### STEP 6 — Configure deployment

**6a. Persistent storage (recommended, free).** Streamlit Cloud forgets files when
the app restarts. The app can mirror `data/` (chunks, FAISS indexes, feedback) to a
private Hugging Face dataset and restore it on start:

1. Create a free account at huggingface.co.
2. **New → Dataset** → name `datasheet-rag-data` → **Private** → Create.
3. **Settings → Access Tokens → Create new token → Fine-grained** → give *write*
   access to that one dataset only → copy the token (`hf_…`).

**6b. Secrets.** Open `.streamlit/secrets.toml.example`, fill in your values. You will
paste them in step 7. Choose a long `ADMIN_PASSWORD`; with `PUBLIC_UPLOADS = "false"`
only you (after unlocking) can upload, delete or re-index documents, while everyone
can ask questions.

**6c. (Optional) LLM-written answers on the public app.** Set `LLM_BASE_URL` to an
OpenAI-compatible endpoint you control — e.g. the Oracle VM from Path B
(`https://your-domain/…` behind Caddy with an API key) or any llama.cpp / vLLM server.
Leave it empty for extractive answers.

### STEP 7 — Deploy
1. Go to share.streamlit.io → sign in with GitHub.
2. **Create app → Deploy a public app from GitHub** → repository `datasheet-rag`,
   branch `main`, main file `app.py`.
3. **Advanced settings** → Python version **3.12** → paste your secrets
   (from step 6b) into the **Secrets** box → Save.
4. Click **Deploy**. First start installs packages and downloads two small ONNX
   models — wait until the page loads.

### STEP 8 — Get the public URL
The address bar shows `https://<something>.streamlit.app`. Open the app, unlock
admin, upload your datasheets once (indexing happens once; the indexes are mirrored
to your HF dataset). Share the URL.

### STEP 9 — Test from another network
Open the URL on a computer on a different network (office, friend's Wi-Fi) that has
**no Ollama installed**. Ask a question; check citations and the Sources panel.

### STEP 10 — Test from mobile data
Turn Wi-Fi off on your phone, open the URL on mobile data. Tap a `[REF-n]` citation —
the popup opens on tap (no hover needed).

### STEP 11 — Test with your PC completely switched OFF (mandatory)
Shut down your PC. From the phone, open the URL, ask a question, compare two
datasheets, open the **Browse** tab. Everything must work: nothing in Path A runs on
your PC. If the app was asleep (12 h without visitors) click the wake-up button and
wait ~1 minute; documents are restored from the HF dataset automatically.

### Path B — Oracle Cloud Always Free VM (full Ollama, 24×7)
1. Create an Oracle Cloud account (free tier; card used for identity verification).
2. **Compute → Instances → Create**: image *Ubuntu 24.04*, shape *Ampere
   VM.Standard.A1.Flex* with 2 OCPU / 12 GB → add your SSH key → Create.
3. **Networking → VCN → Security List**: allow ingress TCP 80 and 443
   (and 8501 if you skip HTTPS). On the VM: `sudo iptables -I INPUT -p tcp --dport 80 -j ACCEPT`
   (same for 443/8501) then `sudo netfilter-persistent save`.
4. SSH in, install Docker:
   `curl -fsSL https://get.docker.com | sh && sudo usermod -aG docker $USER` (log out/in).
5. `git clone https://github.com/<you>/datasheet-rag.git && cd datasheet-rag && cp .env.example .env`
   → edit `.env`: `ADMIN_PASSWORD`, `PUBLIC_UPLOADS=false`, optionally a smaller
   `LLM_MODEL` (e.g. `qwen2.5:3b`) for faster CPU answers.
6. `docker compose up -d --build` then `./scripts/pull_models.sh`.
7. Open `http://<vm-public-ip>:8501`. For HTTPS, point a (free) domain/DNS name at the
   VM, set `DOMAIN=` in `.env` and run `docker compose --profile https up -d`.
8. Run steps 9–11 above against this URL.

ASSUMPTION: a 7B model on 2 ARM cores answers noticeably slower than on a desktop
GPU/CPU; measure with `benchmark.py` on the VM and switch to a 3B model if needed.

---

## 5. Final acceptance tests

| # | Test | How |
|---|---|---|
| 1 | PC runs local mode | Step 3 |
| 2 | RAG works correctly | Ask 5 questions whose answers you know; `benchmark.py` with expected values |
| 3 | Citations work | Each answer sentence ends with `[REF-n]`; popup shows doc/page/section/text |
| 4 | Multiple PDFs work | Upload 2 datasheets, select both, "Compare the maximum output current of A and B" → citations from both |
| 5 | Public URL | Step 8 |
| 6 | Other network | Step 9 |
| 7 | No Ollama on client | Step 9 |
| 8 | Dev PC switched off | Step 11 |
| 9 | Public app still works | Step 11 |
| 10 | No paid API | Only open models / your own server are used; no billing configured anywhere |
| 11 | No hard-coded key | `git grep -nE "hf_[A-Za-z0-9]{10}|sk-[A-Za-z0-9]{10}"` returns nothing |
| 12 | Grounded answers | Ask something absent (price, a pin that doesn't exist) → NOT FOUND; Diagnostics tab shows evidence gate |

---

## 6. Configuration

All settings live in `config.py`; override any of them with an environment
variable, a Streamlit secret or `.env` (that order). Most useful keys:

| Key | Default | Meaning |
|---|---|---|
| `APP_MODE` | `local` | `local` / `cloud` profile |
| `LLM_PROVIDER` | `ollama` (local), `auto` (cloud) | `ollama`, `openai_compat`, `extractive`, `auto` |
| `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` | — | OpenAI-compatible open-model server |
| `OLLAMA_HOST`, `OLLAMA_AUTH_HEADER` | `http://localhost:11434` | local or remote Ollama |
| `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL` | per profile | `ollama`, `fastembed`, `sentence_transformers` |
| `RERANKER_PROVIDER`, `ENABLE_RERANKER` | `fastembed`, true | cross-encoder; falls back to lexical |
| `TOP_K_RETRIEVAL`, `TOP_K_FINAL` | 20, 6 | candidates per channel / evidence blocks sent to the LLM |
| `RERANK_MIN_SCORE`, `DENSE_MIN_SCORE`, `LEXICAL_MIN_COVERAGE` | 0.02, 0.45, 0.5 | evidence gate → NOT FOUND |
| `MAX_CHUNK_TOKENS`, `OVERLAP_TOKENS` | 400, 60 | chunking (changes trigger automatic re-index) |
| `HF_TOKEN`, `HF_DATASET_REPO` | — | enable the HF dataset mirror |
| `MAX_UPLOAD_MB`, `MAX_PAGES`, `QUERIES_PER_MINUTE` | 50, 400, 10 | abuse limits |
| `ADMIN_PASSWORD`, `PUBLIC_UPLOADS` | —, true | protect uploads/deletes |

**Several answer models (e.g. Kimi).** Besides the main LLM you can offer up to four
more in a sidebar menu "Answer model" with `LLM2_LABEL`, `LLM2_BASE_URL`, `LLM2_MODEL`,
`LLM2_API_KEY` … `LLM5_*`. Every visitor picks per question; retrieval, citations and the
NOT FOUND gate are identical for all of them, and "Extractive" is always offered.
Switching models never reloads the embedding or reranker models.

Changing the embedding model is safe: vectors are stored per model namespace and are
rebuilt from stored chunks (no PDF re-parse).

## 7. Licences and privacy
* PyMuPDF is AGPL-3.0: if you publish a modified network service, publish its source
  (a public GitHub repo satisfies this). Other dependencies are permissive.
* Uploaded datasheets are visible to every visitor of a public deployment. Use
  `PUBLIC_UPLOADS=false` and do not upload confidential documents.
* Streamlit Community Cloud apps are public by URL; the HF dataset must be **private**.

## 8. Project layout
See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the data flow and the purpose
and connections of every file.
