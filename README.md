# RAG pipeline — comparing chunking / embedding / retrieval methods

School project (4th year IIR, TP RAG with Dr. EL MKHALET MOUNA). I started from the TP notebooks and turned the whole thing into a small website so I could test every method on the same corpus and see which one actually works best.

Basically you paste a text + a question, and the app runs the full RAG pipeline and shows you the results side by side with charts. No black box, you see every step.

## How it works

You have 4 buttons on the page:

1. Chunking — cuts the text with 7 different methods (fixed size, recursive, token based, by sentence, sliding window, by paragraph, semantic)
2. Chunking metrics — same 7 methods but with stats (nb chunks, avg length, coherence score) and it tells you the best one
3. Embedding — converts the best chunks into vectors with 5 methods (tfidf, word2vec, fasttext, doc2vec, sentence-bert)
4. Embedding & Retrieval metrics — runs embeddings + retrieval (cosine brute, faiss, bm25, hybrid) and shows precision@k / recall@k / MRR

Then at the bottom you have the real RAG query: it retrieves the top-k chunks and builds a response from them.

The "best method" for chunking is just `coherence - 0.01 * nb_chunks`, nothing fancy, I took it from the TP. It penalizes methods that make way too many tiny chunks.

## Project layout

```
backend/app.py      -> FastAPI, all the logic (chunking + embedding + retrieval)
frontend/index.html -> the page
frontend/app.js     -> calls the API, draws tables and charts
frontend/style.css  -> nothing special
requirements.txt
run.bat             -> double-click to start
```

I removed the notebooks and the debug capture files, everything useful is already in `app.py`.

## Run it

You need Python with the venv already in the folder, or just install deps yourself.

```bat
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

Then open http://127.0.0.1:8000/app
API docs are at http://127.0.0.1:8000/docs if you want to test endpoints manually.

Or just double-click `run.bat`.

## Stuff I ran into

- First click is slow (10-30s), that's normal, it's downloading/loading `all-MiniLM-L6-v2`. Next clicks are fast.
- I disabled glove + bert + dpr + reranking by default. They need to download 60-400MB on first run and they freeze the whole UI. You can still call them manually via `/docs` with `methods: ["glove", "bert"]` if you want them for the report, but don't put them in the default path.
- Don't double-click the buttons. One click = one heavy job, two clicks = two jobs in parallel and everything hangs. I added loading states but still, wait for it.
- `GET /vectorial` always uses FAISS + default corpus, it ignores what you typed. Use `POST /query` if you want your own corpus and method.

## TODO if I continue

- plug a real LLM (right now the answer is a stub, I want to use Ollama llama3)
- upload PDFs instead of copy-paste text
- keep vectors in Chroma or Qdrant instead of recomputing every time
- proper eval set instead of using chunk[0] as ground truth
