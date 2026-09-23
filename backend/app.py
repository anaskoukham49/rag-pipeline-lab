"""
TP3 - Forme du Projet RAG - FastAPI Backend
Dr. EL MKHALET MOUNA - 4ème IIR Intelligence Artificielle

Expose REST endpoints consumed by the frontend (HTML/React):

  POST /chunking               -> 7 chunking methods
  GET  /chunking/metrics       -> metrics comparatives de chunking
  POST /embedding              -> 7 méthodes d'embedding sur best_chunks
  GET  /embedding/metrics      -> métriques comparatives d'embedding
  POST /retrieval              -> 6 méthodes de retrieval
  GET  /retrieval/metrics      -> dashboard comparatif retrieval
  POST /query                  -> pipeline RAG complet (query -> réponse)
  GET  /vectorial              ->alias end-to-end (TP spec)
  GET  /health / /docs

Frontend choice : HTML (simple) or React — both call the same API.
CORS is enabled for local dev.
"""

import re
import pathlib
import numpy as np
from typing import List, Dict, Any, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Lazy imports — keep startup fast and tolerate missing heavy models
# ---------------------------------------------------------------------------
try:
    from sentence_transformers import SentenceTransformer
    _sbert_available = True
except Exception as e:
    SentenceTransformer = None  # type: ignore
    _sbert_available = False
    _sbert_error = str(e)

try:
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    cosine_similarity = None  # type: ignore

try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
except Exception:
    sent_tokenize = word_tokenize = None  # type: ignore

try:
    from langchain_text_splitters import (
        CharacterTextSplitter,
        RecursiveCharacterTextSplitter,
        TokenTextSplitter,
    )
except Exception:
    CharacterTextSplitter = RecursiveCharacterTextSplitter = TokenTextSplitter = None  # type: ignore

try:
    import faiss  # type: ignore
except Exception:
    faiss = None  # type: ignore

try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None  # type: ignore

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="TP3 RAG - FastAPI",
    description="Backend RAG : chunking / embedding / retrieval / génération. TP3 Forme du Projet - 4ème IIR.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Dataset par défaut (identique aux TP1/TP2)
# ---------------------------------------------------------------------------
DEFAULT_TEXT = """Artificial Intelligence is transforming civil engineering.
Machine learning can predict structural damage after earthquakes.
Deep learning models detect cracks in bridges and roads.
Structural health monitoring uses IoT sensors to track building integrity.
RAG systems combine retrieval and generation to answer domain-specific questions.
Chunking strategy has a direct impact on retrieval quality in RAG pipelines.
Embeddings represent text as dense vectors capturing semantic meaning.
Vector databases enable fast similarity search over large document collections."""

DEFAULT_QUERY = "How is AI used in civil engineering?"

# Global lazy singletons ----------------------------------------------------
_embed_model = None  # SentenceTransformer('all-MiniLM-L6-v2')
_dpr_model = None
_reranker = None
_glove_model = None


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        if not _sbert_available:
            raise RuntimeError(f"sentence-transformers not available: {_sbert_error}")
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def _sent_tokenize(text: str) -> List[str]:
    if sent_tokenize is not None:
        try:
            return sent_tokenize(text.strip())
        except Exception:
            pass
    # fallback — regex split on sentence terminators
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _word_tokenize(text: str) -> List[str]:
    if word_tokenize is not None:
        try:
            return word_tokenize(text)
        except Exception:
            pass
    return re.findall(r"\w+", text.lower())


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class ChunkingRequest(BaseModel):
    text: Optional[str] = None
    chunk_size: int = 100
    chunk_overlap: int = 20


class EmbeddingRequest(BaseModel):
    text: Optional[str] = None
    # If chunks are supplied we use them, else we compute best_chunks via chunking
    chunks: Optional[List[str]] = None
    methods: Optional[List[str]] = None  # subset to run


class RetrievalRequest(BaseModel):
    query: str = DEFAULT_QUERY
    text: Optional[str] = None
    method: Optional[str] = None  # if None -> run all
    k: int = 3
    alpha: float = 0.5  # for hybrid


class QueryRequest(BaseModel):
    query: str = DEFAULT_QUERY
    text: Optional[str] = None
    k: int = 3
    retrieval_method: str = "similarite_brute"


# ---------------------------------------------------------------------------
# Chunking — 7 méthodes (TP2 partie 1)
# ---------------------------------------------------------------------------
def chunk_fixed_size(text, chunk_size=100, chunk_overlap=0):
    if CharacterTextSplitter is None:
        # fallback: sliding window on chars
        res = []
        n = len(text)
        step = max(1, chunk_size - chunk_overlap)
        for i in range(0, n, step):
            res.append(text[i : i + chunk_size])
            if i + chunk_size >= n:
                break
        return [r for r in res if r.strip()]
    s = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator="")
    return s.split_text(text)


def chunk_recursive(text, chunk_size=100, chunk_overlap=20):
    if RecursiveCharacterTextSplitter is None:
        return chunk_fixed_size(text, chunk_size, chunk_overlap)
    s = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return s.split_text(text)


def chunk_token_based(text, chunk_size=20, chunk_overlap=5):
    if TokenTextSplitter is None:
        words = _word_tokenize(text)
        res, step = [], max(1, chunk_size - chunk_overlap)
        for i in range(0, len(words), step):
            res.append(" ".join(words[i : i + chunk_size]))
            if i + chunk_size >= len(words):
                break
        return res
    s = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return s.split_text(text)


def chunk_by_sentence(text):
    return [s.strip() for s in _sent_tokenize(text) if s.strip()]


def chunk_sliding_window(text, chunk_size=100, chunk_overlap=50):
    return chunk_fixed_size(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def chunk_by_paragraph(text):
    return [p.strip() for p in text.split("\n") if p.strip()]


def chunk_semantic(text, similarity_threshold=0.5):
    sentences = _sent_tokenize(text)
    if len(sentences) <= 1:
        return [s for s in sentences if s.strip()]
    try:
        model = get_embed_model()
        embs = model.encode(sentences)
        chunks, cur = [], [sentences[0]]
        for i in range(1, len(sentences)):
            sim = float(cosine_similarity([embs[i]], [embs[i - 1]])[0][0]) if cosine_similarity else 0.0
            if sim >= similarity_threshold:
                cur.append(sentences[i])
            else:
                chunks.append(" ".join(cur))
                cur = [sentences[i]]
        chunks.append(" ".join(cur))
        return chunks
    except Exception:
        # degrade gracefully to sentence split
        return sentences


CHUNKING_FUNCS: Dict[str, Any] = {
    "fixed_size": lambda t, cs, co: chunk_fixed_size(t, chunk_size=100, chunk_overlap=0),
    "recursive": lambda t, cs, co: chunk_recursive(t, chunk_size=cs, chunk_overlap=co),
    "token_based": lambda t, cs, co: chunk_token_based(t),
    "sentence": lambda t, cs, co: chunk_by_sentence(t),
    "sliding_window": lambda t, cs, co: chunk_sliding_window(t, chunk_size=cs, chunk_overlap=50),
    "paragraph": lambda t, cs, co: chunk_by_paragraph(t),
    "semantic": lambda t, cs, co: chunk_semantic(t),
}


def run_all_chunking(text: str, chunk_size=100, chunk_overlap=20) -> Dict[str, List[str]]:
    out = {}
    for name, fn in CHUNKING_FUNCS.items():
        try:
            out[name] = fn(text, chunk_size, chunk_overlap)
        except Exception as e:
            out[name] = [f"[error {name}: {e}]"]
    return out


def chunking_stats(chunks: List[str]) -> Dict[str, Any]:
    lengths = [len(c) for c in chunks]
    return {
        "nb_chunks": len(chunks),
        "longueur_moyenne": float(np.mean(lengths)) if lengths else 0.0,
        "ecart_type": float(np.std(lengths)) if lengths else 0.0,
        "min_len": int(min(lengths)) if lengths else 0,
        "max_len": int(max(lengths)) if lengths else 0,
    }


def _is_error_chunk(chunks: List[str]) -> bool:
    return any(c.startswith("[error") for c in chunks)


def chunking_intra_similarity(chunks: List[str]) -> float:
    if _is_error_chunk(chunks):
        return -1.0  # never win as best
    if len(chunks) < 2:
        return 1.0
    try:
        model = get_embed_model()
        embs = model.encode(chunks)
        m = cosine_similarity(embs)
        np.fill_diagonal(m, np.nan)
        return float(np.nanmean(m))
    except Exception:
        return 0.0


def best_chunking_key(metrics: Dict[str, Dict[str, Any]]) -> str:
    # même critère que TP2 : intra - 0.01 * nb_chunks — exclut méthodes en erreur
    valid = {k: v for k, v in metrics.items() if v.get("intra_similarity", -1) >= 0}
    pool = valid if valid else metrics
    return max(pool, key=lambda k: pool[k].get("intra_similarity", 0) - 0.01 * pool[k].get("nb_chunks", 0))


# ---------------------------------------------------------------------------
# Embeddings — 7 méthodes (TP2 partie 2) — lazy / degrade if deps missing
# ---------------------------------------------------------------------------
def embed_tfidf(chunks: List[str]):
    from sklearn.feature_extraction.text import TfidfVectorizer

    v = TfidfVectorizer()
    return v.fit_transform(chunks).toarray()


def embed_word2vec(chunks, tokenized):
    from gensim.models import Word2Vec

    m = Word2Vec(sentences=tokenized, vector_size=50, window=5, min_count=1, workers=1, epochs=50)
    vecs = []
    for toks in tokenized:
        wv = [m.wv[t] for t in toks if t in m.wv]
        vecs.append(np.mean(wv, axis=0) if wv else np.zeros(50))
    return np.array(vecs)


def embed_glove(chunks, tokenized):
    global _glove_model
    import gensim.downloader as api

    if _glove_model is None:
        _glove_model = api.load("glove-wiki-gigaword-50")
    vecs = []
    for toks in tokenized:
        wv = [_glove_model[t] for t in toks if t in _glove_model]
        vecs.append(np.mean(wv, axis=0) if wv else np.zeros(50))
    return np.array(vecs)


def embed_fasttext(chunks, tokenized):
    from gensim.models import FastText

    m = FastText(sentences=tokenized, vector_size=50, window=5, min_count=1, workers=1, epochs=50)
    return np.array([np.mean([m.wv[t] for t in toks], axis=0) if toks else np.zeros(50) for toks in tokenized])


def embed_doc2vec(chunks, tokenized):
    from gensim.models import Doc2Vec
    from gensim.models.doc2vec import TaggedDocument

    docs = [TaggedDocument(toks, [i]) for i, toks in enumerate(tokenized)]
    m = Doc2Vec(docs, vector_size=50, window=5, min_count=1, workers=1, epochs=50)
    return np.array([m.infer_vector(toks) for toks in tokenized])


def embed_bert(chunks):
    from transformers import AutoTokenizer, AutoModel
    import torch

    global _bert_tok, _bert_model
    try:
        _bert_tok  # type: ignore
    except NameError:
        globals()["_bert_tok"] = None
        globals()["_bert_model"] = None
    if globals()["_bert_model"] is None:
        globals()["_bert_tok"] = AutoTokenizer.from_pretrained("bert-base-uncased")
        globals()["_bert_model"] = AutoModel.from_pretrained("bert-base-uncased")
        globals()["_bert_model"].eval()
    tok = globals()["_bert_tok"]
    mdl = globals()["_bert_model"]
    vecs = []
    with torch.no_grad():
        for ch in chunks:
            inp = tok(ch, return_tensors="pt", truncation=True, padding=True)
            out = mdl(**inp)
            mask = inp["attention_mask"].unsqueeze(-1)
            summed = (out.last_hidden_state * mask).sum(1)
            vecs.append((summed / mask.sum(1)).squeeze().numpy())
    return np.array(vecs)


def embed_sentence_bert(chunks):
    return get_embed_model().encode(chunks)


EMBEDDING_FUNCS: Dict[str, Any] = {
    "tfidf": lambda ch, tok: embed_tfidf(ch),
    "word2vec": lambda ch, tok: embed_word2vec(ch, tok),
    "glove": lambda ch, tok: embed_glove(ch, tok),
    "fasttext": lambda ch, tok: embed_fasttext(ch, tok),
    "doc2vec": lambda ch, tok: embed_doc2vec(ch, tok),
    "bert": lambda ch, tok: embed_bert(ch),
    "sentence_bert": lambda ch, tok: embed_sentence_bert(ch),
}


def embedding_metrics(embeddings: np.ndarray) -> Dict[str, float]:
    if embeddings is None or len(embeddings) < 2:
        return {"intra_similarity": 1.0}
    try:
        m = cosine_similarity(embeddings)
        np.fill_diagonal(m, np.nan)
        return {"intra_similarity": float(np.nanmean(m))}
    except Exception:
        return {"intra_similarity": 0.0}


# ---------------------------------------------------------------------------
# Retrieval — 6 méthodes (TP2 partie 3)
# ---------------------------------------------------------------------------
def retrieve_similarite_brute(query, chunks, k=3):
    model = get_embed_model()
    q_emb = model.encode([query])
    c_emb = model.encode(chunks)
    sims = cosine_similarity(q_emb, c_emb)[0]
    idx = np.argsort(sims)[::-1][:k]
    return [(chunks[i], float(sims[i]), int(i)) for i in idx], sims


def retrieve_faiss(query, chunks, k=3):
    if faiss is None:
        raise RuntimeError("faiss-cpu not installed")
    model = get_embed_model()
    embs = model.encode(chunks).astype("float32")
    faiss.normalize_L2(embs)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    q = model.encode([query]).astype("float32")
    faiss.normalize_L2(q)
    scores, indices = index.search(q, min(k, len(chunks)))
    return [(chunks[i], float(scores[0][j]), int(i)) for j, i in enumerate(indices[0])], scores[0]


def retrieve_bm25(query, chunks, k=3):
    if BM25Okapi is None:
        raise RuntimeError("rank_bm25 not installed")
    corpus_tok = [_word_tokenize(c.lower()) for c in chunks]
    bm25 = BM25Okapi(corpus_tok)
    scores = bm25.get_scores(_word_tokenize(query.lower()))
    idx = np.argsort(scores)[::-1][:k]
    return [(chunks[i], float(scores[i]), int(i)) for i in idx], scores


def retrieve_dpr(query, chunks, k=3):
    global _dpr_model
    if _dpr_model is None:
        _dpr_model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
    q = _dpr_model.encode([query])
    c = _dpr_model.encode(chunks)
    sims = cosine_similarity(q, c)[0]
    idx = np.argsort(sims)[::-1][:k]
    return [(chunks[i], float(sims[i]), int(i)) for i in idx], sims


def retrieve_hybride(query, chunks, k=3, alpha=0.5):
    if BM25Okapi is None:
        raise RuntimeError("rank_bm25 not installed")
    corpus_tok = [_word_tokenize(c.lower()) for c in chunks]
    bm25 = BM25Okapi(corpus_tok)
    bm25_scores = np.array(bm25.get_scores(_word_tokenize(query.lower())))
    bm25_norm = bm25_scores / (bm25_scores.max() + 1e-9)
    model = get_embed_model()
    q = model.encode([query])
    dense = cosine_similarity(q, model.encode(chunks))[0]
    dense_norm = dense / (dense.max() + 1e-9)
    combined = alpha * dense_norm + (1 - alpha) * bm25_norm
    idx = np.argsort(combined)[::-1][:k]
    return [(chunks[i], float(combined[i]), int(i)) for i in idx], combined


def retrieve_reranking(query, chunks, k=3, candidate_pool=5):
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder

            _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception as e:
            raise RuntimeError(f"CrossEncoder unavailable: {e}")
    # candidate pool via brute dense
    cands, _ = retrieve_similarite_brute(query, chunks, k=min(candidate_pool, len(chunks)))
    cand_texts = [c for c, _, _ in cands]
    pairs = [[query, c] for c in cand_texts]
    scores = _reranker.predict(pairs)
    order = np.argsort(scores)[::-1][:k]
    # map back to original chunk index
    return [(cand_texts[i], float(scores[i]), int(cands[i][2])) for i in order], scores


RETRIEVAL_FUNCS: Dict[str, Any] = {
    "similarite_brute": retrieve_similarite_brute,
    "faiss": retrieve_faiss,
    "bm25": retrieve_bm25,
    "dpr": retrieve_dpr,
    "hybride": retrieve_hybride,
    "reranking": retrieve_reranking,
}


def retrieval_metrics_results(results, relevant_texts, k=3):
    retrieved = [r[0] for r in results]
    hits = [1 if t in relevant_texts else 0 for t in retrieved]
    precision = sum(hits) / len(hits) if hits else 0
    recall = sum(hits) / len(relevant_texts) if relevant_texts else 0
    # MRR / RR: first hit rank
    rr = 0.0
    for i, h in enumerate(hits, start=1):
        if h:
            rr = 1.0 / i
            break
    # Also compute cosine-based metrics if sbert available
    return {"precision@k": round(precision, 4), "recall@k": round(recall, 4), "mrr": round(rr, 4)}


# ---------------------------------------------------------------------------
# Helpers to pick best chunks (needed by embedding retrieval steps)
# ---------------------------------------------------------------------------
def _effective_text(text: Optional[str]) -> str:
    if text and text.strip():
        return text.strip()
    return DEFAULT_TEXT


def _best_chunks_for_text(text: str, chunk_size=100, chunk_overlap=20):
    chunk_dict = run_all_chunking(text, chunk_size, chunk_overlap)
    # build metric dict to choose best
    m = {}
    for k, ch in chunk_dict.items():
        s = chunking_stats(ch)
        s["intra_similarity"] = chunking_intra_similarity(ch)
        m[k] = s
    best_key = best_chunking_key(m)
    return chunk_dict, m, best_key, chunk_dict[best_key]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "TP3 RAG API — see /docs", "default_query": DEFAULT_QUERY}


@app.get("/health")
def health():
    return {"status": "ok"}


# -- Chunking ---------------------------------------------------------------
@app.post("/chunking")
def post_chunking(req: ChunkingRequest):
    text = _effective_text(req.text)
    chunk_dict = run_all_chunking(text, req.chunk_size, req.chunk_overlap)
    return {
        "text_preview": text[:300],
        "params": {"chunk_size": req.chunk_size, "chunk_overlap": req.chunk_overlap},
        "results": chunk_dict,
        "summary": {k: {"nb_chunks": len(v), "preview": v[0][:120] if v else ""} for k, v in chunk_dict.items()},
    }


@app.get("/chunking/metrics")
@app.post("/chunking/metrics")
def chunking_metrics(req: Optional[ChunkingRequest] = None):
    text = _effective_text(req.text if req else None)
    cs = req.chunk_size if req else 100
    co = req.chunk_overlap if req else 20
    chunk_dict = run_all_chunking(text, cs, co)
    metrics = {}
    for name, ch in chunk_dict.items():
        s = chunking_stats(ch)
        s["intra_similarity"] = round(chunking_intra_similarity(ch), 4)
        s["longueur_moyenne"] = round(s["longueur_moyenne"], 2)
        s["ecart_type"] = round(s["ecart_type"], 2)
        s["exemple"] = ch[0][:120] if ch else ""
        metrics[name] = s
    best = best_chunking_key(metrics)
    return {"metrics": metrics, "best_method": best, "best_chunks": chunk_dict[best]}


@app.get("/chunking/{method}")
def chunking_single(method: str, chunk_size: int = 100, chunk_overlap: int = 20):
    if method not in CHUNKING_FUNCS:
        return {"error": f"Unknown method {method}. Available: {list(CHUNKING_FUNCS)}"}
    text = DEFAULT_TEXT
    chunks = CHUNKING_FUNCS[method](text, chunk_size, chunk_overlap)
    return {"method": method, "chunks": chunks, "stats": chunking_stats(chunks)}


# -- Embedding --------------------------------------------------------------
@app.post("/embedding")
def post_embedding(req: EmbeddingRequest):
    text = _effective_text(req.text)
    if req.chunks:
        best_chunks = req.chunks
        best_method = "custom"
        chunk_metrics = {}
    else:
        _, chunk_metrics, best_method, best_chunks = _best_chunks_for_text(text)
    tokenized = [_word_tokenize(c.lower()) for c in best_chunks]
    # default = light only: glove (66MB) + bert (440MB) freeze UI for minutes, request them explicitly if needed
    DEFAULT_LIGHT = ["tfidf", "word2vec", "fasttext", "doc2vec", "sentence_bert"]
    methods = req.methods or DEFAULT_LIGHT

    results = {}
    vectors_info = {}
    for name in methods:
        if name not in EMBEDDING_FUNCS:
            results[name] = {"error": "unknown method"}
            continue
        try:
            vecs = EMBEDDING_FUNCS[name](best_chunks, tokenized)
            vecs = np.array(vecs)
            intra = embedding_metrics(vecs)["intra_similarity"]
            # dimension
            dim = int(vecs.shape[1]) if vecs.ndim == 2 else int(vecs.shape[0])
            results[name] = {"intra_similarity": round(float(intra), 4), "dim": dim, "nb_vectors": int(vecs.shape[0])}
            vectors_info[name] = {"shape": list(vecs.shape)}
        except Exception as e:
            results[name] = {"error": str(e)[:400]}

    valid = {k: v for k, v in results.items() if "intra_similarity" in v}
    best_emb = max(valid, key=lambda k: valid[k]["intra_similarity"]) if valid else None

    return {
        "best_chunking_method": best_method,
        "best_chunks": best_chunks,
        "chunk_metrics": chunk_metrics,
        "embedding_metrics": results,
        "best_embedding_method": best_emb,
        "vectors_info": vectors_info,
    }


@app.get("/embedding/metrics")
@app.post("/embedding/metrics")
def embedding_metrics_route(req: Optional[EmbeddingRequest] = None):
    # alias for convenience — same as POST /embedding but focused on metrics
    return post_embedding(req or EmbeddingRequest())


# -- Retrieval --------------------------------------------------------------
@app.post("/retrieval")
def post_retrieval(req: RetrievalRequest):
    text = _effective_text(req.text)
    _, _, _, best_chunks = _best_chunks_for_text(text)
    if not best_chunks:
        return {"error": "No chunks produced"}
    # define ground-truth relevant chunk(s) as first chunk (user can adapt)
    relevant_texts = [best_chunks[0]]

    to_run = [req.method] if req.method and req.method in RETRIEVAL_FUNCS else list(RETRIEVAL_FUNCS.keys())
    out = {}
    for name in to_run:
        try:
            fn = RETRIEVAL_FUNCS[name]
            if name == "hybride":
                results, _ = fn(req.query, best_chunks, k=req.k, alpha=req.alpha)
            else:
                results, _ = fn(req.query, best_chunks, k=req.k)
            metrics = retrieval_metrics_results(results, relevant_texts, k=req.k)
            out[name] = {
                "results": [{"chunk": c, "score": round(s, 4), "index": idx} for c, s, idx in results],
                "metrics": metrics,
            }
        except Exception as e:
            out[name] = {"error": str(e)[:500]}

    return {
        "query": req.query,
        "best_chunks": best_chunks,
        "relevant_texts": relevant_texts,
        "retrieval": out,
    }


@app.get("/retrieval/metrics")
def retrieval_metrics_get(k: int = 3, query: str = DEFAULT_QUERY):
    return post_retrieval(RetrievalRequest(query=query, k=k))


# -- End-to-end RAG ---------------------------------------------------------
@app.post("/query")
def post_query(req: QueryRequest):
    text = _effective_text(req.text)
    _, chunk_metrics, best_chunk_method, best_chunks = _best_chunks_for_text(text)
    method = req.retrieval_method if req.retrieval_method in RETRIEVAL_FUNCS else "similarite_brute"
    try:
        fn = RETRIEVAL_FUNCS[method]
        if method == "hybride":
            results, _ = fn(req.query, best_chunks, k=req.k)
        else:
            results, _ = fn(req.query, best_chunks, k=req.k)
        retrieved = [c for c, _, _ in results]
        scores = [float(s) for _, s, _ in results]
    except Exception as e:
        return {"error": str(e), "best_chunks": best_chunks}

    context = "\n".join(retrieved)
    # Stub generation — students should plug Ollama / OpenAI here
    generated = (
        f"[Réponse générée — stub] Question : '{req.query}'. "
        f"Contexte récupéré ({len(retrieved)} chunks via {method}) : {retrieved[0][:160]}..."
        if retrieved
        else "Aucun contexte récupéré."
    )

    return {
        "query": req.query,
        "best_chunking_method": best_chunk_method,
        "chunk_metrics": chunk_metrics,
        "retrieval_method": method,
        "retrieved_chunks": [{"chunk": c, "score": round(s, 4)} for c, s in zip(retrieved, scores)],
        "context": context,
        "generated_response": generated,
        "note": "Remplace 'generated_response' par un appel Ollama (ollama.chat) ou API externe — voir TP2_Complet.ipynb.",
    }


@app.get("/vectorial")
def vectorial(query: str = DEFAULT_QUERY, k: int = 3, text: Optional[str] = None):
    """GET /vectorial — spec TP: lance chunking + embeddings FAISS + retrieval."""
    return post_query(QueryRequest(query=query, text=text, k=k, retrieval_method="faiss"))


# -- Dedicated metrics dashboard (all-in-one) --------------------------------
@app.get("/metrics/all")
def metrics_all():
    text = DEFAULT_TEXT
    query = DEFAULT_QUERY
    chunk_dict = run_all_chunking(text)
    chunk_metrics = {}
    for name, ch in chunk_dict.items():
        s = chunking_stats(ch)
        s["intra_similarity"] = round(chunking_intra_similarity(ch), 4)
        chunk_metrics[name] = s
    best_key = best_chunking_key(chunk_metrics)
    best_chunks = chunk_dict[best_key]

    # embedding quick metrics (only sentence_bert + tfidf to stay fast)
    emb_metrics = {}
    tokenized = [_word_tokenize(c.lower()) for c in best_chunks]
    for m_name in ["sentence_bert", "tfidf"]:
        try:
            v = EMBEDDING_FUNCS[m_name](best_chunks, tokenized)
            emb_metrics[m_name] = embedding_metrics(np.array(v))
        except Exception as e:
            emb_metrics[m_name] = {"error": str(e)[:200]}

    # retrieval quick metrics
    relevant = [best_chunks[0]] if best_chunks else []
    retr_metrics = {}
    for r_name in list(RETRIEVAL_FUNCS.keys()):
        try:
            fn = RETRIEVAL_FUNCS[r_name]
            res, _ = fn(query, best_chunks, k=3)
            retr_metrics[r_name] = retrieval_metrics_results(res, relevant)
        except Exception as e:
            retr_metrics[r_name] = {"error": str(e)[:200]}

    return {
        "chunking": chunk_metrics,
        "best_chunking": best_key,
        "embedding": emb_metrics,
        "retrieval": retr_metrics,
    }


# -- Serve frontend static files (so http://127.0.0.1:8000/ can also serve UI) --
try:
    _frontend_dir = pathlib.Path(__file__).resolve().parent.parent / "frontend"
    if _frontend_dir.exists():
        app.mount("/static", StaticFiles(directory=str(_frontend_dir)), name="static")

        @app.get("/app")
        def serve_frontend():
            return FileResponse(str(_frontend_dir / "index.html"))
except Exception:
    pass
