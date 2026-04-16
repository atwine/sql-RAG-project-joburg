# RAG Improvement Journey
### A Teaching Session on Building Better RAG Systems

---

## Slide 1 — What is RAG?

**Retrieval-Augmented Generation**

Instead of asking an LLM to answer from memory alone, RAG:

1. **Retrieves** relevant documents from a knowledge base
2. **Augments** the prompt with those documents as context
3. **Generates** an answer grounded in the retrieved data

> *"Give the LLM the right pages from the book before asking the question."*

---

## Slide 2 — Our Project

**Goal:** Query a Health & Demographic Surveillance System (HDSS) dataset using natural language

- 50 synthetic demographic records
- Fields: names, date of birth, village, migration history, household relations
- Fully offline — no cloud APIs, no data leaves the machine

**Stack:**
| Component | Tool |
|-----------|------|
| Embedding | `nomic-embed-text` via Ollama |
| Vector DB | ChromaDB |
| LLM | `gemma3:4b` via Ollama |
| Language | Python |

---

## Slide 3 — Where We Started

**The Basic RAG Pipeline**

```
Load 50 records
      ↓
Embed each record → store in ChromaDB
      ↓
User query → embed query → find top 5 similar docs
      ↓
Send docs as context to LLM
      ↓
Streamed answer
```

This is textbook RAG. It worked for some queries — but broke quickly under real conditions.

---

## Slide 4 — Problem 1: The System Wouldn't Start

**Error:**
```
ConnectionError: Failed to connect to Ollama
```

Ollama was running. `curl http://localhost:11434` responded fine.
But the Python client failed every time.

**Root Cause:**
```bash
OLLAMA_HOST=0.0.0.0:11434   # set in the environment
```

`0.0.0.0` is a **server bind address** (listen on all interfaces).
It is not a valid address for a **client** to connect to.

**Fix — before importing ollama:**
```python
_raw_host = os.getenv("OLLAMA_HOST", "")
if _raw_host.startswith("0.0.0.0"):
    os.environ["OLLAMA_HOST"] = _raw_host.replace("0.0.0.0", "127.0.0.1")
```

> **Lesson:** Environment variables and SDK initialisation order matter.
> Always verify what your client is actually connecting to.

---

## Slide 5 — Problem 2: Right Record, Wrong Answer

**Query:**
```
> What is the date of birth for Miriam Kirunda?
```

**Response:**
```
I don't have enough information based on the data provided.
```

Miriam Kirunda **was** in the dataset (IND-0005, DOB: 1973-09-14).

**Diagnosis — testing different retrieval sizes:**
```
n_results=5  → Miriam Kirunda found: False
n_results=10 → Miriam Kirunda found: False
n_results=15 → Miriam Kirunda found: False
```

The correct record was never retrieved at all.

---

## Slide 6 — Why Semantic Search Missed It

**How vector/semantic search works:**

- Each document is converted to a vector (a list of numbers)
- At query time, the query is also converted to a vector
- The system returns documents whose vectors are *geometrically closest*

**The problem:**

The embedding of `"miriam kirunda date of birth"` was not close enough in vector space to Miriam's record.

Semantic search is optimised for **conceptual similarity** — not **exact name matching**.

> **Lesson:** Semantic search is powerful for meaning and concepts.
> It is weak for proper nouns, IDs, and exact string lookups.

---

## Slide 7 — Failed Fix: Retrieve Everything

**First attempt:** Set `n_results=50` — retrieve all 50 records.

✅ Miriam's record was now in the context.

❌ The LLM still said: *"I don't have enough information."*

**Why?**

Small models (4B parameters) suffer from the **"lost in the middle" problem** — when context is very long, they lose track of details buried in the middle of it.

> **Lesson:** More context ≠ better answers.
> Quality and focus of context matters more than quantity.

---

## Slide 8 — Real Fix: Hybrid Search

**Two retrieval lanes working together:**

```
Query
  ├── Semantic search  → top 10 by vector similarity
  └── Keyword search   → docs containing words from the query
            ↓
     Merge & deduplicate
            ↓
     ~15–20 candidates
```

**Keyword search implementation:**
```python
def keyword_search(query, all_docs, all_ids, max_results):
    words = [w.lower() for w in query.split() if len(w) >= 4]
    hits = []
    for doc_id, doc in zip(all_ids, all_docs):
        if any(w in doc.lower() for w in words):
            hits.append((doc_id, doc))
    return hits
```

| Query type | Best retrieval method |
|---|---|
| *"who migrated for health reasons?"* | Semantic (conceptual) |
| *"date of birth for Miriam Kirunda"* | Keyword (exact name) |
| Most real queries | **Both** |

> **Lesson:** Hybrid search is the industry standard for production RAG.

---

## Slide 9 — Problem 3: Document Format

**Query:**
```
> What is the other name for Patrick Nansubuga?
```

**Response:**
```
The records do not contain any other name for Patrick Nansubuga.
```

But the record had `other_name1 = "Faith"`.

**The document was formatted as:**
```
Name: Patrick Faith Nansubuga
```

The LLM had no way to know "Faith" was an *other name* — it looked like part of the full name string.

---

## Slide 10 — Fix: Explicit Field Labels

**Before:**
```python
f"Name: {record.get('name')} {record.get('other_name1')} {record.get('surname')}\n"
```

**After:**
```python
f"First Name: {record.get('name')}\n"
f"Other Name: {other_name}\n"
f"Surname: {record.get('surname')}\n"
```

**Result:**
```
> What is the other name for Patrick Nansubuga?
→ The other name for Patrick Nansubuga is Faith. (IND-0022)
```

> **Lesson:** Your document format is part of your prompt.
> The LLM reads your documents literally — label every field explicitly.
> Merged fields lose their semantic distinction.

> ⚠️ Whenever you change document format, delete the vector store and re-embed.
> Stale embeddings from the old format will give wrong results.

---

## Slide 11 — Problem 4: Which Candidates to Send?

Hybrid search gives us ~20 candidates. Which 5 do we actually send to the LLM?

**Option A — Take the top 5 semantic results**
Fast, but may miss keyword-matched records.

**Option B — Take the first 5 from the merged list**
Arbitrary — not based on actual relevance to the query.

**Option C — Rerank all candidates, keep the best 5**
Slower, but picks the most relevant docs for *this exact query*.

---

## Slide 12 — Fix: Cross-Encoder Reranking

**Two types of relevance models:**

| | Bi-encoder (embeddings) | Cross-encoder (reranker) |
|---|---|---|
| How it works | Query and doc encoded **separately** | Query and doc read **together** |
| Speed | Fast | Slower |
| Accuracy | Good for recall | Better for precision |
| Use case | Retrieve candidates | Re-score candidates |

**Implementation:**
```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

pairs  = [(query, doc) for doc in candidates]
scores = reranker.predict(pairs)
top5   = [doc for _, doc in sorted(zip(scores, candidates), reverse=True)][:5]
```

> **Lesson:** Use bi-encoders to cast a wide net. Use a cross-encoder to pick the best catch.

---

## Slide 13 — The Final Pipeline

```
User Query
      ↓
┌─────────────────────────────┐
│   Stage 1: Hybrid Retrieval │  ← Cast a wide net
│  Semantic search  → 10 docs │
│  Keyword search   → 10 docs │
│  Merge & deduplicate        │
└─────────────────────────────┘
      ↓ ~20 candidates
┌─────────────────────────────┐
│   Stage 2: Reranking        │  ← Pick the best
│  CrossEncoder scores each   │
│  (query, doc) pair          │
│  Keep top 5                 │
└─────────────────────────────┘
      ↓ 5 focused docs
┌─────────────────────────────┐
│   Stage 3: Generation       │  ← Answer the question
│  gemma3:4b reads context    │
│  Streams answer to user     │
└─────────────────────────────┘
```

---

## Slide 14 — Summary of Improvements

| # | Change | Problem solved |
|---|--------|---------------|
| 1 | Fixed `OLLAMA_HOST` env var | System couldn't connect to Ollama |
| 2 | Diagnosed `n_results` failure | Right records not being retrieved |
| 3 | Added keyword search (hybrid) | Semantic search misses exact name matches |
| 4 | Explicit document field labels | LLM couldn't distinguish merged fields |
| 5 | Cross-encoder reranking | LLM gets the best 5 docs, not just any 5 |

---

## Slide 15 — Key Takeaways

1. **Your document format is part of your prompt**
   Label every field explicitly. Merged fields lose their meaning.

2. **Semantic search ≠ keyword search — use both**
   Hybrid retrieval covers concepts AND exact matches.

3. **More context is not always better**
   Small models get lost in long contexts. Focus beats volume.

4. **Re-embed when you change document format**
   Stale embeddings from the old format will silently give wrong results.

5. **Two-stage retrieval is the production standard**
   Recall (fast, wide) → Rerank (slow, precise) → Generate.

6. **Debug retrieval separately from generation**
   Always check *what docs were retrieved* before blaming the LLM.

---

## Slide 16 — What's Next

- [ ] **Narrative embeddings** — embed full prose descriptions instead of structured fields for richer semantic retrieval
- [ ] **Larger datasets** — chunking strategies for thousands of records
- [ ] **Web UI** — Gradio or Streamlit front-end for non-technical users
- [ ] **Evaluation framework** — measure retrieval precision and answer accuracy systematically

---

*Built with Python · ChromaDB · Ollama · sentence-transformers*
*https://github.com/atwine/sql-RAG-project-joburg*
