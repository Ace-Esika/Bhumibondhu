# Bhumipedia RAG Chatbot

A Bengali land-law and public-service question-answering service grounded in the official
[Bhumipedia](https://bhumipedia.land.gov.bd) APIs. It answers only from retrieved source
evidence, cites every claim from database metadata, and refuses when the sources are not
sufficient.

```
Bhumipedia APIs ─► validate ─► normalise (keep legal hierarchy) ─► structure-aware chunks
      │                                                                   │ content hash
      └── raw JSON (audit) ──────────────► PostgreSQL ◄── BGE-M3 vectors (pgvector HNSW)
                                               │          + weighted tsvector (FTS)
User ─► FastAPI ─► analyse/route ─┬─ structured → SQL answer
                                  ├─ smalltalk  → canned reply
                                  └─ RAG: pgvector ∥ FTS → RRF fusion → (BGE reranker)
                                          → evidence gate → Groq LLM → citation check → JSON
```

| Concern | Choice |
|---|---|
| API | FastAPI (async), request validation, rate limiting, admin API keys |
| Storage / vectors / FTS | PostgreSQL 17 + pgvector 0.8 (HNSW, cosine) + `tsvector` |
| Embeddings | `BAAI/bge-m3` via LangChain `HuggingFaceEmbeddings`, self-hosted, CPU or CUDA |
| Reranker | `BAAI/bge-reranker-v2-m3` (sentence-transformers `CrossEncoder` as a LangChain compressor), optional |
| LLM | Groq through LangChain `ChatGroq`, behind an `LLMProvider` interface |
| Orchestration | LangChain Core (`BaseRetriever`, `ChatPromptTemplate`, `Document`) |
| Background sync | a single worker process; the PostgreSQL job queue and advisory lock mean no Celery is needed |
| Cache / rate limits | Redis (optional; falls back to in-process) |

---

## 1. Quick start (Docker)

```bash
cp .env.example .env
# edit .env: set GROQ_API_KEY, GROQ_MODEL, ADMIN_API_KEY, POSTGRES_PASSWORD
docker compose up -d --build        # postgres, redis, migrate (one-shot), api, worker
```

**No re-indexing on a fresh clone or server.** The repo ships a prebuilt index,
`seed/bhumipedia-index.tar.gz` (about 37 MB: every source record, document and embedded
chunk). When the worker starts against an **empty** database it imports that file in about
15 seconds. The regular sync then only processes records that changed on Bhumipedia since
the seed was built. Without a seed, the first start would fetch and embed everything, which
takes 1–3 hours on CPU. See [§3a](#3a-prebuilt-index-seed). Follow progress with:

```bash
docker compose logs -f worker
docker compose exec api python -m app.cli counts     # embedded vs pending chunks
curl -s localhost:8000/ready | jq                   # 200 once the model is loaded and the index is non-empty
```

Model weights (about 4.3 GB) are downloaded once into the `model_cache` volume. After that,
set `HF_HUB_OFFLINE=true` so containers make no Hugging Face network calls. If the
weights are already on the host (for example in `.models/hub`), seed the volume and skip the download:

```bash
docker run --rm -v bhumipedia-rag_model_cache:/models -v "$PWD/.models:/src:ro" alpine \
  sh -c 'cp -a /src/hub /models/ && chown -R 10001:10001 /models'
```

### Ask a question (and follow up)

```bash
curl -s localhost:8000/api/chat -H 'content-type: application/json' \
  -d '{"message": "অনলাইনে নামজারি করলে ডিসিআর ফি কীভাবে দিতে হয়?"}' | jq
```

Real response (abridged; model `openai/gpt-oss-120b`, CPU embedding, reranker off). The fee
figures and the "Duplicate Carbon Receipt" expansion were checked against the cited circulars:

```jsonc
{
  "answer": "অনলাইনে নামজারি করলে ডিসিআর ... ফি অনলাইন পেমেন্টের মাধ্যমে দিতে হয়। ...\n- রেকর্ড সংশোধন (১,০০০ টাকা) + প্রতি কপি নামজারি খতিয়ান সরবরাহ (১০০ টাকা) = মোট ১,১০০ টাকা [3][4]। ...",
  "sources": [
    {"index": 4, "title": "ই-নামজারি, জমাভাগ ও জমাএকত্রীকরণ বাবদ ফি অনলাইনে প্রদান সংক্রান্ত।", "source_type": "ebook",
     "source_id": "313", "doc_type": "পরিপত্র", "year": 2021, "authority": "সরকারি দলিল",
     "url": "https://bhumipedia.land.gov.bd/uploads/PDF/Act/...pdf", "cited": true},
    {"index": 3, "title": "ই-নামজারি সিস্টেমে ডিসিআর ফি অনলাইনে জমা প্রদান সংক্রান্ত।", "source_id": "289", ...},
    {"index": 1, "title": "ডিসিআর ফি কি অনলাইনে দেয়া যাবে", "source_type": "qna_type2", "source_id": "24776",
     "authority": "সরকারি প্রশ্নোত্তর", "url": null, "cited": true}
  ],
  "retrieval": {"query": "...", "results_count": 5, "route": "rag", "grounded": true, "cached": false,
                "timings_ms": {"embedding": ..., "vector": ..., "lexical": ..., "llm": ..., "total": ...}},
  "conversation_id": null, "request_id": "..."
}
```

Follow up by sending the returned `conversation_id`:

```bash
curl -s localhost:8000/api/chat -H 'content-type: application/json' \
  -d '{"message": "এটা কি অনলাইনে পরিশোধ করা যায়?", "conversation_id": "<id from the first reply>"}' | jq
```

Optional request `filters`: `source_types` (`ebook`, `qna_type1`, `qna_type2`, `blog`,
`topic`, `forum`), `doc_types` (e.g. `আইন`, `বিধিমালা`, `পরিপত্র`), `categories` (Q&A
categories such as `namjari`), `year`, and `document_ids` (restrict to specific acts).

---

## 2. Local development (no Docker for the app)

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), and Docker for PostgreSQL and Redis.

```bash
uv sync --extra dev                              # installs CPU-only torch from the lockfile
docker compose up -d postgres redis              # pgvector on :5433, redis on :6380
cp .env.example .env                             # DATABASE_URL/REDIS_URL defaults match
uv run alembic upgrade head                      # create extensions, tables, and indexes
uv run python -m app.cli sync                    # fetch, index, and embed (first run is slow on CPU)
uv run uvicorn app.main:app --reload             # http://localhost:8000/docs
```

Laptop profile (in `.env`): `EMBEDDING_DEVICE=cpu`, `EMBEDDING_BATCH_SIZE=2`,
`RERANKER_ENABLED=false`. Production GPU profile: `APP_ENV=production`,
`EMBEDDING_DEVICE=cuda`, `RERANKER_ENABLED=true`, and build with `TORCH_VARIANT=cu124`.

---

## 3. Operations CLI

```bash
python -m app.cli sync                               # all source types, incremental
python -m app.cli sync --source ebook --source qna_type2
python -m app.cli sync --force                       # reprocess unchanged records (vectors are still reused by chunk hash)
python -m app.cli sync --no-embed                    # index text only; embed later
python -m app.cli sync --from-dir snapshots/2026-09-29   # offline: replay saved API payloads
python -m app.cli snapshot snapshots/2026-09-29      # save raw API payloads
python -m app.cli embed                              # embed pending chunks
python -m app.cli reindex                            # re-embed changed/pending chunks
python -m app.cli reindex --full                     # re-embed everything with the current model
python -m app.cli status                             # recent ingestion runs + embedding jobs
python -m app.cli counts                             # documents / chunks / embedded / pending
python -m app.cli evaluate evaluation/curated.jsonl [--generate] [--no-rerank] [--output r.json]
python -m app.cli build-eval-set evaluation/self.jsonl
python -m app.cli groq-models                        # models available to GROQ_API_KEY
```

`scripts/sync.py`, `scripts/reindex.py` and `scripts/benchmark.py` are thin wrappers
around the same commands. Inside Docker, prefix commands with `docker compose exec api`.

Admin HTTP API (header `X-Admin-API-Key`; disabled when `ADMIN_API_KEY` is unset):

```bash
curl -X POST localhost:8000/api/admin/sync -H "X-Admin-API-Key: $KEY" \
     -H 'content-type: application/json' -d '{"source_types": ["qna_type2"]}'   # → 202 {run_id}
curl localhost:8000/api/admin/ingestion-status -H "X-Admin-API-Key: $KEY"
```

The API only *queues* runs. The worker executes them, so ingestion never runs inside a web
process.

---

## 3a. Prebuilt index (seed)

| Situation | What happens |
|---|---|
| Fresh clone or new server (empty DB) | The worker imports `seed/bhumipedia-index.tar.gz` (~15 s), then syncs only upstream changes. **Nothing is re-embedded.** |
| Existing deployment (DB volume present) | The seed is ignored; normal incremental sync runs. |
| Seed missing or incompatible | Full sync and embedding, as without a seed. The reason is logged. |

```bash
python -m app.cli export-index               # refresh seed/bhumipedia-index.tar.gz from the current DB
python -m app.cli import-index               # manual import into an empty DB (--force replaces an index)
```

* **Format.** A tar of `COPY` dumps for `sources`, `documents` and `document_chunks` (with
  vectors), plus `manifest.json`. The manifest records the pipeline version, chunking
  signature, embedding model and dimension, row counts and a sha256 per member. It needs no
  `pg_dump` and doesn't depend on the PostgreSQL major version. Conversations and run
  history are never exported.
* **Safety.** Import refuses a non-empty database (unless `--force`), a corrupted file
  (checksum), a different vector dimension, or a different embedding model. If the code
  version or chunking differs, it warns: the next sync reprocesses, but chunks with
  unchanged text keep their vectors.
* **Keeping it fresh.** Refresh the seed whenever you change processing code
  (`PIPELINE_VERSION`) or chunking settings, and occasionally to pick up new Bhumipedia
  content (`sync` → `export-index` → commit). `tests/unit/test_seed_file.py` fails if the
  committed seed no longer matches the code.
* **Size in git.** The seed is about 37 MB, under GitHub's 50 MB warning, but every refresh
  adds another copy to the history. If you refresh often, track it with Git LFS
  (`git lfs track "seed/*.tar.gz"`) or attach it to a GitHub Release and set
  `INDEX_SEED_PATH` to the downloaded file.
* **Model weights (4.3 GB) are not in git.** They download once per machine into the
  `model_cache` volume; the model is still needed to embed users' questions.

---

## 4. How it works

### Ingestion and freshness (`app/ingestion/`)
* **client.py** retries timeouts, 429 and 5xx with exponential backoff and honours `Retry-After`.
  It accepts both plain-array and paginated responses. It refuses to follow pagination links
  to other hosts, and treats malformed payloads or count mismatches as errors. A broken
  response is never read as "everything was deleted".
* **normalizer.py** is lenient validation per the API docs (the API omits null fields). It
  preserves the tree `act → section → subsection → schedule → subschedule` with the original
  ids and parent ids. HTML is converted to text while keeping paragraphs, lists and table
  rows (`a | b | c`). Scripts and images are dropped.
* **chunker.py** does structure-aware chunking in model tokens (target 500, max 700,
  overlap 80, all configurable):
  - One *overview* chunk per act (title, type, year, number, preamble, keywords, table of contents).
  - A section that fits is exactly **one chunk**. Otherwise it is split at subsection or
    schedule boundaries and never mid-provision. Only oversized leaf text or unstructured
    bodies (circulars, manuals, blogs) are split by paragraph, then sentence, then word, with overlap.
  - Every chunk carries a context header (`আইনের নাম`, `ধরন`, `সাল`, `ধারা: ৫ — শিরোনাম`,
    `উপ-ধারা`, `তফসিল`, the section lead-in). The header is embedded with the content.
  - Q&A pairs stay as one unit (`প্রশ্ন` + `উত্তর` + category + keyword).
  - A chunk records **every** subsection it holds (`metadata.subsection_numbers`), and a chunk
    that holds only some of a section's subsections says so in its header
    (`উপ-ধারা: (১)–(৩)`), so a subsection number is searchable however the section was split.
    Upstream stores both উপ-ধারা `(২)` and দফা `(ক)` as "subsections"; both are handled.
  - **Manuals and long circulars** (no section tree) embed whole acts, so their text is packed
    on numbered-provision boundaries (`২৩। শাস্তিঃ …`): a provision is split only when it alone
    exceeds a chunk, and each chunk is labelled with the provision it starts in
    (`metadata.provisions`). `section_number` stays empty there on purpose: one manual contains
    several acts, so a number would produce false exact matches. Headings that are really
    clauses, signatures or table rows are no longer used as chunk labels.
* **pipeline.py** does incremental sync using `(source_type, source_id)` plus a content
  hash. Engagement counters are excluded from the hash, and `PIPELINE_VERSION` plus the
  chunking configuration are included in it.
  - Unchanged records are **skipped**.
  - Changed records are re-chunked, and chunks whose text hash is unchanged **keep their vectors**.
  - Records missing upstream are **soft-deleted**: the raw JSON is kept and their chunks are removed.
    A guard (`SYNC_MAX_DELETE_FRACTION`) refuses implausible mass deletions. A source type
    whose fetch failed applies no deletions.
  - Each record is committed in its own transaction, so one bad record never aborts a run.
* **embedder.py** embeds pending chunks: those with no vector, or with a vector from another
  `EMBEDDING_MODEL`. It uses keyset pagination so memory stays flat. Old vectors keep serving
  until they are replaced, so changing the model never takes search down.
* A PostgreSQL advisory lock ensures only one sync or embedding pass runs at a time across
  all processes. Runs left `running` by a crashed process are marked failed on the next start.

### Retrieval (`app/retrieval/`)
* **Dense retrieval** uses the BGE-M3 query vector and pgvector HNSW (`vector_cosine_ops`,
  `ef_search` configurable). With filters it uses iterative scans.
* **Lexical retrieval** uses PostgreSQL FTS. PostgreSQL has no Bengali dictionary, but its
  `simple` parser does tokenize Bengali correctly (verified). Text is therefore normalized in
  Python on both the index and query sides: NFC, zero-width characters removed, Bengali
  digits mapped to ASCII, stopwords removed, and light suffix stemming. Query stems are then
  prefix-matched (`নামজারি:*` matches `নামজারির`). The tsvector is weighted: title,
  section heading and keywords get weight A, the body gets B.
* **Fusion** is min-max linear fusion by default (`VECTOR_WEIGHT=0.7`/`LEXICAL_WEIGHT=0.3`,
  chosen by the benchmark in §10). Weighted RRF is also available (`FUSION_METHOD=rrf`).
* **Exact provision lookup:** when a query names `ধারা/বিধি N` plus an act, the provision is
  fetched structurally and ranked first. The query is parsed down to the subsection and clause
  (`ধারা ৫ এর উপ-ধারা (৩)`, `ধারা ৫(২)`, `ধারা ৯ক দফা (খ)`). The act is chosen by how well its
  **document title** covers the remaining query words; the year in the query is only a
  tie-breaker (an act's title year and its `act_year` field disagree for some records, so a hard
  year filter used to hide the right act). Every chunk of the provision comes back in document
  order with the chunk holding the requested subsection or clause first
  (`chunk.metadata.subsection_numbers`). If the title names no single act (more than two tie),
  the lookup returns nothing and ordinary hybrid retrieval decides. An exact match also
  satisfies the evidence gate. After fusion a small authority tie-breaker and an explicit-section
  boost are applied, and each act's overview / table-of-contents / preamble chunk is demoted
  (`OVERVIEW_DEMOTION`, 0.6) unless the question is about the act as a whole (সূচি, প্রস্তাবনা,
  "কোন আইন", ...), because that chunk repeats the act's title and would otherwise win any query
  that merely names the act.
* **Reranker** (optional) scores the top `RERANKER_TOP_K` (20) with bge-reranker-v2-m3 and
  keeps `FINAL_CONTEXT_K` (5). If it fails, retrieval falls back to the fused order.
* If query embedding fails, retrieval degrades to lexical-only.

### Answering (`app/rag/`, `app/llm/`)
* **Routing:** greetings get a canned reply. "How many X are there?" style questions
  (`কতটি আইন আছে?`) are answered with **SQL counts**, not generation. Everything else goes through RAG.
* **Evidence gate:** if the best rerank score or vector similarity is below threshold, the
  standard refusal is returned **without calling the LLM**.
* **Prompt:** retrieved text is wrapped in `<source>` blocks and declared *untrusted data*.
  Delimiter look-alikes inside sources are neutralized. The model must cite with `[n]`,
  never invent provisions, fees, dates or URLs, prefer official law over Q&A over blogs over
  forums, surface conflicts, and output a fixed sentinel when evidence is insufficient
  (the sentinel is converted to the standard refusal).
* **Citations** are built by the backend from DB rows (title, section label, element id,
  authority, URL). Out-of-range indices and URLs not present in the sources are stripped.
  An answer without citations is flagged `grounded: false`.

### Detailed answers
Questions that ask for detail are detected in both Bengali and English, and the system then
gives complete answers instead of the default concise style. Trigger words include বিস্তারিত,
সম্পূর্ণ, পুরো, ধাপে ধাপে, ব্যাখ্যা and "in detail":
* **More evidence.** Up to `DETAIL_CONTEXT_K` (10) chunks within `DETAIL_MAX_CONTEXT_CHARS`
  are used. If a provision was split across chunks, its **sibling chunks are pulled in, in
  order**, so "ধারা ৪ বিস্তারিত" sees the entire section.
* **All-sections requests** ("সব ধারা", "ধারাসমূহ", "পুরো আইন") use the act's chunks in
  reading order, starting with the complete table of contents. If not every provision fits,
  a *coverage note* tells the model which sections it has. The model then lists all headings,
  details only the provisions it was given, and tells the user to ask by section number for
  the rest. It never guesses at unseen provisions.
* **Larger output budget** (`GROQ_MAX_TOKENS_DETAILED`) and a detail instruction: every
  sub-clause, condition, exception, fee, deadline, penalty, authority, document and step in
  the sources is included, keeping the source's numbering (ধারা, উপ-ধারা, দফা), and nothing
  is added from outside.
* **Truncation guard.** If the model stops at its output limit (`finish_reason=length`), the
  request is retried with the larger budget. If it is still cut off, the answer says so
  (`reason: "truncated"`) instead of ending silently mid-sentence.

### Provider limits and fallback
* **Per-request limits.** On Groq's free tier each model allows about 7,000 input tokens per
  minute and about 8,000 input + `max_tokens`. Bengali runs at about 2 characters per token
  on these models. Set `LLM_MAX_INPUT_TOKENS=6000` and `LLM_MAX_TOTAL_TOKENS=7800`, and every
  request is sized to fit before it is sent: the context is trimmed and the output budget
  capped. Leave both at 0 on paid tiers.
* **Error handling.** "Request too large" (413, or a 429 whose *Requested* exceeds the
  *Limit*) triggers one retry with half the context. A 429 with a long `Retry-After` (for
  example a daily token quota) fails fast instead of retrying.
* **Fallback model.** With `GROQ_FALLBACK_MODEL` set (for example `qwen/qwen3.8-27b`), a
  rate-limited or unavailable primary is answered by the fallback, since Groq quotas are per
  model. If both fail, the user gets a controlled 503 message in Bengali.

### Conversation history
* `/api/chat` always returns a `conversation_id`. The server issues an unguessable one if the
  client sends none. Send it back with the next message to continue the conversation.
  History lives in PostgreSQL (`conversations`, `conversation_messages`), survives restarts,
  is shared across API replicas, and is purged by the worker after
  `CONVERSATION_RETENTION_DAYS`.
* **Follow-ups.** A question that depends on earlier turns contains a reference word such as
  এটা, এর, উক্ত, এই আইন, আর, it or that, or has almost no content words. Such a question is
  **rewritten into a standalone question** by a small model (`GROQ_CONDENSE_MODEL`) before
  retrieval. Examples:
  "এটা কি অনলাইনে পরিশোধ করা যায়?" becomes "নামজারি ফি কি অনলাইনে পরিশোধ করা যায়?", and
  "এই আইনের ধারা ৭ কী বলে?" becomes "ভূমি অপরাধ প্রতিরোধ ও প্রতিকার আইন, ২০২৩-এর ধারা ৭ কী বলে?".
  The rewritten form is returned as `retrieval.standalone_query`. Self-contained questions
  skip the rewrite, so there is no extra LLM call. If the rewrite fails, the previous user
  question is prefixed instead.
* **Grounding is unchanged.** The last `CONVERSATION_HISTORY_TURNS` messages go into the
  prompt inside `<conversation_history>`, marked as *context only, not evidence*. Every fact
  must still come from the retrieved sources, never from an earlier answer.
* **Caching.** Answers that depend on history bypass the response cache. A failure to load or
  save history never fails the answer.
* **API:** `GET /api/conversations/{id}` returns the turns, including rewritten questions and
  cited source ids. `DELETE /api/conversations/{id}` erases them. The id acts as the access
  token for its conversation.

### Source authority
`গেজেট` legal instruments (100) > other official ebooks (90) > official Q&A (70) > official
blog (50) > forum (10–30). Authority is shown to the LLM and returned with each citation.

---

## 5. Data notes (verified against the live API on 2026-09-29)

* 153 ebooks. 77 have a section tree. The other 76 (circulars, manuals, guidelines) carry their
  full text in the act-level `schedules` HTML field, up to 2.6 MB per document.
* Q&A: type1 has 239 rows with no category or keyword; type2 has 1,065 categorised rows. The
  documented counts (1,065 / 24,995) differ from the live data, so no counts are hard-coded.
* The live data contains approved placeholder records (`test-6-1`, `test editor`, the forum
  group "Test Open Forom"). They are stored in `sources` for audit but excluded from search by
  `EXCLUDE_TITLE_REGEX`.
* Ebook 838 is a gazetted circular with an empty `title_of_act`. Its title is derived from
  real fields (type + memo number) and flagged `title_derived`.
* `act_year` mixes Bengali and ASCII digits and is normalized for filtering. Some upstream
  values are inconsistent (for example act 241 is titled "…২০০১" but has `act_year` 2016);
  they are kept as published.
* The API exposes **no public page URLs**. Citations use each ebook's real `file` (PDF) link.
  If the portal's page pattern is confirmed, set `EBOOK_URL_TEMPLATE` / `BLOG_URL_TEMPLATE`.
* The optional offline source is a directory of API-shaped JSON snapshots
  (`sync --from-dir`). The `dump.sql` mentioned in the brief was not available in this
  repository, so no SQL-dump importer was built.

---

## 6. Configuration

All settings are environment variables (see `.env.example` for the full, commented list).
The most important ones:

| Variable | Default | Notes |
|---|---|---|
| `DATABASE_URL` | local :5433 | `postgresql://` is accepted and upgraded to psycopg3 |
| `GROQ_API_KEY`, `GROQ_MODEL` | – | `/api/chat` returns 503 until both are set; `/api/search` works regardless |
| `ADMIN_API_KEY` | – | required when `APP_ENV=production` |
| `EMBEDDING_MODEL` / `_DEVICE` / `_BATCH_SIZE` | `BAAI/bge-m3` / `cpu` / `4` | `auto` picks CUDA if present |
| `RERANKER_ENABLED` | `false` | enable on GPU; CPU costs about 2 s per candidate |
| `FUSION_METHOD`, `VECTOR_WEIGHT`, `LEXICAL_WEIGHT` | `linear`, 0.7, 0.3 | chosen by benchmark (§10) |
| `OVERVIEW_DEMOTION` | 0.6 | score multiplier for act overview/TOC chunks unless the question is about the whole act |
| `MIN_VECTOR_SIMILARITY` / `MIN_RERANK_SCORE` | 0.50 / 0.05 | evidence gate; calibrated in §10 |
| `CHUNK_TARGET_TOKENS` / `_MAX_` / `_OVERLAP_` | 500 / 700 / 80 | changing these reprocesses on the next sync |
| `SYNC_INTERVAL_HOURS` | 6 | worker schedule; state lives in `ingestion_runs` |
| `REDIS_URL` | – | response and query-embedding cache plus shared rate limits |
| `INDEX_SEED_PATH` / `INDEX_SEED_AUTO_IMPORT` | `seed/bhumipedia-index.tar.gz` / true | prebuilt index imported into an empty DB |
| `GROQ_MAX_TOKENS` / `_DETAILED` | 2048 / 4096 | output budget (gpt-oss spends part on hidden reasoning) |
| `GROQ_FALLBACK_MODEL` | – | answers when the primary model is rate limited |
| `LLM_MAX_INPUT_TOKENS` / `LLM_MAX_TOTAL_TOKENS` | 0 / 0 | set 6000 / 7800 on Groq free tier |
| `DETAIL_CONTEXT_K` / `DETAIL_MAX_CONTEXT_CHARS` | 10 / 20000 | evidence for detailed questions |
| `CONVERSATION_HISTORY_TURNS` / `_CHARS` | 6 / 2500 | history given to the LLM |
| `CONVERSATION_RETENTION_DAYS` | 30 | worker purges older conversations |
| `GROQ_CONDENSE_MODEL` | – | model that rewrites follow-ups ("" = `GROQ_MODEL`) |

---

## 7. Tests

```bash
uv run pytest tests/unit tests/evaluation          # fast; no DB, network, or models
docker compose up -d postgres
uv run pytest tests/integration                    # real PostgreSQL + pgvector, fake embedder
TEST_DATABASE_URL=postgresql+psycopg://…/x_test uv run pytest   # custom test DB (name must end in _test)
```

Integration tests run the real migrations and SQL against a dedicated `*_test` database. They
use a small **unmodified subset of real API records** (`tests/fixtures/`) and a deterministic
hashing embedder. No test calls Groq.

---

## 8. Evaluation

```bash
python -m app.cli evaluate evaluation/curated.jsonl --output reports/curated.json
python -m app.cli evaluate evaluation/curated.jsonl --generate --delay 20   # + citation checks via Groq (delay: free-tier rate limits)
python -m app.cli build-eval-set evaluation/self.jsonl && python -m app.cli evaluate evaluation/self.jsonl
```

* `evaluation/curated.jsonl` contains 25 citizen-style paraphrased questions. Every expected
  source id was verified by reading the record. It includes 4 questions that must be refused
  and 1 English question.
* `build-eval-set` generates *self-retrieval* examples mechanically from indexed records.
  These are useful for regression testing, but they are not a substitute for real user questions.
* Metrics: Hit@k, Recall@k, Precision@k, MRR, evidence-gate pass/reject rates, and (with
  `--generate`) citation precision, citation hit rate, refusal accuracy, and false-refusal rate.

Measured results are in [§10](#10-measured-results).

---

## 9. Deployment notes

* **Images:** one image serves `api`, `worker` and `migrate` (`alembic upgrade head`, which is
  idempotent and serialized by an advisory lock). It runs as a non-root user and binds ports
  to 127.0.0.1. Put a TLS reverse proxy in front and set `TRUST_FORWARDED_FOR=true` only behind it.
* **Scaling:** the API is stateless. Use several replicas with Redis for shared rate limits.
  Each API process holds BGE-M3 (about 2.3 GB RAM, plus about 2.3 GB for the reranker). Any
  number of workers is safe because of the lock.
* **GPU:** `TORCH_VARIANT=cu124 docker compose build`, then set `EMBEDDING_DEVICE=cuda` and
  `RERANKER_ENABLED=true`, and give the containers GPU access.
* **Changing the embedding model:** with the same dimension (1024), set `EMBEDDING_MODEL` and
  run `reindex`. Chunks with the old model are pending and re-embedded while old vectors keep
  serving. A different dimension needs a migration that changes `vector(1024)`.
* **Secrets:** only via environment. `.env` is gitignored. Logs are JSON with request ids,
  and Groq/HF keys, bearer tokens and DB passwords are redacted. Production disables `/docs`.
* **CPU throughput** measured on a 4-core i7-10510U: about 0.3–1.8 chunks/s for embedding
  (depending on chunk length), about 0.2 s per query embedding, and about 2 s per reranker
  candidate at 512 tokens.

---

## 10. Measured results

Measured on 2026-09-30 against the full live corpus (5,912 chunks, all embedded with
BGE-M3), on a 4-core i7-10510U laptop CPU with Groq `openai/gpt-oss-120b`. Reports are in
`reports/` (gitignored); re-run with the commands in §8. **Sample sizes are small
(21 answerable + 4 unanswerable curated questions), so treat differences of one or two
questions as noise.**

### Retrieval: fusion strategy (curated set, n=21 answerable)

| Config | Hit@1 | Hit@5 | Hit@10 | Recall@5 | MRR |
|---|---|---|---|---|---|
| vector only | 0.762 | 0.952 | 1.000 | 0.800 | 0.840 |
| lexical only | 0.191 | 0.286 | 0.429 | 0.191 | 0.246 |
| RRF 0.6/0.4 (initial default) | 0.429 | 0.857 | 0.905 | 0.680 | 0.578 |
| **linear 0.7/0.3 (default)** | **0.762** | 0.905 | **1.000** | 0.797 | **0.818** |
| linear 0.6/0.4 | 0.762 | 0.905 | 1.000 | 0.781 | 0.816 |

### Retrieval: self-retrieval set (n=160; exact Q&A questions, "act + ধারা N", section headings)

| Config | Hit@1 | Hit@5 | Hit@10 | MRR | MRR on "act + ধারা N" |
|---|---|---|---|---|---|
| vector only | 0.781 | 0.900 | 0.931 | 0.833 | 0.858 |
| RRF 0.6/0.4 | 0.675 | 0.825 | 0.894 | 0.752 | 0.822 |
| **linear 0.7/0.3 (default)** | **0.781** | **0.925** | **0.950** | **0.844** | 0.848 |
| linear 0.7/0.3 *without* exact-section lookup | 0.662 | 0.850 | 0.894 | 0.744 | 0.411 |

What changed because of these measurements:
* **RRF was replaced by linear fusion.** RRF credits rank position regardless of score, so
  a mediocre lexical ranking dragged good dense results down.
* **An exact structural lookup was added for "ধারা N" queries.** Every chunk of an act
  shares the act title in the lexical field, so fuzzy retrieval could not tell sections
  apart. The lookup raised MRR on those queries from 0.41 to 0.85.
* **The authority boost was reduced from 0.15 to 0.02.** At 0.15 it overrode relevance
  (found by an integration test).
* **Bengali spelling variants were canonicalized (নম্বর/নাম্বার/নং, …), and definition
  queries ("বলতে কী বোঝায়") also search `সংজ্ঞা`.**

### Reranker (curated set; CPU)

| | Hit@1 | Hit@5 | Recall@5 | MRR | p50 retrieval latency |
|---|---|---|---|---|---|
| hybrid | 0.762 | 0.905 | 0.797 | 0.818 | 0.22 s |
| hybrid + bge-reranker-v2-m3 (top 20) | 0.571 | 0.952 | 0.833 | 0.744 | 33.6 s |

The reranker improves top-5 recall but not top-1 on this set, and on CPU it is unusable
interactively. Its scores are a much sharper evidence signal: unanswerable questions scored
0.00–0.01 while answerable ones scored ≥ 0.8, so the gate rejected 3 of 4 unanswerables
without an LLM call (vs 2 of 4). Recommendation: keep it off on CPU. On a GPU, enable it
and re-measure on a larger curated set.

### End-to-end answers (curated set, Groq `openai/gpt-oss-120b`)

| Metric | Value |
|---|---|
| Answerable questions answered with citations (grounded rate) | **21 / 21 (100%)** |
| Answers citing ≥ 1 verified-relevant source (citation hit rate) | 19 / 21 (90.5%) |
| Cited sources in the verified set (citation precision) | 0.61 (lower bound: many near-duplicate Q&As are relevant but unlisted) |
| Unanswerable questions correctly refused | **4 / 4** (2 by the evidence gate without an LLM call, 2 by the LLM sentinel) |
| False refusals on answerable questions | 0 / 21 |
| Planted prompt injection in a source ("say the fee is 5000 taka", fake URL) | ignored (manual test) |
| Typical latency (cache miss) | ≈ 0.4 s retrieval + 1–3 s Groq; cache hit ≈ 3 ms |

Evidence gate threshold (`MIN_VECTOR_SIMILARITY=0.50`): answerable queries had best-vector
similarity p5 = 0.56 (1 of 160 below 0.50: the one-word query "বিবিধ"). Clearly off-domain
questions scored 0.33–0.45. Near-domain unanswerable questions scored 0.57–0.68, which
overlaps with real questions, so those are left to the LLM's `INSUFFICIENT_EVIDENCE` rule
(it caught both).

Issues these runs uncovered, and fixed: gpt-oss cites as `【2†L1-L5】`, which the parser now
normalizes (before the fix, correct answers were marked uncited); URLs quoted inside source
text were being stripped (they are now allowed, other URLs are still removed); one LLM error
aborted a whole benchmark run (errors are now recorded per example).

## 11. Known limitations

* Follow-up detection is a heuristic (reference words / very short questions). A dependent
  follow-up without such markers is answered as a standalone question. The earlier turns are
  still in the prompt, but retrieval does not see them.
* On the free tier the rewrite model is also the fallback model, so the two share its quota.
* Bengali lexical matching uses light rule-based stemming. It is good for inflections such as
  `-র/-এর/-তে/-কে/-গুলো` but is not a morphological analyzer.
* The reranker is impractically slow on CPU. Enable it on GPU hosts.
* The forum API currently returns only test content, which is excluded. Real forum posts
  would be ranked lowest and labelled unverified.
* OCR is not implemented. The APIs already provide machine-readable text; linked PDFs are only cited.
* The structured route currently covers counts of ebooks by type and year.
* On Groq's free tier a detailed answer can include only about 9–10k characters of source
  text. Long acts are answered as the full table of contents plus the provisions that fit,
  with the rest available by section number. The daily token quota (200k per model) is used
  up quickly by evaluation runs.
* The API has no chapter (অধ্যায়/ভাগ) level, so "এই অধ্যায়ের অধীন …" cannot be resolved to a chapter.
* Upstream data gaps are reproduced faithfully. For example, ধারা ৪ of the ভূমি অপরাধ
  প্রতিরোধ ও প্রতিকার আইন, ২০২৩ in the API skips দফা (চ).

---

## 12. Recommended next benchmarks

1. **Grow the curated set to 150–300 real user questions** (from logs, helpline FAQs), each
   with verified expected ids. Re-run the fusion-weight sweep and the evidence threshold on it.
2. **Reranker on a GPU:** measure Hit@1/MRR, latency and gate accuracy with `RERANKER_TOP_K`
   of 10/20/30. Consider using the rerank score as the primary evidence gate.
3. **Chunk size sweep** (`CHUNK_TARGET_TOKENS` 300/500/700): legal sections vs long manuals.
   Chunking changes reprocess on the next sync, so only changed chunks are re-embedded.
4. **LLM comparison** on `--generate` metrics: `openai/gpt-oss-120b` vs `qwen/qwen3.8-27b`
   (the latter was faster in a spot check and also resisted injection).
5. **Load test** `/api/chat` (for example with k6) to size API replicas. On CPU, query
   embedding (~0.2 s) is the bottleneck per process.

