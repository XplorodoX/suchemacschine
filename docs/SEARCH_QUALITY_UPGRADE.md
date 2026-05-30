# Search Quality Upgrade

This change set fixes the main reasons search results were poor and upgrades the
embedding model. It touches both ranking (no re-index needed) and the embedding
model (re-index **required**).

## What changed

### 1. Score-fusion bug (biggest impact, no re-index)
`boost_and_rank()` blended the raw RRF retrieval score (scale ~0.01–0.1) with the
lexical score (scale 0–1). The semantic signal was effectively discarded and
ranking collapsed to keyword overlap. The RRF score is now min-max normalised to
[0, 1] and fused as `0.70 * vector + 0.30 * lexical` (previously an inverted
`0.35 / 0.65`).

### 2. Boosts no longer dominate
Intent and NavBoost used large additive constants (+1.2 … +2.0) on top of scores
that were ~0–1, so a single boost decided the ranking. They are now bounded
multipliers (e.g. `*1.5`), and NavBoost is a small additive nudge capped at +0.2.

### 3. Cross-Encoder reranking
The top `RERANK_POOL` (default 30) candidates are re-scored jointly with
`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` and blended 50/50 with the pipeline
score. Configurable via `CROSS_ENCODER_MODEL` / `RERANK_POOL`. Search still works
if the model fails to load (the step becomes a no-op).

### 4. Embedding model → `intfloat/multilingual-e5-base` (re-index required)
Replaces `paraphrase-multilingual-MiniLM-L12-v2` (384-dim) with e5-base (768-dim),
which is markedly better for German retrieval. e5 needs task prefixes:
`query: ` for queries and `passage: ` for documents — handled centrally.

The dense model now lives in one place: `scrapers/hybrid_utils.py`
(`encode_passage` / `encode_passages` / `encode_query` / `dense_vector_size`),
driven by the `EMBEDDING_MODEL` env var. Every prepare/index script and the API
read from there, so switching models is a single change.

## Re-indexing (required for #4)

Because the vector dimension changes (384 → 768), all collections must be rebuilt
with the new model. Queries and documents must use the **same** model.

The easiest path is the helper script, which builds the images, rebuilds every
collection and restarts the backend, then verifies the dimensions:

```bash
./reindex.sh            # full run
./reindex.sh --no-build # skip image rebuild if already current
```

To do it by hand instead:

```bash
export EMBEDDING_MODEL=intfloat/multilingual-e5-base   # already set in docker-compose

# Re-run the prepare + index steps for every source, e.g.:
python scrapers/prepare_hs_aalen_extended_data.py && python scrapers/index_hs_aalen_to_qdrant.py
python scrapers/prepare_asta_data.py               && python scrapers/index_asta_to_qdrant.py
python scrapers/prepare_starplan_semesters_data.py && python scrapers/index_starplan_semesters_to_qdrant.py
# hs_aalen_search: prepare_data.py FIRST — it rewrites processed_data.jsonl with
# fresh 768-dim embeddings. index_to_qdrant.py only reads those embeddings, so
# skipping prepare would try to upsert stale 384-dim vectors and fail.
python backend/prepare_data.py && python backend/index_to_qdrant.py
# hs_aalen_pdfs (init_pdf_index.py computes its own embeddings):
python backend/init_pdf_index.py
```

(With Docker, run the `scrapers` profile, then restart the backend.)

## Rolling back the model only

Set `EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2` everywhere and
re-index. The ranking fixes (#1–#3) are independent and need no re-index.
