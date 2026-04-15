# 020 — Parallel Search & Limit UI

**Date**: 2026-03-23  **Tool**: multi_replace_file_content  **Model**: Antigravity
**Iterations**: 20

## Prompt
**2026-03-23 14:49**
[User approved performance and limits plan V8]
[I implemented the changes and verified them]

## Changes
- Parallelized Qdrant queries using `ThreadPoolExecutor`.
- Reduced search latency by ~85% (1s -> 150ms).
- Added explicit 429 error handling and UI warning in `SummaryBox.tsx`.
