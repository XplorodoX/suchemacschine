# 023 — Timetable Visibility & Deduplication Fix

**Date**: 2026-03-23  **Tool**: multi_replace_file_content  **Model**: Antigravity
**Iterations**: 23

## Prompt
**2026-03-23 14:58**
Vorlsungspläne werden jetztz gar ned mehr angezeigt in dersuche, die fehlenn  jetztv komplett irgendwie.... 

## Changes
- Updated `hybrid_search` to always fetch timetable results for tab compatibility.
- Fixed deduplication logic: results without URLs (Starplan) are now deduplicated via text snippet.
- Verified that "Informatik" now correctly returns 12 timetable entries.
 village
