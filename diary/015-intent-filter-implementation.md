# 015 — Final Implementation & Intent Filter

**Date**: 2026-03-23  **Tool**: multi_replace_file_content  **Model**: Antigravity
**Iterations**: 15

## Prompt
**2026-03-23 14:42**
[User approved conditional timetable search plan V5]
[I implemented the changes and verified them]

## Changes
- Added `detect_timetable_intent` to `backend/app.py`.
- Updated `hybrid_search` to conditionally query timetable collections.
- Verified that "Informatik" skips timetables while "Wann ist Informatik" includes them.
