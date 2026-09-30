# Test fixtures

`snapshot/*.json` are a **small, unmodified subset of real records** fetched from the live
Bhumipedia APIs on 2026-09-29 (same shape as the API responses). They were selected to cover:

| file | records | why |
|---|---|---|
| ebook.json | 164, 241, 838, 291, 633 | sections → subsections → schedules → subschedules (164); Bengali act with schedules (241); gazetted circular with an **empty title** upstream (838); short circular without sections (291); approved placeholder record `test-6-1` (633) |
| qna_type2.json | 24776 + 5 | categorised Q&A incl. the DCR-fee example |
| qna_type1.json | first 4 | uncategorised Q&A |
| blog.json | 2 shortest | HTML blog bodies |
| forum.json | all | the live forum groups (currently test content) |

They are test inputs only and are never loaded into a production database.
Refresh with `python -m app.cli snapshot <dir>` and re-select if the upstream schema changes.
