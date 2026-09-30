"""Structure-aware chunking.

Legal documents are never split blindly by length:

1. An ebook gets an *overview* chunk (title, type, year, number, preamble, keywords and a
   table of contents) so "which act covers X" questions have a target.
2. Each section is emitted as ONE chunk when it fits `chunk_max_tokens`.
3. Otherwise it is split at structural boundaries: consecutive subsections/schedules are
   greedily grouped up to `chunk_target_tokens`; an oversized child recurses.
4. Only an oversized *leaf* text (or unstructured bodies like circulars, manuals, blogs)
   is split by paragraph → sentence → word, with `chunk_overlap_tokens` of overlap.

Every chunk carries a context header with its parents (act title/type/year, section number
and heading, subsection, schedule) so the retrieved text is self-describing for the LLM and
for citations. The header is embedded together with the content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.core.text import ascii_to_bn_digits, normalize_section_number
from app.ingestion.hasher import text_hash
from app.ingestion.normalizer import NormalizedDocument, Unit
from app.ingestion.tokens import TokenCounter

_SENTENCE_SPLIT = re.compile(r"(?<=[।?!؟.;])\s+")
_TOC_MAX_ENTRIES = 400
_LEAD_IN = "প্রারম্ভিক অংশ:"


@dataclass(frozen=True)
class ChunkingConfig:
    target_tokens: int = 500
    max_tokens: int = 700
    overlap_tokens: int = 80
    min_tokens: int = 30

    @property
    def signature(self) -> str:
        return f"t{self.target_tokens}-m{self.max_tokens}-o{self.overlap_tokens}-n{self.min_tokens}"


@dataclass
class Chunk:
    source_type: str
    source_id: str
    parent_source_id: str | None
    context: str
    content: str
    token_count: int
    section_number: str | None = None
    section_heading: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_index: int = 0

    @property
    def embed_text(self) -> str:
        return f"{self.context}\n\n{self.content}" if self.context else self.content

    @property
    def content_hash(self) -> str:
        return text_hash(self.context, self.content)


def schedule_label(unit: Unit) -> str:
    """Bhumipedia stores lettered clauses ((ক), (a), (i) ...) as nested "schedule" objects.
    Call them দফা (clause) unless the record itself says it is a তফসিল (schedule)."""
    text = f"{unit.number or ''} {unit.heading or ''}"
    return "তফসিল" if re.search(r"তফসিল|schedule", text, re.IGNORECASE) else "দফা"


def _num(s: str | None) -> str:
    """Display form of a number: strip trailing danda/brackets noise but keep Bengali digits."""
    return (s or "").strip().rstrip("।.:").strip()


class Chunker:
    def __init__(self, count_tokens: TokenCounter, config: ChunkingConfig):
        self.tok = count_tokens
        self.cfg = config

    # ------------------------------------------------------------------ public
    def chunk(self, doc: NormalizedDocument) -> list[Chunk]:
        if doc.source_type == "ebook":
            chunks = self._chunk_ebook(doc)
        elif doc.source_type in ("qna_type1", "qna_type2"):
            chunks = self._chunk_qna(doc)
        else:
            chunks = self._chunk_plain(doc)
        chunks = [c for c in chunks if c.content.strip()]
        for i, c in enumerate(chunks):
            c.chunk_index = i
        return chunks

    # ------------------------------------------------------------------ text splitting
    def split_text(self, text: str, budget: int | None = None) -> list[str]:
        """Split free text into pieces of ~target tokens with overlap, preferring paragraph,
        then sentence, then word boundaries. Pieces never exceed max_tokens (approximately:
        token counts of concatenations are estimated as sums)."""
        target = min(self.cfg.target_tokens, budget or self.cfg.target_tokens)
        hard_max = min(self.cfg.max_tokens, budget or self.cfg.max_tokens)
        if self.tok(text) <= hard_max:
            return [text]
        atoms: list[tuple[str, int, str]] = []  # (text, tokens, joiner)
        for para in [p for p in text.split("\n") if p.strip()]:
            pt = self.tok(para)
            if pt <= hard_max:
                atoms.append((para, pt, "\n"))
                continue
            for sent in [s for s in _SENTENCE_SPLIT.split(para) if s.strip()]:
                st = self.tok(sent)
                if st <= hard_max:
                    atoms.append((sent, st, " "))
                    continue
                atoms.extend((w, wt, " ") for w, wt in self._word_windows(sent, target))
        return self._pack(atoms, target, hard_max)

    def _word_windows(self, sentence: str, target: int) -> list[tuple[str, int]]:
        words = sentence.split()
        out, cur, cur_t = [], [], 0
        for w in words:
            wt = self.tok(w) or 1
            if cur and cur_t + wt > target:
                out.append((" ".join(cur), cur_t))
                cur, cur_t = [], 0
            cur.append(w)
            cur_t += wt
        if cur:
            out.append((" ".join(cur), cur_t))
        return out

    def _pack(self, atoms: list[tuple[str, int, str]], target: int, hard_max: int) -> list[str]:
        pieces: list[str] = []
        cur: list[tuple[str, int, str]] = []
        cur_t = 0
        for atom in atoms:
            if cur and cur_t + atom[1] > target:
                pieces.append(self._join(cur))
                # Carry trailing atoms as overlap (never the whole previous piece).
                overlap, ot = [], 0
                for a in reversed(cur[1:]):
                    if ot + a[1] > self.cfg.overlap_tokens:
                        break
                    overlap.insert(0, a)
                    ot += a[1]
                if not overlap and self.cfg.overlap_tokens > 0:
                    # Last atom is longer than the overlap budget (typical for Bengali legal
                    # paragraphs): carry its trailing words instead of no overlap at all.
                    tail = self._tail_words(cur[-1][0], self.cfg.overlap_tokens)
                    if tail:
                        overlap, ot = [(tail, self.tok(tail), "\n")], self.tok(tail)
                if ot + atom[1] > hard_max:
                    overlap, ot = [], 0
                cur, cur_t = overlap, ot
            cur.append(atom)
            cur_t += atom[1]
        if cur:
            tail = self._join(cur)
            # Merge a tiny trailing remainder into the previous piece when it fits.
            if pieces and cur_t < self.cfg.min_tokens and self.tok(pieces[-1]) + cur_t <= hard_max:
                pieces[-1] = pieces[-1] + "\n" + tail
            else:
                pieces.append(tail)
        return pieces

    def _tail_words(self, text: str, budget: int) -> str:
        words = text.split()
        out: list[str] = []
        used = 0
        for w in reversed(words):
            wt = self.tok(w) or 1
            if used + wt > budget:
                break
            out.insert(0, w)
            used += wt
        return " ".join(out) if len(out) < len(words) else ""

    @staticmethod
    def _join(atoms: list[tuple[str, int, str]]) -> str:
        out = atoms[0][0]
        for text, _, joiner in atoms[1:]:
            out += joiner + text
        return out

    # ------------------------------------------------------------------ ebooks
    def _doc_context(self, doc: NormalizedDocument) -> list[str]:
        name_label = {"আইন": "আইনের নাম", "বিধিমালা": "বিধিমালার নাম", "অধ্যাদেশ": "অধ্যাদেশের নাম",
                      "নীতিমালা": "নীতিমালার নাম"}.get(doc.doc_type or "", "দলিলের নাম")
        lines = [f"{name_label}: {doc.title}"]
        meta = []
        if doc.doc_type:
            meta.append(f"ধরন: {doc.doc_type}")
        year_raw = doc.metadata.get("act_year_raw")
        if year_raw or doc.year:
            meta.append(f"সাল: {ascii_to_bn_digits(str(year_raw or doc.year))}")
        if doc.act_number:
            meta.append(f"নম্বর: {doc.act_number}")
        if meta:
            lines.append(" | ".join(meta))
        return lines

    def _unit_header(self, unit: Unit, doc: NormalizedDocument) -> str | None:
        label = doc.section_label
        num = _num(unit.number)
        if unit.source_type == "section":
            head = f"{label}: {num}" if num else label
            return f"{head} — {unit.heading}" if unit.heading else head
        if unit.source_type == "subsection":
            return f"উপ-{label}: {num}" + (f" — {unit.heading}" if unit.heading else "") if num else None
        if unit.source_type in ("schedule", "subschedule"):
            label = schedule_label(unit)
            if unit.source_type == "subschedule":
                label = f"উপ-{label}"
            return label + (f" {num}" if num else "") + (f" — {unit.heading}" if unit.heading else "")
        return None

    def _render(self, unit: Unit, doc: NormalizedDocument, top: bool = True) -> str:
        """Render a unit and all descendants as readable text."""
        parts: list[str] = []
        if unit.source_type == "section":
            if unit.text:
                parts.append(unit.text)
        elif unit.source_type == "subsection":
            prefix = _num(unit.number)
            if prefix and not prefix.startswith("("):
                prefix = f"({prefix})"
            parts.append(f"{prefix} {unit.text}".strip() if unit.text else prefix)
        elif unit.source_type in ("schedule", "subschedule"):
            if not top:
                parts.append(self._unit_header(unit, doc) or "")
            if unit.text:
                parts.append(unit.text)
        else:
            parts.append(unit.text)
        for child in unit.children:
            parts.append(self._render(child, doc, top=False))
        return "\n".join(p for p in parts if p)

    def _section_number_of(self, path: list[Unit]) -> tuple[str | None, str | None]:
        for u in path:
            if u.source_type == "section":
                return normalize_section_number(u.number), u.heading
        return None, None

    def _emit(self, doc, path: list[Unit], identity: Unit, ctx: list[str], content: str,
              extra_meta: dict | None = None) -> Chunk:
        sec_no, sec_heading = self._section_number_of(path)
        meta: dict[str, Any] = {"path": [{"type": u.source_type, "id": u.source_id,
                                          "number": _num(u.number) or None} for u in path]}
        sub = next((u for u in path if u.source_type == "subsection"), None)
        if sub is not None and _num(sub.number):
            meta["subsection_number"] = _num(sub.number)
        sch = next((u for u in path if u.source_type == "schedule"), None)
        if sch is not None:
            key = "schedule_number" if schedule_label(sch) == "তফসিল" else "clause_number"
            meta[key] = _num(sch.number) or None
        if identity.metadata.get("note"):
            meta["note"] = identity.metadata["note"]
        if extra_meta:
            meta.update(extra_meta)
        context = "\n".join(ctx)
        return Chunk(
            source_type=identity.source_type, source_id=identity.source_id,
            parent_source_id=identity.parent_source_id, context=context, content=content,
            token_count=self.tok(context) + self.tok(content),
            section_number=sec_no, section_heading=sec_heading, metadata=meta,
        )

    def _pack_unit(self, doc: NormalizedDocument, unit: Unit, path: list[Unit], ctx: list[str]) -> list[Chunk]:
        path = [*path, unit]
        header = self._unit_header(unit, doc)
        my_ctx = [*ctx, header] if header else list(ctx)
        ctx_tokens = self.tok("\n".join(my_ctx))
        budget = max(self.cfg.min_tokens * 2, self.cfg.max_tokens - ctx_tokens)
        full = self._render(unit, doc)
        if self.tok(full) <= budget:
            return [self._emit(doc, path, unit, my_ctx, full)]

        chunks: list[Chunk] = []
        own = self._render(Unit(unit.source_type, unit.source_id, unit.parent_source_id, unit.text,
                                unit.number, unit.heading), doc) if unit.text else ""
        own_t = self.tok(own)
        # A short lead-in ("এই আইনে—") gives meaning to every child chunk: repeat it in context.
        child_ctx = list(my_ctx)
        if own and own_t <= 120:
            # Keep only the nearest lead-in; deeper nesting would otherwise stack them up.
            child_ctx = [c for c in my_ctx if not c.startswith(_LEAD_IN)] + [f"{_LEAD_IN} {own}"]

        if own and own_t > budget:
            for i, piece in enumerate(self.split_text(own, budget)):
                chunks.append(self._emit(doc, path, unit, my_ctx, piece, {"part": i + 1}))
            own = ""

        group: list[tuple[Unit, str]] = []
        group_t = own_t if own else 0
        group_has_own = bool(own)

        def flush() -> None:
            nonlocal group, group_t, group_has_own
            if not group and not group_has_own:
                return
            texts = ([own] if group_has_own else []) + [t for _, t in group]
            if group_has_own:
                identity, ids = unit, [unit.source_id] + [u.source_id for u, _ in group]
                c = self._emit(doc, path, identity, my_ctx, "\n".join(texts))
            else:
                identity = group[0][0]
                ids = [u.source_id for u, _ in group]
                c = self._emit(doc, [*path, identity], identity, child_ctx, "\n".join(texts))
            if len(ids) > 1:
                c.metadata["covers"] = ids
            chunks.append(c)
            group, group_t, group_has_own = [], 0, False

        child_budget = max(self.cfg.min_tokens * 2, self.cfg.max_tokens - self.tok("\n".join(child_ctx)))
        for child in unit.children:
            rendered = self._render(child, doc, top=False)
            ct = self.tok(rendered)
            if ct > child_budget:
                flush()
                chunks.extend(self._pack_unit(doc, child, path, child_ctx))
                continue
            if (group or group_has_own) and group_t + ct > min(self.cfg.target_tokens, child_budget):
                flush()
            group.append((child, rendered))
            group_t += ct
        flush()
        return chunks

    def _chunk_ebook(self, doc: NormalizedDocument) -> list[Chunk]:
        base_ctx = self._doc_context(doc)
        chunks = self._overview_chunks(doc, base_ctx)
        for unit in doc.units:
            if unit.source_type == "ebook_text":
                chunks.extend(self._chunk_long_text(doc, unit, base_ctx))
            else:
                chunks.extend(self._pack_unit(doc, unit, [], base_ctx))
        return chunks

    def _overview_chunks(self, doc: NormalizedDocument, ctx: list[str]) -> list[Chunk]:
        """Overview = identifying metadata + the COMPLETE table of contents (continued in
        extra "toc" chunks if long, never truncated), then the preamble as its own chunk.
        The TOC must survive: it is what lets "all sections of <act>" questions list every
        provision heading even when the provisions' full text does not fit the context."""
        m = doc.metadata
        meta: list[str] = []
        if m.get("publication_by"):
            meta.append(f"প্রকাশ: {m['publication_by']}" +
                        (f", তারিখ: {m['publication_date_raw']}" if m.get("publication_date_raw") else ""))
        for key, label in (("issued_date", "জারির তারিখ"), ("applicable_date", "কার্যকর তারিখ"),
                           ("branch", "শাখা"), ("signature", "স্বাক্ষরকারী"), ("heading", "শিরোনাম"),
                           ("motto", "মূলমন্ত্র")):
            if m.get(key):
                meta.append(f"{label}: {m[key]}")
        if m.get("meta_keywords"):
            meta.append("মূলশব্দ: " + ", ".join(m["meta_keywords"]))
        toc = [f"{_num(u.number)} {u.heading or ''}".strip() for u in doc.units
               if u.source_type == "section" and (u.heading or u.number)][:_TOC_MAX_ENTRIES]
        head = f"সূচি ({doc.section_label}সমূহ, মোট {ascii_to_bn_digits(str(len(toc)))}টি):"
        toc_text = "\n".join([head, *toc]) if toc else ""
        ov_ctx = [*ctx, "অংশ: পরিচিতি ও সূচি"]
        budget = self.cfg.max_tokens - self.tok("\n".join(ov_ctx))
        first = "\n".join([*meta, toc_text]).strip() or doc.title
        pieces = [first] if self.tok(first) <= budget else self.split_text(first, budget)
        chunks = []
        for i, piece in enumerate(pieces):
            role = "overview" if i == 0 else "toc"
            piece_ctx = ov_ctx if i == 0 else [*ctx, "অংশ: সূচি (পূর্ববর্তী অংশের ধারাবাহিকতা)"]
            chunks.append(self._emit(doc, [], Unit("ebook", doc.source_id, None, piece), piece_ctx, piece,
                                     {"role": role}))
        preamble = [f"{label}: {m[key]}" for key, label in (("proposal", "প্রস্তাবনা"), ("objective", "উদ্দেশ্য"))
                    if m.get(key)]
        if preamble:
            pre_ctx = [*ctx, "অংশ: প্রস্তাবনা ও উদ্দেশ্য"]
            pre_budget = self.cfg.max_tokens - self.tok("\n".join(pre_ctx))
            for piece in self.split_text("\n".join(preamble), pre_budget):
                chunks.append(self._emit(doc, [], Unit("ebook", doc.source_id, None, piece), pre_ctx, piece,
                                         {"role": "preamble"}))
        return chunks

    @staticmethod
    def _is_heading(line: str) -> bool:
        s = line.strip()
        return (3 <= len(s) <= 80 and " | " not in s and not s.startswith("- ")
                and not s.endswith(("।", ".", ",", ";", "—", ":-")) and not re.fullmatch(r"[\d০-৯().\s-]+", s))

    def _chunk_long_text(self, doc: NormalizedDocument, unit: Unit, ctx: list[str]) -> list[Chunk]:
        """Unstructured bodies (circulars, manuals, act appendices): paragraph packing with
        overlap, tracking the most recent heading-like line as local context."""
        role = unit.metadata.get("role")
        lines = [ln for ln in unit.text.split("\n") if ln.strip()]
        base = [*ctx, "অংশ: তফসিল/সংযুক্তি" if role == "appendix" else "অংশ: মূল পাঠ"]
        budget = self.cfg.max_tokens - self.tok("\n".join(base)) - 30  # room for a heading line
        pieces = self.split_text("\n".join(lines), budget)
        chunks: list[Chunk] = []
        heading: str | None = None
        cursor = 0
        for i, piece in enumerate(pieces):
            # Find the heading in effect at the start of this piece.
            first_line = piece.split("\n", 1)[0]
            scan, found_heading = cursor, heading
            while scan < len(lines) and lines[scan] != first_line:
                if self._is_heading(lines[scan]):
                    found_heading = lines[scan]
                scan += 1
            if scan < len(lines):  # not found (word-window split): keep previous position/heading
                cursor, heading = scan, found_heading
            piece_ctx = [*base, f"প্রসঙ্গ: {heading}"] if heading and heading != first_line else base
            src_type = "schedule" if role == "appendix" else "ebook"
            ident = Unit(src_type, unit.source_id, doc.source_id if role == "appendix" else None, piece)
            chunks.append(self._emit(doc, [], ident, piece_ctx, piece, {"part": i + 1, "role": role}))
        return chunks

    # ------------------------------------------------------------------ Q&A / plain
    def _chunk_qna(self, doc: NormalizedDocument) -> list[Chunk]:
        unit = doc.units[0]
        ctx = ["সরকারি প্রশ্নোত্তর"]
        if doc.category:
            ctx.append(f"বিষয়শ্রেণি: {doc.category}")
        if unit.metadata.get("keyword"):
            ctx.append(f"মূলশব্দ: {unit.metadata['keyword']}")
        budget = self.cfg.max_tokens - self.tok("\n".join(ctx))
        if self.tok(unit.text) <= budget:  # the normal case: one Q&A = one retrieval unit
            return [self._emit(doc, [], unit, ctx, unit.text)]
        question = unit.metadata.get("question") or doc.title
        ctx_q = [*ctx, f"প্রশ্ন: {question}"]
        answer = unit.text.split("\nউত্তর: ", 1)[-1]
        budget = self.cfg.max_tokens - self.tok("\n".join(ctx_q))
        return [self._emit(doc, [], unit, ctx_q, f"উত্তর (অংশ {i + 1}): {p}", {"part": i + 1})
                for i, p in enumerate(self.split_text(answer, budget))]

    def _chunk_plain(self, doc: NormalizedDocument) -> list[Chunk]:
        if doc.source_type == "blog":
            ctx = [f"ব্লগ: {doc.title}"] + ([f"লেখক: {doc.author}"] if doc.author else [])
            if doc.publication_date:
                ctx.append(f"তারিখ: {doc.publication_date.isoformat()}")
        elif doc.source_type == "topic":
            ctx = [f"ফোরাম আলোচনা ({doc.metadata.get('forum_name', '')}): {doc.title}"]
        else:
            ctx = [f"ফোরাম: {doc.title}"]
        chunks = []
        for unit in doc.units:
            budget = self.cfg.max_tokens - self.tok("\n".join(ctx))
            for i, piece in enumerate(self.split_text(unit.text, budget)):
                chunks.append(self._emit(doc, [], unit, ctx, piece, {"part": i + 1}))
        return chunks
