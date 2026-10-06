"""Bengali-aware text utilities shared by ingestion and retrieval.

PostgreSQL has no Bengali stemmer/dictionary. Verified behaviour: the `simple` parser
tokenises Bengali words correctly (combining vowel signs and conjuncts stay intact), but
does no stemming. We therefore normalise text *in Python* identically on both the index
side (`lexical_text`) and the query side (`query_terms`), and use prefix matching on
lightly-stemmed query terms so e.g. "নামজারির"/"নামজারিতে" match "নামজারি".
"""

from __future__ import annotations

import html as html_lib
import re
import unicodedata

from bs4 import BeautifulSoup, NavigableString, Tag

BN_DIGITS = "০১২৩৪৫৬৭৮৯"
_BN_TO_ASCII = str.maketrans(BN_DIGITS, "0123456789")
_ASCII_TO_BN = str.maketrans("0123456789", BN_DIGITS)
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)

_BLOCK_TAGS = {
    "p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "table",
    "blockquote", "section", "article", "header", "footer", "pre", "hr",
}
_DROP_TAGS = {"script", "style", "noscript", "iframe", "object", "embed", "svg", "img", "head"}


def bn_to_ascii_digits(s: str) -> str:
    return s.translate(_BN_TO_ASCII)


def ascii_to_bn_digits(s: str) -> str:
    return s.translate(_ASCII_TO_BN)


def clean_unicode(s: str) -> str:
    """NFC-normalise and strip zero-width / soft-hyphen characters."""
    return unicodedata.normalize("NFC", s).translate(_ZERO_WIDTH)


def normalize_whitespace(s: str) -> str:
    s = s.replace("\xa0", " ").replace("　", " ")
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r" *\n *", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def html_to_text(value: str | None) -> str:
    """Convert (TinyMCE) HTML to clean text while preserving block structure.

    - Paragraphs/headings/list items become separate lines; list items get a "- " prefix.
    - Table rows become " | "-joined cells, one row per line.
    - Scripts, styles, images and other active content are dropped (safe HTML handling).
    Plain text input passes through (after entity unescaping).
    """
    if not value:
        return ""
    if "<" not in value:
        return normalize_whitespace(clean_unicode(html_lib.unescape(value)))
    soup = BeautifulSoup(value, "lxml")
    for t in soup.find_all(_DROP_TAGS):
        t.decompose()

    out: list[str] = []

    def walk(node) -> None:
        for child in node.children:
            if isinstance(child, NavigableString):
                if child.__class__.__name__ in ("Comment", "Doctype", "ProcessingInstruction"):
                    continue
                out.append(str(child))
            elif isinstance(child, Tag):
                name = child.name
                if name == "tr":
                    cells = [normalize_whitespace(c.get_text(" ")) for c in child.find_all(["td", "th"])]
                    cells = [c for c in cells if c]
                    if cells:
                        out.append("\n" + " | ".join(cells) + "\n")
                    continue
                if name == "br":
                    out.append("\n")
                    continue
                if name == "li":
                    out.append("\n- ")
                elif name in _BLOCK_TAGS:
                    out.append("\n")
                walk(child)
                if name in _BLOCK_TAGS:
                    out.append("\n")

    walk(soup)
    text = "".join(out)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return normalize_whitespace(clean_unicode(text))


def parse_year(value: str | int | None) -> int | None:
    """Parse a year in Bengali or ASCII digits ('২০২৩', '2023'); None if not a plausible year."""
    if value is None:
        return None
    m = re.search(r"\d{4}", bn_to_ascii_digits(str(value)))
    if not m:
        return None
    year = int(m.group())
    return year if 1700 <= year <= 2200 else None


_SECTION_JUNK = "()[]।.:-৷ \u2010\u2011\u2013"


def normalize_section_number(value: str | None) -> str | None:
    """'১। ' → '1', '(২) ' → '2', '৫ক' → '5ক', '৩৷' → '3', '9a' → '9A', '26[86' → '26'.
    '0' / '' → None.

    The upstream API writes the separator as danda, full stop, space or U+09F7 (৷); a Latin
    suffix is case-folded to upper case so a query's "9a" meets the stored "9A"."""
    if not value:
        return None
    v = bn_to_ascii_digits(clean_unicode(value)).strip().strip(_SECTION_JUNK)
    v = re.sub(r"\s+", "", v)
    # Footnote residue such as "26[86": keep the leading number.
    v = re.sub(r"^(\d+[A-Za-zক-হ]{0,2})[\[\(].*$", r"\1", v)
    v = "".join(ch.upper() if "a" <= ch <= "z" else ch for ch in v)
    if not v or v == "0":
        return None
    return v


def lexical_coverage(terms: list[str], lexical_body: str) -> float:
    """Fraction of query terms present in a normalised text (prefix terms ``x:*`` match as prefixes)."""
    if not terms:
        return 0.0
    tokens = set(lexical_body.split())
    hit = 0
    for t in terms:
        if t.endswith(":*"):
            p = t[:-2]
            hit += any(tok.startswith(p) for tok in tokens)
        else:
            hit += t in tokens
    return hit / len(terms)


# ---------------------------------------------------------------------------------------
# Lexical normalisation
# ---------------------------------------------------------------------------------------

# Bengali question words, particles and very common function words. Kept deliberately
# small: legal terms (ধারা, আইন, কর, ফি ...) must never be stopwords.
BN_STOPWORDS = frozenset(
    clean_unicode(w) for w in """
    কি কী কেন কিভাবে কীভাবে কোথায় কখন কোন কোনো কোনটি কে কারা কাকে কার কত কতটি কয়টি কতগুলো
    কতদিন কতদিনে কতজন কিনা কি-না কিংবা অথবা বা ও এবং আর যে যা যদি তবে তাহলে তাই সেই এই ওই
    এটা এটি ওটা সেটা সেটি ইহা উহা তাহা তা এর এতে এখানে সেখানে আমি আমার আমাদের আপনি আপনার
    আপনারা তুমি তোমার সে তার তারা তাদের তিনি তাঁর হয় হবে হলে হলো হল হয়েছে হইবে হইয়াছে হইতে
    করা করে করতে করবো করব করবেন করেন করলে করিয়া করিবে করিতে পারি পারব পারবো পারবেন পারে যায়
    যাবে যাবেনা দিতে দেয়া দেওয়া নিতে নেয়া লাগে লাগবে প্রয়োজন জন্য জন্যে থেকে হতে দিয়ে নিয়ে
    সাথে সঙ্গে মধ্যে উপর নিচে পর আগে না নয় নাই নেই হ্যাঁ জি একটি একটা এক সব সকল সমস্ত
    বিষয়ে সম্পর্কে ব্যাপারে দয়া করে অনুগ্রহ প্লিজ চাই চাচ্ছি জানতে জানাবেন বলুন বলবেন আছে আছেন থাকে থাকবে জিনিস আসলে মানে বলতে বোঝায় বুঝায় বোঝানো কাকে বলে
    """.split()
)
EN_STOPWORDS = frozenset(
    "a an the is are was were be to of in on for and or what how when where which who why do does "
    "can i my me you your it this that with from by about please tell".split()
)

# Inflectional suffixes, longest first. Applied only to query terms.
_BN_SUFFIXES = (
    "গুলোতে", "গুলোর", "গুলির", "গুলো", "গুলি", "দেরকে", "দের", "কেই", "টির", "টার",
    "তেই", "েরা", "ের", "এর", "য়ের", "কে", "তে", "টি", "টা", "রা", "ে", "র",
)
_BN_SUFFIXES = tuple(clean_unicode(s) for s in _BN_SUFFIXES)
_VOWEL_END = re.compile(r"[\u09be-\u09cc]$")
_TOKEN_RE = re.compile(r"[\wঀ-৿]+", re.UNICODE)


# Common spelling variants in Bangladeshi land-service text, canonicalised on BOTH the
# index and query side (e.g. a question with "নম্বর" must match an answer with "নাম্বার").
# Only orthographic variants of the same word belong here — not semantic synonyms.
_VARIANTS = {
    "নাম্বার": "নম্বর", "নাম্বর": "নম্বর", "নং": "নম্বর", "নম্বরে": "নম্বর",
    "জরীপ": "জরিপ", "রেজিষ্ট্রেশন": "রেজিস্ট্রেশন", "রেজিষ্ট্রি": "রেজিস্ট্রি",
    "সার্টিফিকেট": "সনদ", "ওয়ারিশান": "ওয়ারিশ", "উত্তরাধিকারী": "ওয়ারিশ",
    "কী": "কি",
}


_VARIANTS = {clean_unicode(k): clean_unicode(v) for k, v in _VARIANTS.items()}


def tokenize(text: str) -> list[str]:
    """Normalised tokens: NFC, zero-width removed, lower-cased, Bengali digits → ASCII,
    common spelling variants canonicalised."""
    text = bn_to_ascii_digits(clean_unicode(text)).lower()
    # Treat danda and other punctuation as separators; keep Bengali combining marks.
    return [_VARIANTS.get(t, t) for t in _TOKEN_RE.findall(text) if t and t != "_"]


def lexical_text(text: str) -> str:
    """Index-side normalisation: space-joined normalised tokens (no stopword removal, so
    phrase-y titles still match; `simple` config handles the rest)."""
    return " ".join(tokenize(text))


def light_stem(token: str) -> str:
    """Strip one common Bengali inflectional suffix if a reasonable stem remains."""
    if not re.search(r"[ঀ-৿]", token):
        return token
    for suf in _BN_SUFFIXES:
        if token.endswith(suf) and len(token) - len(suf) >= 3:
            stem = token[: -len(suf)]
            # Genitive "র" only attaches after a vowel (নামজারি+র); "ডিসিআর" keeps its র.
            if suf == "র" and not _VOWEL_END.search(stem):
                continue
            return stem
    return token


def query_terms(query: str) -> list[str]:
    """Query-side terms for tsquery construction (deduplicated, order preserved).

    Each term is either an exact token or a stem to be prefix-matched (suffix ":*").
    Only characters from the tokenizer alphabet survive, so terms are always safe to
    splice into a tsquery string (which is itself passed as a bind parameter).
    """
    seen: set[str] = set()
    terms: list[str] = []
    for tok in tokenize(query):
        if tok in BN_STOPWORDS or tok in EN_STOPWORDS:
            continue
        stem = light_stem(tok)
        # Prefix-match only reasonably long stems; short ones would match too broadly, so
        # for those emit the exact inflected and bare forms instead (জমির → জমির | জমি).
        if len(stem) >= 4 and not stem.isdigit():
            candidates = [f"{stem}:*"]
        else:
            candidates = [tok] if stem == tok else [tok, stem]
        for term in candidates:
            if term not in seen:
                seen.add(term)
                terms.append(term)
    return terms


def normalize_query(query: str) -> str:
    """Canonical form of a user query for cache keys and logging."""
    return " ".join(tokenize(query))
