#!/usr/bin/python3 -I
"""Generate tests/data/tokenizer-cases.txt for test-tokenizer.

Each line holds a text in hexadecimal, a tab, and the token ids the
Hugging Face tokenizers library gives it.  The texts are synthetic:
words in many scripts, accents, punctuation, white space and control
characters, long words, special tokens, and samples of every Unicode
block.  Code points that the library's older Unicode tables classify
differently from the current ones are left out, since tokenizer.c
deliberately follows the current tables (Python's unicodedata must
match utf8proc's Unicode version).

Usage: gen-tokenizer-cases.py TOKENIZER.JSON > tokenizer-cases.txt"""

import random, sys, unicodedata
from tokenizers import Tokenizer

random.seed(20261108)
tok = Tokenizer.from_file(sys.argv[1])
abcd = tok.encode("abcd").ids
ab, cd = tok.encode("ab").ids[1], tok.encode("cd").ids[1]

# The library's (and tokenizer.c's) CJK ranges, which are split off.
CJK = [(0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2A6DF),
       (0x2A700, 0x2B73F), (0x2B740, 0x2B81F), (0x2B920, 0x2CEAF),
       (0xF900, 0xFAFF), (0x2F800, 0x2FA1F)]

def mark_or_control(c):
    return unicodedata.category(c) in ("Mn", "Cc", "Cf", "Co", "Cs")

def agrees(c):
    """Whether the library treats C in "ab" C "cd" as current Unicode
    says: deleted (marks and controls), split off (punctuation), or
    part of the word."""
    ids = tok.encode("ab" + c + "cd").ids
    deleted = ids == abcd
    split = len(ids) == 5 and ids[1] == ab and ids[3] == cd
    if c.isspace() or ord(c) < 0x80:
        return True
    if any(lo <= ord(c) <= hi for lo, hi in CJK):
        return split
    if all(mark_or_control(d) for d in unicodedata.normalize("NFD", c)):
        return deleted
    if unicodedata.category(c).startswith("P"):
        return split
    return not deleted and not split

words = ["Hello", "WORLD", "naïve", "Café", "İstanbul", "ΣΊΣΥΦΟΣ", "straße",
         "ﬁle", "Ⅻ", "x²", "don't", "e-mail", "U.S.A.", "##", "[CLS]",
         "[SEP]", "[MASK]", "[UNK]x", "[PAD", "unaffable", "électricité",
         "日本語テキスト", "한국어", "Ελληνικά", "русский", "عربي", "עברית",
         "हिन्दी", "😀👍🏽", "\t\n\r\x0b\x0c", "  　​",
         "‍﻿", "\x85", "a" * 99, "b" * 100, "c" * 101, "Ａｂｃ",
         "１２３", "ǅ", "ﬀ", "Å", "Å", "é", "é", "℃", "…", "—", "«quoted»",
         "$100", "50%", "C++", "foo_bar", "<html>", "a/b\\c", "�",
         "\U000e0001", "\U0010ffff", "\x7f", "͸", "\U0002b820",
         "search_document:", "File name:",
         "~/Documents/tax-return_2025.PDF", "JPEG image data, 4032x3024"]
texts = ["", " ", "a", "hello world", "search_query: penguins"]
for _ in range(400):
    texts.append(" ".join(random.choice(words)
                          for _ in range(random.randint(1, 12))))
for _ in range(100):
    texts.append("".join(random.choice(words)
                         for _ in range(random.randint(1, 5))))
for start in range(0x80, 0x30000, 0x400):
    texts.append("".join(chr(c) for c in range(start, start + 48)
                         if not 0xD800 <= c <= 0xDFFF and agrees(chr(c))))
for t in texts:
    ids = tok.encode(t).ids
    print(t.encode("utf-8").hex() + "\t" + " ".join(map(str, ids)))
