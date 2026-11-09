/* tokenizer.c -- the BERT WordPiece tokenizer of nomic-embed-text.
   Copyright (C) 2026 Martin Rixham

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This follows the Hugging Face "tokenizers" library with the settings
   in the model's tokenizer.json: a BertNormalizer (clean text, pad
   Chinese characters, strip accents, lowercase), a BertPreTokenizer
   (split on white space and punctuation) and a WordPiece model with
   "##" continuations and at most 100 characters per word.

   Characters are classified with utf8proc's current Unicode tables.
   The tokenizers library uses older ones, so about 660 recent marks,
   punctuation and format characters, which it sees as unassigned,
   tokenize differently.  None of them is in the vocabulary.  */

#include "tokenizer.h"
#include "common.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <utf8proc.h>

/* Longest word that is split into pieces rather than made [UNK].  */
#define MAX_WORD_CHARS 100

static uint64_t
hash_bytes (const char *s, size_t len)
{
  uint64_t h = 14695981039346656037ULL;

  for (size_t i = 0; i < len; i++)
    {
      h ^= (unsigned char) s[i];
      h *= 1099511628211ULL;
    }
  return h;
}

int64_t
tokenizer_lookup (const struct tokenizer *t, const char *word, size_t len)
{
  size_t mask = t->table_size - 1;

  for (size_t i = hash_bytes (word, len) & mask; t->table[i];
       i = (i + 1) & mask)
    {
      const char *w = t->words[t->table[i] - 1];

      if (strncmp (w, word, len) == 0 && w[len] == '\0')
        return t->table[i] - 1;
    }
  return -1;
}

bool
tokenizer_load (struct tokenizer *t, const char *path)
{
  FILE *f = fopen (path, "r");
  char *line = NULL;
  size_t line_size = 0, cap = 0;
  ssize_t len;

  memset (t, 0, sizeof *t);
  if (!f)
    return false;
  while ((len = getline (&line, &line_size, f)) >= 0)
    {
      while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
        line[--len] = '\0';
      if (t->n_words == cap)
        {
          cap = cap ? 2 * cap : 32768;
          t->words = xrealloc (t->words, cap * sizeof *t->words);
        }
      t->words[t->n_words++] = xstrdup (line);
    }
  free (line);
  fclose (f);

  for (t->table_size = 1; t->table_size < 2 * t->n_words + 1;)
    t->table_size *= 2;
  t->table = xmalloc (t->table_size * sizeof *t->table);
  memset (t->table, 0, t->table_size * sizeof *t->table);
  for (size_t id = 0; id < t->n_words; id++)
    {
      const char *w = t->words[id];
      size_t mask = t->table_size - 1, i;

      /* The first of duplicate tokens wins.  */
      if (tokenizer_lookup (t, w, strlen (w)) >= 0)
        continue;
      for (i = hash_bytes (w, strlen (w)) & mask; t->table[i];
           i = (i + 1) & mask)
        continue;
      t->table[i] = id + 1;
    }

  t->unk = tokenizer_lookup (t, "[UNK]", 5);
  t->cls = tokenizer_lookup (t, "[CLS]", 5);
  t->sep = tokenizer_lookup (t, "[SEP]", 5);
  if (t->unk < 0 || t->cls < 0 || t->sep < 0)
    {
      tokenizer_free (t);
      errno = EINVAL;
      return false;
    }
  return true;
}

void
tokenizer_free (struct tokenizer *t)
{
  for (size_t i = 0; i < t->n_words; i++)
    free (t->words[i]);
  free (t->words);
  free (t->table);
  memset (t, 0, sizeof *t);
}

static bool
is_whitespace (utf8proc_int32_t c)
{
  switch (c)
    {
    case 0x09: case 0x0A: case 0x0B: case 0x0C: case 0x0D: case 0x20:
    case 0x85: case 0xA0: case 0x1680: case 0x2028: case 0x2029:
    case 0x202F: case 0x205F: case 0x3000:
      return true;
    default:
      return c >= 0x2000 && c <= 0x200A;
    }
}

static bool
is_control (utf8proc_int32_t c)
{
  if (c == '\t' || c == '\n' || c == '\r')
    return false;
  switch (utf8proc_category (c))
    {
    /* Not CN: unassigned characters are kept.  */
    case UTF8PROC_CATEGORY_CC: case UTF8PROC_CATEGORY_CF:
    case UTF8PROC_CATEGORY_CO: case UTF8PROC_CATEGORY_CS:
      return true;
    default:
      return false;
    }
}

static bool
is_punctuation (utf8proc_int32_t c)
{
  if (c < 0x80)
    return (c >= 0x21 && c <= 0x2F) || (c >= 0x3A && c <= 0x40)
      || (c >= 0x5B && c <= 0x60) || (c >= 0x7B && c <= 0x7E);
  switch (utf8proc_category (c))
    {
    case UTF8PROC_CATEGORY_PC: case UTF8PROC_CATEGORY_PD:
    case UTF8PROC_CATEGORY_PS: case UTF8PROC_CATEGORY_PE:
    case UTF8PROC_CATEGORY_PI: case UTF8PROC_CATEGORY_PF:
    case UTF8PROC_CATEGORY_PO:
      return true;
    default:
      return false;
    }
}

static bool
is_chinese (utf8proc_int32_t c)
{
  return (c >= 0x4E00 && c <= 0x9FFF) || (c >= 0x3400 && c <= 0x4DBF)
    || (c >= 0x20000 && c <= 0x2A6DF) || (c >= 0x2A700 && c <= 0x2B73F)
    /* Sic: the tokenizers library has 0x2B920 for 0x2B820.  */
    || (c >= 0x2B740 && c <= 0x2B81F) || (c >= 0x2B920 && c <= 0x2CEAF)
    || (c >= 0xF900 && c <= 0xFAFF) || (c >= 0x2F800 && c <= 0x2FA1F);
}

/* The state of a tokenization in progress.  */
struct run
{
  const struct tokenizer *t;
  int64_t *ids;
  size_t n, max;                /* MAX leaves room for [SEP].  */
  struct buf word;              /* The word being collected.  */
  size_t word_chars;
  struct buf piece;
};

static void
push (struct run *r, int64_t id)
{
  if (r->n < r->max)
    r->ids[r->n++] = id;
}

/* Split the collected word into word pieces, if there is one.  */
static void
flush_word (struct run *r)
{
  const char *w = r->word.data;
  size_t len = r->word.len, start = 0, first = r->n;

  if (len == 0)
    return;
  if (r->word_chars > MAX_WORD_CHARS)
    {
      push (r, r->t->unk);
      goto out;
    }
  while (start < len)
    {
      size_t end = len;
      int64_t id = -1;

      /* Find the longest piece from START that is in the vocabulary.  */
      for (; end > start; end--)
        {
          /* Pieces end on character boundaries.  */
          if (end < len && (w[end] & 0xC0) == 0x80)
            continue;
          r->piece.len = 0;
          r->piece.data[0] = '\0';
          if (start > 0)
            buf_puts (&r->piece, "##");
          buf_append (&r->piece, w + start, end - start);
          id = tokenizer_lookup (r->t, r->piece.data, r->piece.len);
          if (id >= 0)
            break;
        }
      if (id < 0)
        {
          /* The whole word is unknown.  */
          r->n = first;
          push (r, r->t->unk);
          goto out;
        }
      push (r, id);
      start = end;
    }

 out:
  r->word.len = 0;
  r->word.data[0] = '\0';
  r->word_chars = 0;
}

/* Add the normalized character C to the text.  */
static void
add_char (struct run *r, utf8proc_int32_t c)
{
  utf8proc_uint8_t bytes[4];
  utf8proc_ssize_t n;

  if (is_whitespace (c))
    {
      flush_word (r);
      return;
    }
  n = utf8proc_encode_char (c, bytes);
  if (is_punctuation (c))
    {
      flush_word (r);
      buf_append (&r->word, (char *) bytes, n);
      r->word_chars = 1;
      flush_word (r);
      return;
    }
  buf_append (&r->word, (char *) bytes, n);
  r->word_chars++;
}

/* Strip accents from and lowercase the character C, and add the
   result.  */
static void
add_normalized (struct run *r, utf8proc_int32_t c)
{
  utf8proc_int32_t parts[32];
  utf8proc_ssize_t n;
  int boundclass = 0;

  n = utf8proc_decompose_char (c, parts, 32, UTF8PROC_DECOMPOSE,
                               &boundclass);
  if (n < 0 || n > 32)
    {
      parts[0] = c;
      n = 1;
    }
  for (utf8proc_ssize_t i = 0; i < n; i++)
    {
      utf8proc_int32_t d = parts[i];

      if (utf8proc_category (d) == UTF8PROC_CATEGORY_MN)
        continue;
      /* The one character whose full lowercase is two characters.  */
      if (d == 0x130)
        {
          add_char (r, 'i');
          add_char (r, 0x307);
        }
      else
        add_char (r, utf8proc_tolower (d));
    }
}

/* If S starts with a special token, store its length in *LEN and
   return its id; otherwise return -1.  */
static int64_t
special_token (const struct tokenizer *t, const char *s, utf8proc_ssize_t *len)
{
  static const char *const names[] = {
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"
  };

  if (*s != '[')
    return -1;
  for (size_t i = 0; i < sizeof names / sizeof *names; i++)
    if (strncmp (s, names[i], strlen (names[i])) == 0)
      {
        *len = strlen (names[i]);
        return tokenizer_lookup (t, names[i], *len);
      }
  return -1;
}

size_t
tokenize (const struct tokenizer *t, const char *text, int64_t *ids,
          size_t max)
{
  struct run r = { .t = t, .ids = ids, .max = max - 1 };
  const utf8proc_uint8_t *p = (const utf8proc_uint8_t *) text;
  utf8proc_ssize_t left = strlen (text);

  buf_init (&r.word);
  buf_init (&r.piece);
  push (&r, t->cls);
  while (left > 0 && r.n < r.max)
    {
      utf8proc_int32_t c;
      utf8proc_ssize_t n;
      int64_t special = special_token (t, (const char *) p, &n);

      /* Special tokens in the text stand for themselves.  */
      if (special >= 0)
        {
          flush_word (&r);
          push (&r, special);
          p += n;
          left -= n;
          continue;
        }
      n = utf8proc_iterate (p, left, &c);

      if (n <= 0)
        {
          /* Skip one byte of invalid UTF-8.  */
          p++;
          left--;
          continue;
        }
      p += n;
      left -= n;
      if (c == 0 || c == 0xFFFD || is_control (c))
        continue;
      if (is_chinese (c))
        {
          flush_word (&r);
          add_normalized (&r, c);
          flush_word (&r);
        }
      else
        add_normalized (&r, c);
    }
  flush_word (&r);
  buf_free (&r.word);
  buf_free (&r.piece);
  ids[r.n++] = t->sep;
  return r.n;
}
