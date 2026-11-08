/* tokenizer.h -- the BERT WordPiece tokenizer of nomic-embed-text.
   Copyright (C) 2026 Martin Rixham

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.  */

#ifndef EMBED_TOKENIZER_H
#define EMBED_TOKENIZER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* A WordPiece vocabulary.  */
struct tokenizer
{
  char **words;                 /* By token id.  */
  size_t n_words;
  uint32_t *table;              /* Hash table of id + 1, 0 if empty.  */
  size_t table_size;            /* A power of two.  */
  int64_t unk, cls, sep;        /* Ids of the special tokens.  */
};

/* Load the vocabulary in PATH, one token per line, the line number
   being its id.  Return false, with errno set, if it cannot be read
   or lacks the special tokens.  */
bool tokenizer_load (struct tokenizer *t, const char *path);
void tokenizer_free (struct tokenizer *t);

/* Return the id of the token WORD, LEN bytes long, or -1.  */
int64_t tokenizer_lookup (const struct tokenizer *t, const char *word,
                          size_t len);

/* Tokenize the UTF-8 string TEXT as the Hugging Face BertTokenizer
   does for nomic-embed-text: lowercase, strip accents, split off
   punctuation and Chinese characters, and split the words into word
   pieces.  Store at most MAX ids in IDS, starting with [CLS] and ending
   with [SEP] (MAX must be at least 2), and return how many there are.
   Invalid UTF-8 is ignored.  */
size_t tokenize (const struct tokenizer *t, const char *text, int64_t *ids,
                 size_t max);

#endif /* EMBED_TOKENIZER_H */
