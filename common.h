/* common.h -- shared declarations for embed-index and embed-query.
   Copyright (C) 2026 Martin Rixham

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.  */

#ifndef EMBED_COMMON_H
#define EMBED_COMMON_H

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

#define PACKAGE_VERSION "0.2"

/* The embedding model, as named in the store's header.  */
#define EMBED_MODEL "nomic-embed-text-v1.5"

/* Prefixes that tell the model whether it is looking at a document or
   a search query.  */
#define DOCUMENT_PREFIX "search_document: "
#define QUERY_PREFIX "search_query: "

/* A growable byte buffer, always NUL terminated.  */
struct buf
{
  char *data;
  size_t len;
  size_t cap;
};

void buf_init (struct buf *b);
void buf_free (struct buf *b);
void buf_append (struct buf *b, const char *s, size_t n);
void buf_puts (struct buf *b, const char *s);

void *xmalloc (size_t n);
void *xrealloc (void *p, size_t n);
char *xstrdup (const char *s);

/* Parse S as a whole decimal number from 1 to MAX and store it in
   *VALUE.  Return false if S is anything else.  */
bool parse_positive (const char *s, long max, long *value);

/* Create the missing directories leading up to PATH.  */
void make_parent_dirs (const char *path);

/* Return ~/.local/share/embed/NAME, freshly allocated.  */
char *data_path (const char *name);

/* Return the path of the embedding store, freshly allocated.  */
char *store_path (void);

/* Parse the store header line LINE.  Store the model name in MODEL
   (at most MODEL_SIZE bytes) and the dimension in *DIM.  */
bool parse_header (const char *line, char *model, size_t model_size,
                   size_t *dim);

/* Write the DIM floats of VEC to F, each as the 8 hex digits of its
   IEEE 754 bit pattern, most significant first, with no separators.
   This is exact, and much quicker to read back than decimal.  */
void write_hex_vector (FILE *f, const float *vec, size_t dim);

/* Decode a vector written by write_hex_vector from S into the DIM
   floats of VEC.  Return false unless S holds exactly DIM * 8 hex
   digits followed by a newline or the end of the string.  */
bool parse_hex_vector (const char *s, float *vec, size_t dim);

#endif /* EMBED_COMMON_H */
