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
#include <curl/curl.h>

#define PACKAGE_VERSION "0.1"
/* The embedding model.  Both programs use it, and pull it into the
   server's model store when it is missing.  */
#ifndef EMBED_MODEL
# define EMBED_MODEL "nomic-embed-text"
#endif

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

/* Connection to an Ollama server.  */
struct embedder
{
  CURL *curl;
  char *host;
  char *url;
  char *model;
};

/* Whether embed_text reports failures on standard error.  */
extern bool embed_report_errors;

void embedder_init (struct embedder *e, const char *host, const char *model);
void embedder_free (struct embedder *e);

/* Embed TEXT, giving up after TIMEOUT seconds.  On success store a
   freshly allocated, L2-normalized vector in *VEC and its length in
   *DIM and return true.  */
bool embed_text (struct embedder *e, const char *text, long timeout,
                 float **vec, size_t *dim);

/* Make sure the server has the model, pulling it from the Ollama
   registry if it does not.  Report problems on standard error and
   return false if the model is still unavailable.  */
bool ensure_model (struct embedder *e);

/* Model-specific prefixes that tell the model whether it is looking at
   a document or a search query.  */
const char *document_prefix (const char *model);
const char *query_prefix (const char *model);

/* Parse S as a whole decimal number from 1 to MAX and store it in
   *VALUE.  Return false if S is anything else.  */
bool parse_positive (const char *s, long max, long *value);

/* Create the missing directories leading up to PATH.  */
void make_parent_dirs (const char *path);

/* Return the path of the embedding store, freshly allocated.  */
char *store_path (void);

/* Parse the store header line LINE.  Store the model name in MODEL
   (at most MODEL_SIZE bytes) and the dimension in *DIM.  */
bool parse_header (const char *line, char *model, size_t model_size,
                   size_t *dim);

#endif /* EMBED_COMMON_H */
