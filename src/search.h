/* search.h -- scoring the store's records against a query.
   Copyright (C) 2026 Martin Rixham

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   embed-query and embed-search-provider both search the store with
   these, so that they rank files alike.  A file scores the best of
   its record and its chunks, the vectors of passages spread through
   it.  */

#ifndef EMBED_SEARCH_H
#define EMBED_SEARCH_H

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <sys/types.h>

/* How the cosine between a query and a picture is put on the scale
   of text scores; see search.c.  */
#ifndef PICTURE_SCALE
#define PICTURE_SCALE 2.5
#endif
#ifndef PICTURE_OFFSET
#define PICTURE_OFFSET 0.40
#endif

/* How the cosine between two pictures is put on the same scale.  */
#ifndef PAIR_SCALE
#define PAIR_SCALE 1.5
#endif
#ifndef PAIR_OFFSET
#define PAIR_OFFSET (-0.5)
#endif

/* What a query is compared with.  */
struct query
{
  float *text;                  /* Its text vector, for texts.  */
  float *centred;               /* The same less its mean, for pictures.  */
  float *picture;               /* What it shows, or NULL.  */
  double picture_sum;           /* The sum of PICTURE's components.  */
};

struct match
{
  float score;
  char *path;
};

/* The line number of the newest record for a path, and the
   modification time and size of the file it was made from.  */
struct latest
{
  char *path;
  size_t line;
  long long mtime;
  long long size;
};

/* Insert PATH with SCORE into the N best MATCHES, which hold *USED
   entries sorted best first.  */
void insert_match (struct match *matches, size_t n, size_t *used,
                   const char *path, float score);

/* If LINE, of LEN bytes, is a complete record, end its path with a NUL
   and return the path; store the start of its vector in *VEC.  If
   PICTURE is not NULL, end the vector with a NUL too and store in
   *PICTURE the start of the picture vector, or NULL if there is none.
   Otherwise return NULL.  */
char *split_record (char *line, ssize_t len, char **vec, char **picture);

/* Whether PATH is one of the NULL-terminated WANT.  */
bool wanted (const char *path, char *const *want);

/* Read the records of F, from the current position to the end, and
   return the newest record for each path, sorted by path.  Line
   numbers count from 1 at the current position.  Store the number of
   paths in *N.  If WANT is not NULL, store in *FOUND a copy of the
   newest record for any of the paths it lists, or NULL if there is
   none.  */
struct latest *find_latest (FILE *f, size_t *n, char *const *want,
                            char **found);

void free_latest (struct latest *v, size_t n);

/* Fill in Q's centred vector and picture sum, from its text vector and
   picture vector of DIM floats.  */
void query_prepare (struct query *q, size_t dim);

void query_free (struct query *q);

/* Return how well a record with the text vector VEC and the picture
   vector PICTURE (or NULL) matches Q.  */
double score_record (const struct query *q, const float *vec,
                     const float *picture, size_t dim);

/* Open the chunk store, if there is one for vectors of DIM floats
   from the model that made the store, and return it positioned after
   its header; otherwise return NULL.  */
FILE *open_chunks (size_t dim);

/* Score the records of F, from the current position to the end, that
   LATEST, from find_latest at the same position, says are the newest,
   leaving out those in the NULL-terminated EXCLUDE (which may be NULL).
   A file whose chunks are in CHUNKS (which may be NULL), from
   open_chunks, scores the best of its record and those of its chunks
   that are of the file as it was when its record was made.  Return
   the COUNT best matches, best first, freshly allocated, and store how
   many there are in *USED.  */
struct match *search_records (FILE *f, const struct latest *latest,
                              size_t n_latest, const struct query *q,
                              size_t dim, char *const *exclude,
                              FILE *chunks, size_t count, size_t *used);

void free_matches (struct match *matches, size_t used);

#endif /* EMBED_SEARCH_H */
