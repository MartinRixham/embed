/* search.c -- scoring the store's records against a query.
   Copyright (C) 2026 Martin Rixham

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   A file with a picture vector, from an image or video, scores the
   better of its text and its picture.

   Pictures are compared with a text query as Nomic's vision model
   expects: after a layer norm of the text model's output, which, since
   both vectors are of unit length, comes to taking out the query's
   mean.  Text queries match pictures with much lower cosines than they
   match texts, and over a narrower range: a picture of what the query
   names scores about 0.1 and others 0.01 to 0.05, while texts score
   about 0.45 when unrelated and 0.6 to 0.75 when they match.  So
   picture scores are mapped linearly onto the text scale.

   A query that is a file has a text vector, compared as above, and
   may have a picture vector, compared with pictures directly and with
   texts the other way about, by taking out each text's mean.  Pictures
   are much alike to the vision model: unrelated ones score about 0.8,
   and near copies above 0.95, so those cosines too are mapped onto the
   scale of texts, on which texts compared with other texts score about
   0.65 when unrelated.  */

#include "search.h"
#include "common.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

void
insert_match (struct match *matches, size_t n, size_t *used,
              const char *path, float score)
{
  size_t i;

  if (*used == n && score <= matches[n - 1].score)
    return;
  if (*used == n)
    free (matches[--(*used)].path);

  for (i = *used; i > 0 && matches[i - 1].score < score; i--)
    matches[i] = matches[i - 1];
  matches[i].score = score;
  matches[i].path = xstrdup (path);
  (*used)++;
}

static int
compare_latest (const void *a, const void *b)
{
  const struct latest *x = a, *y = b;
  int c = strcmp (x->path, y->path);

  if (c != 0)
    return c;
  return (x->line > y->line) - (x->line < y->line);
}

static int
compare_paths (const void *a, const void *b)
{
  const struct latest *x = a, *y = b;
  return strcmp (x->path, y->path);
}

char *
split_record (char *line, ssize_t len, char **vec, char **picture)
{
  char *path, *p;

  if (len <= 0 || line[len - 1] != '\n')
    return NULL;
  path = strchr (line, '\t');
  if (!path)
    return NULL;
  path++;
  p = strchr (path, '\t');
  if (!p)
    return NULL;
  *p = '\0';
  *vec = p + 1;
  if (picture)
    {
      *picture = NULL;
      if ((p = strchr (*vec, '\t')))
        {
          *p = '\0';
          if (p[1] != '-')
            *picture = p + 1;
        }
    }
  return path;
}

bool
wanted (const char *path, char *const *want)
{
  for (; want && *want; want++)
    if (strcmp (path, *want) == 0)
      return true;
  return false;
}

struct latest *
find_latest (FILE *f, size_t *n, char *const *want, char **found)
{
  struct latest *v = NULL;
  size_t used = 0, cap = 0, out = 0, line_no = 0, line_size = 0;
  char *line = NULL, *path, *vec;
  ssize_t len;

  if (found)
    *found = NULL;
  while ((len = getline (&line, &line_size, f)) > 0)
    {
      char *copy = want ? xstrdup (line) : NULL;

      line_no++;
      if (!(path = split_record (line, len, &vec, NULL)))
        {
          free (copy);
          continue;
        }
      if (copy && wanted (path, want))
        {
          free (*found);
          *found = copy;
        }
      else
        free (copy);
      if (used == cap)
        {
          cap = cap ? 2 * cap : 1024;
          v = xrealloc (v, cap * sizeof *v);
        }
      v[used].path = xstrdup (path);
      v[used].line = line_no;
      used++;
    }
  free (line);

  if (used > 0)
    qsort (v, used, sizeof *v, compare_latest);
  for (size_t i = 0; i < used; i++)
    {
      if (out > 0 && strcmp (v[out - 1].path, v[i].path) == 0)
        {
          free (v[out - 1].path);
          v[out - 1] = v[i];
        }
      else
        v[out++] = v[i];
    }
  *n = out;
  return v;
}

void
free_latest (struct latest *v, size_t n)
{
  for (size_t i = 0; i < n; i++)
    free (v[i].path);
  free (v);
}

/* Return a freshly allocated unit vector: VEC less the mean of its DIM
   components, or all zeros if that is zero.  */
static float *
centre (const float *vec, size_t dim)
{
  float *c = xmalloc (dim * sizeof *c);
  double mean = 0, norm = 0;

  for (size_t i = 0; i < dim; i++)
    mean += vec[i];
  mean /= dim;
  for (size_t i = 0; i < dim; i++)
    {
      c[i] = vec[i] - mean;
      norm += (double) c[i] * c[i];
    }
  norm = sqrt (norm);
  for (size_t i = 0; i < dim; i++)
    c[i] = norm > 0 ? c[i] / norm : 0;
  return c;
}

void
query_prepare (struct query *q, size_t dim)
{
  q->centred = centre (q->text, dim);
  q->picture_sum = 0;
  if (q->picture)
    for (size_t i = 0; i < dim; i++)
      q->picture_sum += q->picture[i];
}

void
query_free (struct query *q)
{
  free (q->text);
  free (q->centred);
  free (q->picture);
  q->text = q->centred = q->picture = NULL;
}

static double
dot (const float *a, const float *b, size_t dim)
{
  double d = 0;

  for (size_t i = 0; i < dim; i++)
    d += (double) a[i] * b[i];
  return d;
}

double
score_record (const struct query *q, const float *vec, const float *picture,
              size_t dim)
{
  double best = dot (vec, q->text, dim);

  if (picture)
    {
      double s = PICTURE_OFFSET + PICTURE_SCALE
        * dot (picture, q->centred, dim);
      if (s > best)
        best = s;
    }
  if (q->picture)
    {
      /* Against VEC less its mean, at unit length: VEC is of unit
         length, so that length is sqrt (1 - DIM * MEAN^2).  */
      double mean = 0, len, s;

      for (size_t i = 0; i < dim; i++)
        mean += vec[i];
      mean /= dim;
      len = 1 - dim * mean * mean;
      if (len > 0)
        {
          s = (dot (vec, q->picture, dim) - mean * q->picture_sum)
            / sqrt (len);
          s = PICTURE_OFFSET + PICTURE_SCALE * s;
          if (s > best)
            best = s;
        }
      if (picture)
        {
          s = PAIR_OFFSET + PAIR_SCALE * dot (picture, q->picture, dim);
          if (s > best)
            best = s;
        }
    }
  return best;
}

struct match *
search_records (FILE *f, const struct latest *latest, size_t n_latest,
                const struct query *q, size_t dim, char *const *exclude,
                size_t count, size_t *used)
{
  struct match *matches = xmalloc (count * sizeof *matches);
  float *vec = xmalloc (dim * sizeof *vec);
  float *pvec = xmalloc (dim * sizeof *pvec);
  char *line = NULL;
  size_t line_size = 0;
  ssize_t len;

  *used = 0;
  for (size_t line_no = 1;
       (len = getline (&line, &line_size, f)) > 0; line_no++)
    {
      struct latest key;
      const struct latest *newest;
      char *path, *p, *picture;
      float *pic = NULL;

      if (!(path = split_record (line, len, &p, &picture)))
        continue;
      key.path = path;
      newest = n_latest ? bsearch (&key, latest, n_latest, sizeof *latest,
                                   compare_paths) : NULL;
      if (!newest || newest->line != line_no || wanted (path, exclude))
        continue;

      if (!parse_hex_vector (p, vec, dim))
        continue;
      if (picture && parse_hex_vector (picture, pvec, dim))
        pic = pvec;
      insert_match (matches, count, used, path,
                    score_record (q, vec, pic, dim));
    }
  free (line);
  free (vec);
  free (pvec);
  return matches;
}

void
free_matches (struct match *matches, size_t used)
{
  for (size_t i = 0; i < used; i++)
    free (matches[i].path);
  free (matches);
}
