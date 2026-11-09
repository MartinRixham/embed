/* embed-query -- find files whose embeddings best match a query.
   Copyright (C) 2026 Martin Rixham

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   The query is embedded on the CPU, which for a single short text is
   quicker than starting up the GPU, by the model that built the store,
   and the store is scanned, keeping the best matches by cosine
   similarity.  A file with a picture vector, from an image or video,
   scores the better of its text and its picture.

   Pictures are compared with the query as Nomic's vision model
   expects: after a layer norm of the text model's output, which, since
   both vectors are of unit length, comes to taking out the query's
   mean.  Text queries match pictures with much lower cosines than they
   match texts, and over a narrower range: a picture of what the query
   names scores about 0.1 and others 0.01 to 0.05, while texts score
   about 0.45 when unrelated and 0.6 to 0.75 when they match.  So
   picture scores are mapped linearly onto the text scale.  */

#include "common.h"
#include "model.h"

#include <errno.h>
#include <math.h>
#include <error.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* How the cosine between a query and a picture is put on the scale
   of text scores; see above.  */
#ifndef PICTURE_SCALE
#define PICTURE_SCALE 2.5
#endif
#ifndef PICTURE_OFFSET
#define PICTURE_OFFSET 0.40
#endif

struct match
{
  float score;
  char *path;
};

static struct option const long_options[] = {
  {"count", required_argument, NULL, 'n'},
  {"scores", no_argument, NULL, 's'},
  {"help", no_argument, NULL, 'h'},
  {"version", no_argument, NULL, 'V'},
  {NULL, 0, NULL, 0}
};

static _Noreturn void
usage (int status)
{
  if (status != EXIT_SUCCESS)
    {
      fprintf (stderr, "Try 'embed-query --help' for more information.\n");
      exit (status);
    }
  printf ("Usage: embed-query [OPTION]... QUERY...\n"
          "List the files that best match QUERY, best first.\n\n"
          "  -n, --count=N        show N matches (default 10)\n"
          "  -s, --scores         show the similarity score of each match\n"
          "      --help           display this help and exit\n"
          "      --version        output version information and exit\n\n"
          "Run embed-index first to build the store.\n");
  exit (status);
}

/* Insert PATH with SCORE into the N best MATCHES, which hold *USED
   entries sorted best first.  */
static void
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

/* The line number of the newest record for a path.  */
struct latest
{
  char *path;
  size_t line;
};

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

/* If LINE, of LEN bytes, is a complete record, end its path with a NUL
   and return the path; store the start of its vector in *VEC.  If
   PICTURE is not NULL, end the vector with a NUL too and store in
   *PICTURE the start of the picture vector, or NULL if there is none.
   Otherwise return NULL.  */
static char *
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

/* Read the records of F, from the current position to the end, and
   return the newest record for each path, sorted by path.  Line
   numbers count from 1 at the current position.  Store the number of
   paths in *N.  */
static struct latest *
find_latest (FILE *f, size_t *n)
{
  struct latest *v = NULL;
  size_t used = 0, cap = 0, out = 0, line_no = 0, line_size = 0;
  char *line = NULL, *path, *vec;
  ssize_t len;

  while ((len = getline (&line, &line_size, f)) > 0)
    {
      line_no++;
      if (!(path = split_record (line, len, &vec, NULL)))
        continue;
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

int
main (int argc, char **argv)
{
  size_t count = 10, used = 0, dim, qdim;
  bool show_scores = false;
  char model[256];
  char *store = store_path ();
  char *line = NULL;
  size_t line_size = 0;
  ssize_t len;
  struct buf query;
  struct model *m;
  struct match *matches;
  struct latest *latest;
  size_t n_latest;
  long data_start;
  float *qvec, *pvec, *vec;
  double qmean = 0, qnorm = 0;
  FILE *f;
  int c;

  while ((c = getopt_long (argc, argv, "n:s", long_options, NULL)) != -1)
    switch (c)
      {
      case 'n':
        {
          long n;
          if (!parse_positive (optarg, 1000000, &n))
            error (EXIT_FAILURE, 0, "invalid count: %s", optarg);
          count = n;
        }
        break;
      case 's':
        show_scores = true;
        break;
      case 'h':
        usage (EXIT_SUCCESS);
      case 'V':
        printf ("embed-query %s\n", PACKAGE_VERSION);
        return EXIT_SUCCESS;
      default:
        usage (EXIT_FAILURE);
      }
  if (optind == argc)
    {
      error (0, 0, "missing query");
      usage (EXIT_FAILURE);
    }

  f = fopen (store, "r");
  if (!f)
    error (EXIT_FAILURE, errno, "%s (run embed-index first)", store);
  len = getline (&line, &line_size, f);
  if (len <= 0 || !parse_header (line, model, sizeof model, &dim))
    error (EXIT_FAILURE, 0, "%s: missing or invalid header", store);
  if (strcmp (model, EMBED_MODEL) != 0)
    error (EXIT_FAILURE, 0, "%s was built with %s; run embed-index to "
           "rebuild it with %s", store, model, EMBED_MODEL);

  buf_init (&query);
  buf_puts (&query, QUERY_PREFIX);
  for (int i = optind; i < argc; i++)
    {
      if (i > optind)
        buf_puts (&query, " ");
      buf_puts (&query, argv[i]);
    }

  /* A single short query is quicker on the CPU than it takes the GPU
     just to compile the model.  */
  m = model_open (false);
  if (!m)
    error (EXIT_FAILURE, 0, "cannot open the %s model", EMBED_MODEL);
  if (!model_embed (m, query.data, &qvec, &qdim))
    error (EXIT_FAILURE, 0, "cannot embed the query");
  model_close (m);
  if (qdim != dim)
    error (EXIT_FAILURE, 0, "query has dimension %zu but store has %zu",
           qdim, dim);

  /* The query as compared with pictures.  */
  pvec = xmalloc (dim * sizeof *pvec);
  for (size_t i = 0; i < dim; i++)
    qmean += qvec[i];
  qmean /= dim;
  for (size_t i = 0; i < dim; i++)
    {
      pvec[i] = qvec[i] - qmean;
      qnorm += (double) pvec[i] * pvec[i];
    }
  qnorm = sqrt (qnorm);
  for (size_t i = 0; i < dim; i++)
    pvec[i] = qnorm > 0 ? pvec[i] / qnorm : 0;

  /* A path can have several records, if an interrupted indexing run
     left them, or one is in progress: only the newest counts.  Find
     those first, so that a stale record cannot take the place of a
     better match.  */
  data_start = ftell (f);
  latest = find_latest (f, &n_latest);
  if (fseek (f, data_start, SEEK_SET) != 0)
    error (EXIT_FAILURE, errno, "%s", store);

  matches = xmalloc (count * sizeof *matches);
  vec = xmalloc (dim * sizeof *vec);
  for (size_t line_no = 1;
       (len = getline (&line, &line_size, f)) > 0; line_no++)
    {
      struct latest key, *newest;
      char *path, *p, *picture;
      double dot = 0;

      if (!(path = split_record (line, len, &p, &picture)))
        continue;
      key.path = path;
      newest = n_latest ? bsearch (&key, latest, n_latest, sizeof *latest,
                                   compare_paths) : NULL;
      if (!newest || newest->line != line_no)
        continue;

      if (!parse_hex_vector (p, vec, dim))
        continue;
      for (size_t i = 0; i < dim; i++)
        dot += (double) vec[i] * qvec[i];
      if (picture && parse_hex_vector (picture, vec, dim))
        {
          double pdot = 0;

          for (size_t i = 0; i < dim; i++)
            pdot += (double) vec[i] * pvec[i];
          pdot = PICTURE_OFFSET + PICTURE_SCALE * pdot;
          if (pdot > dot)
            dot = pdot;
        }
      insert_match (matches, count, &used, path, dot);
    }
  fclose (f);

  for (size_t i = 0; i < used; i++)
    {
      if (show_scores)
        printf ("%.3f\t", matches[i].score);
      printf ("%s\n", matches[i].path);
      free (matches[i].path);
    }

  for (size_t i = 0; i < n_latest; i++)
    free (latest[i].path);
  free (latest);
  free (matches);
  free (vec);
  free (qvec);
  free (pvec);
  free (line);
  buf_free (&query);
  return EXIT_SUCCESS;
}
