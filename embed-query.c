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
   picture scores are mapped linearly onto the text scale.

   With --file, the query is a file: its record in the store if that is
   up to date, and otherwise the vectors embed-index would give it,
   made on the spot.  Its text vector is compared with texts directly
   and with pictures as above, and its picture vector, if it has one,
   with pictures directly and with texts the other way about, by taking
   out each text's mean.  Pictures are much alike to the vision model:
   unrelated ones score about 0.8, and near copies above 0.95, so those
   cosines too are mapped onto the scale of texts, on which texts
   compared with other texts score about 0.65 when unrelated.  The
   file itself is left out of the matches.  */

#include "common.h"
#include "describe.h"
#include "model.h"

#include <errno.h>
#include <math.h>
#include <error.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/* How the cosine between a query and a picture is put on the scale
   of text scores; see above.  */
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

static struct option const long_options[] = {
  {"count", required_argument, NULL, 'n'},
  {"file", required_argument, NULL, 'f'},
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
          "  or:  embed-query [OPTION]... --file=FILE\n"
          "List the files that best match QUERY, or are most like FILE, "
          "best first.\n\n"
          "  -f, --file=FILE      find files like FILE: a document, "
          "image, video or\n"
          "                       recording, as indexed or else embedded "
          "now\n"
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

/* Whether PATH is one of the NULL-terminated WANT.  */
static bool
wanted (const char *path, char *const *want)
{
  for (; want && *want; want++)
    if (strcmp (path, *want) == 0)
      return true;
  return false;
}

/* Read the records of F, from the current position to the end, and
   return the newest record for each path, sorted by path.  Line
   numbers count from 1 at the current position.  Store the number of
   paths in *N.  If WANT is not NULL, store in *FOUND a copy of the
   newest record for any of the paths it lists, or NULL if there is
   none.  */
static struct latest *
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

static double
dot (const float *a, const float *b, size_t dim)
{
  double d = 0;

  for (size_t i = 0; i < dim; i++)
    d += (double) a[i] * b[i];
  return d;
}

/* Return how well a record with the text vector VEC and the picture
   vector PICTURE (or NULL) matches Q.  */
static double
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

/* Return the paths under which the store may know FILE, freshly
   allocated and ending with NULL: its real path and, if that is under
   the real path of HOME, the same under HOME as given, which is how
   embed-index names what it finds when it walks the home directory.
   Exit if FILE is not a regular file.  */
static char **
file_paths (const char *file, const char *home, struct stat *st)
{
  char **paths = xmalloc (3 * sizeof *paths);
  char *real = realpath (file, NULL), *real_home;
  size_t n = 0;

  if (!real || stat (real, st) != 0)
    error (EXIT_FAILURE, errno, "%s", file);
  if (!S_ISREG (st->st_mode))
    error (EXIT_FAILURE, 0, "%s: not a regular file", file);
  paths[n++] = real;
  real_home = home ? realpath (home, NULL) : NULL;
  if (real_home)
    {
      size_t len = strlen (real_home);

      if (strncmp (real, real_home, len) == 0 && real[len] == '/'
          && strcmp (real_home, home) != 0)
        {
          struct buf b;

          buf_init (&b);
          buf_puts (&b, home);
          buf_puts (&b, real + len);
          paths[n++] = b.data;
        }
      free (real_home);
    }
  paths[n] = NULL;
  return paths;
}

/* If RECORD, a line of the store, is up to date for a file whose
   status is ST, decode its vectors into Q and return true.  */
static bool
use_record (char *record, const struct stat *st, size_t dim,
            struct query *q)
{
  long long mtime, size;
  char *vec, *picture;

  if (!record || sscanf (record, "%lld %lld", &mtime, &size) != 2
      || mtime != (long long) st->st_mtime
      || size != (long long) st->st_size
      || !split_record (record, strlen (record), &vec, &picture))
    return false;
  q->text = xmalloc (dim * sizeof *q->text);
  if (!parse_hex_vector (vec, q->text, dim))
    {
      free (q->text);
      q->text = NULL;
      return false;
    }
  if (picture)
    {
      q->picture = xmalloc (dim * sizeof *q->picture);
      if (!parse_hex_vector (picture, q->picture, dim))
        {
          free (q->picture);
          q->picture = NULL;
        }
    }
  return true;
}

/* Embed FILE, known by PATH, into Q as embed-index would.  */
static void
embed_query_file (const char *file, const char *path, const char *home,
                  size_t dim, struct query *q)
{
  magic_t magic_mime = magic_open (MAGIC_MIME_TYPE | MAGIC_ERROR);
  magic_t magic_desc = magic_open (MAGIC_ERROR);
  struct model *m;

  if (!magic_mime || !magic_desc || magic_load (magic_mime, NULL) != 0
      || magic_load (magic_desc, NULL) != 0)
    error (EXIT_FAILURE, 0, "cannot load libmagic database");
  /* As for a text query, one file is quicker on the CPU.  */
  use_cpu = true;
  m = model_open (false);
  if (!m)
    error (EXIT_FAILURE, 0, "cannot open the %s model", EMBED_MODEL);
  if (!embed_file (m, magic_mime, magic_desc, path, home ? home : "/", dim,
                   &q->text, &q->picture))
    error (EXIT_FAILURE, 0, "%s: cannot embed the file", file);
  model_close (m);
  close_media_models ();
  magic_close (magic_mime);
  magic_close (magic_desc);
}

int
main (int argc, char **argv)
{
  size_t count = 10, used = 0, dim, qdim;
  bool show_scores = false;
  const char *query_file = NULL;
  const char *home = getenv ("HOME");
  char **file_ids = NULL, *record = NULL;
  struct query q = { NULL, NULL, NULL, 0 };
  char model[256];
  char *store = store_path ();
  char *line = NULL;
  size_t line_size = 0;
  ssize_t len;
  struct buf query;
  struct model *m;
  struct match *matches;
  struct latest *latest;
  struct stat st;
  size_t n_latest;
  long data_start;
  float *vec, *pvec;
  FILE *f;
  int c;

  while ((c = getopt_long (argc, argv, "f:n:s", long_options, NULL)) != -1)
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
      case 'f':
        query_file = optarg;
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
  if (optind == argc && !query_file)
    {
      error (0, 0, "missing query");
      usage (EXIT_FAILURE);
    }
  if (optind < argc && query_file)
    {
      error (0, 0, "a query cannot be given with --file");
      usage (EXIT_FAILURE);
    }
  if (query_file)
    file_ids = file_paths (query_file, home, &st);

  f = fopen (store, "r");
  if (!f)
    error (EXIT_FAILURE, errno, "%s (run embed-index first)", store);
  len = getline (&line, &line_size, f);
  if (len <= 0 || !parse_header (line, model, sizeof model, &dim))
    error (EXIT_FAILURE, 0, "%s: missing or invalid header", store);
  if (strcmp (model, EMBED_MODEL) != 0)
    error (EXIT_FAILURE, 0, "%s was built with %s; run embed-index to "
           "rebuild it with %s", store, model, EMBED_MODEL);

  /* A path can have several records, if an interrupted indexing run
     left them, or one is in progress: only the newest counts.  Find
     those first, so that a stale record cannot take the place of a
     better match.  */
  data_start = ftell (f);
  latest = find_latest (f, &n_latest, file_ids, &record);
  if (fseek (f, data_start, SEEK_SET) != 0)
    error (EXIT_FAILURE, errno, "%s", store);

  buf_init (&query);
  if (query_file)
    {
      if (!use_record (record, &st, dim, &q))
        embed_query_file (query_file, file_ids[file_ids[1] ? 1 : 0], home,
                          dim, &q);
    }
  else
    {
      buf_puts (&query, QUERY_PREFIX);
      for (int i = optind; i < argc; i++)
        {
          if (i > optind)
            buf_puts (&query, " ");
          buf_puts (&query, argv[i]);
        }

      /* A single short query is quicker on the CPU than it takes the
         GPU just to compile the model.  */
      m = model_open (false);
      if (!m)
        error (EXIT_FAILURE, 0, "cannot open the %s model", EMBED_MODEL);
      if (!model_embed (m, query.data, &q.text, &qdim))
        error (EXIT_FAILURE, 0, "cannot embed the query");
      model_close (m);
      if (qdim != dim)
        error (EXIT_FAILURE, 0, "query has dimension %zu but store has "
               "%zu", qdim, dim);
    }
  q.centred = centre (q.text, dim);
  if (q.picture)
    for (size_t i = 0; i < dim; i++)
      q.picture_sum += q.picture[i];

  matches = xmalloc (count * sizeof *matches);
  vec = xmalloc (dim * sizeof *vec);
  pvec = xmalloc (dim * sizeof *pvec);
  for (size_t line_no = 1;
       (len = getline (&line, &line_size, f)) > 0; line_no++)
    {
      struct latest key, *newest;
      char *path, *p, *picture;
      float *pic = NULL;

      if (!(path = split_record (line, len, &p, &picture)))
        continue;
      key.path = path;
      newest = n_latest ? bsearch (&key, latest, n_latest, sizeof *latest,
                                   compare_paths) : NULL;
      if (!newest || newest->line != line_no || wanted (path, file_ids))
        continue;

      if (!parse_hex_vector (p, vec, dim))
        continue;
      if (picture && parse_hex_vector (picture, pvec, dim))
        pic = pvec;
      insert_match (matches, count, &used, path,
                    score_record (&q, vec, pic, dim));
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
  free (pvec);
  free (q.text);
  free (q.centred);
  free (q.picture);
  free (record);
  for (size_t i = 0; file_ids && file_ids[i]; i++)
    free (file_ids[i]);
  free (file_ids);
  free (line);
  buf_free (&query);
  return EXIT_SUCCESS;
}
