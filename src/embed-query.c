/* embed-query -- find files whose embeddings best match a query.
   Copyright (C) 2026 Martin Rixham

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   The query is embedded on the CPU, which for a single short text is
   quicker than starting up the GPU, by the model that built the store,
   and the store is scanned, keeping the best matches; see search.c for
   how they are scored.

   With --file, the query is a file: its record in the store if that is
   up to date, and otherwise the vectors embed-index would give it,
   made on the spot.  The file itself is left out of the matches.  */

#include "common.h"
#include "describe.h"
#include "model.h"
#include "search.h"

#include <errno.h>
#include <error.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

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
  query_prepare (&q, dim);
  matches = search_records (f, latest, n_latest, &q, dim, file_ids, count,
                            &used);
  fclose (f);

  for (size_t i = 0; i < used; i++)
    {
      if (show_scores)
        printf ("%.3f\t", matches[i].score);
      printf ("%s\n", matches[i].path);
    }

  free_matches (matches, used);
  free_latest (latest, n_latest);
  query_free (&q);
  free (record);
  for (size_t i = 0; file_ids && file_ids[i]; i++)
    free (file_ids[i]);
  free (file_ids);
  free (line);
  buf_free (&query);
  return EXIT_SUCCESS;
}
