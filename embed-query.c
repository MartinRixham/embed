/* embed-query -- find files whose embeddings best match a query.
   Copyright (C) 2026 Martin Rixham

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   The query is embedded with the same model that built the store, and
   the store is scanned once, keeping the best matches by cosine
   similarity.  */

#include "common.h"

#include <errno.h>
#include <error.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct match
{
  float score;
  char *path;
};

static struct option const long_options[] = {
  {"count", required_argument, NULL, 'n'},
  {"host", required_argument, NULL, 'H'},
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
          "  -H, --host=URL       Ollama server (default %s)\n"
          "      --help           display this help and exit\n"
          "      --version        output version information and exit\n\n"
          "Run embed-index first to build the store.\n", DEFAULT_HOST);
  exit (status);
}

/* Insert PATH with SCORE into the N best MATCHES, which hold *USED
   entries sorted best first.  A path already present is replaced,
   since later records in the store supersede earlier ones.  */
static void
insert_match (struct match *matches, size_t n, size_t *used,
              const char *path, float score)
{
  size_t i;

  for (i = 0; i < *used; i++)
    if (strcmp (matches[i].path, path) == 0)
      {
        free (matches[i].path);
        memmove (matches + i, matches + i + 1,
                 (*used - i - 1) * sizeof *matches);
        (*used)--;
        break;
      }

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

int
main (int argc, char **argv)
{
  const char *host = DEFAULT_HOST;
  size_t count = 10, used = 0, dim, qdim;
  bool show_scores = false;
  char model[256];
  char *store = store_path ();
  char *line = NULL;
  size_t line_size = 0;
  ssize_t len;
  struct buf query;
  struct embedder emb;
  struct match *matches;
  float *qvec;
  FILE *f;
  int c;

  while ((c = getopt_long (argc, argv, "n:H:s", long_options, NULL)) != -1)
    switch (c)
      {
      case 'n':
        count = strtoul (optarg, NULL, 10);
        if (count == 0)
          error (EXIT_FAILURE, 0, "invalid count: %s", optarg);
        break;
      case 'H':
        host = optarg;
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

  buf_init (&query);
  buf_puts (&query, query_prefix (model));
  for (int i = optind; i < argc; i++)
    {
      if (i > optind)
        buf_puts (&query, " ");
      buf_puts (&query, argv[i]);
    }

  embedder_init (&emb, host, model);
  if (!embed_text (&emb, query.data, 120, &qvec, &qdim))
    error (EXIT_FAILURE, 0, "cannot embed query with %s at %s", model,
           host);
  if (qdim != dim)
    error (EXIT_FAILURE, 0, "query has dimension %zu but store has %zu",
           qdim, dim);

  matches = xmalloc (count * sizeof *matches);
  while ((len = getline (&line, &line_size, f)) > 0)
    {
      char *path, *p, *end;
      double dot = 0;
      size_t i;

      if (line[len - 1] != '\n')
        continue;
      path = strchr (line, '\t');
      if (!path)
        continue;
      path++;
      p = strchr (path, '\t');
      if (!p)
        continue;
      *p++ = '\0';

      for (i = 0; i < dim; i++, p = end)
        {
          float v = strtof (p, &end);
          if (end == p)
            break;
          dot += (double) v * qvec[i];
        }
      if (i == dim)
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

  free (matches);
  free (qvec);
  free (line);
  buf_free (&query);
  embedder_free (&emb);
  return EXIT_SUCCESS;
}
