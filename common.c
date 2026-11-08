/* common.c -- shared code for embed-index and embed-query.
   Copyright (C) 2026 Martin Rixham

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.  */

#include "common.h"

#include <errno.h>
#include <error.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

void *
xmalloc (size_t n)
{
  void *p = malloc (n ? n : 1);
  if (!p)
    error (EXIT_FAILURE, errno, "memory exhausted");
  return p;
}

void *
xrealloc (void *p, size_t n)
{
  p = realloc (p, n ? n : 1);
  if (!p)
    error (EXIT_FAILURE, errno, "memory exhausted");
  return p;
}

char *
xstrdup (const char *s)
{
  size_t n = strlen (s) + 1;
  return memcpy (xmalloc (n), s, n);
}

void
buf_init (struct buf *b)
{
  b->cap = 256;
  b->len = 0;
  b->data = xmalloc (b->cap);
  b->data[0] = '\0';
}

void
buf_free (struct buf *b)
{
  free (b->data);
  b->data = NULL;
  b->len = b->cap = 0;
}

void
buf_append (struct buf *b, const char *s, size_t n)
{
  if (b->len + n + 1 > b->cap)
    {
      while (b->len + n + 1 > b->cap)
        b->cap *= 2;
      b->data = xrealloc (b->data, b->cap);
    }
  memcpy (b->data + b->len, s, n);
  b->len += n;
  b->data[b->len] = '\0';
}

void
buf_puts (struct buf *b, const char *s)
{
  buf_append (b, s, strlen (s));
}

void
make_parent_dirs (const char *path)
{
  char *copy = xstrdup (path);

  for (char *p = copy + 1; *p; p++)
    if (*p == '/')
      {
        *p = '\0';
        if (mkdir (copy, 0755) != 0 && errno != EEXIST)
          error (EXIT_FAILURE, errno, "cannot create %s", copy);
        *p = '/';
      }
  free (copy);
}

bool
parse_positive (const char *s, long max, long *value)
{
  char *end;
  long v;

  if (*s < '0' || *s > '9')
    return false;
  errno = 0;
  v = strtol (s, &end, 10);
  if (errno != 0 || *end != '\0' || v < 1 || v > max)
    return false;
  *value = v;
  return true;
}

char *
data_path (const char *name)
{
  const char *home = getenv ("HOME");
  struct buf b;

  if (!home || !*home)
    error (EXIT_FAILURE, 0, "HOME is not set");
  buf_init (&b);
  buf_puts (&b, home);
  buf_puts (&b, "/.local/share/embed/");
  buf_puts (&b, name);
  return b.data;
}

char *
store_path (void)
{
  return data_path ("embeddings.txt");
}

bool
parse_header (const char *line, char *model, size_t model_size,
              size_t *dim)
{
  const char *m = strstr (line, "model=");
  const char *d = strstr (line, "dim=");
  size_t len;

  if (line[0] != '#' || !m || !d)
    return false;
  m += strlen ("model=");
  len = strcspn (m, " \t\n");
  if (len == 0 || len >= model_size)
    return false;
  memcpy (model, m, len);
  model[len] = '\0';
  *dim = strtoul (d + strlen ("dim="), NULL, 10);
  return *dim > 0;
}
