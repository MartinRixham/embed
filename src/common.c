/* common.c -- shared code for embed-index and embed-query.
   Copyright (C) 2026 Martin Rixham

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.  */

#include "common.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <error.h>
#include <stdint.h>
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

char *
chunk_store_path (void)
{
  return data_path ("chunks.txt");
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

void
write_hex_vector (FILE *f, const float *vec, size_t dim)
{
  for (size_t i = 0; i < dim; i++)
    {
      uint32_t u;
      memcpy (&u, &vec[i], sizeof u);
      fprintf (f, "%08" PRIx32, u);
    }
}

/* One more than the value of each hex digit, and 0 for any other
   byte.  */
static const unsigned char hex_value[256] = {
  ['0'] = 1, ['1'] = 2, ['2'] = 3, ['3'] = 4, ['4'] = 5,
  ['5'] = 6, ['6'] = 7, ['7'] = 8, ['8'] = 9, ['9'] = 10,
  ['a'] = 11, ['b'] = 12, ['c'] = 13, ['d'] = 14, ['e'] = 15, ['f'] = 16,
  ['A'] = 11, ['B'] = 12, ['C'] = 13, ['D'] = 14, ['E'] = 15, ['F'] = 16
};

/* Decode the 8 hex digits at the start of S into *X.  */
static bool
parse_hex_vector_prefix (const char *s, float *x)
{
  const unsigned char *p = (const unsigned char *) s;
  uint32_t u = 0;

  for (int j = 0; j < 8; j++)
    {
      unsigned v = hex_value[p[j]];
      if (!v)
        return false;
      u = u << 4 | (v - 1);
    }
  memcpy (x, &u, sizeof u);
  return true;
}

bool
parse_hex_vector (const char *s, float *vec, size_t dim)
{
  const unsigned char *p = (const unsigned char *) s;

  for (size_t i = 0; i < dim; i++, p += 8)
    {
      uint32_t u = 0;

      /* A NUL is not a digit, so this never reads past the end.  */
      for (int j = 0; j < 8; j++)
        {
          unsigned v = hex_value[p[j]];
          if (!v)
            return false;
          u = u << 4 | (v - 1);
        }
      memcpy (&vec[i], &u, sizeof u);
    }
  return *p == '\n' || *p == '\0';
}

void
write_q8_vector (FILE *f, const float *vec, size_t dim)
{
  float max = 0, scale;

  for (size_t i = 0; i < dim; i++)
    if (fabsf (vec[i]) > max)
      max = fabsf (vec[i]);
  scale = max > 0 ? max / 127 : 1;
  write_hex_vector (f, &scale, 1);
  for (size_t i = 0; i < dim; i++)
    {
      long q = lrintf (vec[i] / scale);
      if (q > 127)
        q = 127;
      if (q < -127)
        q = -127;
      fprintf (f, "%02x", (unsigned) (q & 0xff));
    }
}

bool
parse_q8_vector (const char *s, float *vec, size_t dim)
{
  const unsigned char *p = (const unsigned char *) s + 8;
  float scale;

  if (!parse_hex_vector_prefix (s, &scale))
    return false;
  for (size_t i = 0; i < dim; i++, p += 2)
    {
      unsigned hi = hex_value[p[0]], lo;

      /* A NUL is not a digit, so this never reads past the end.  */
      if (!hi || !(lo = hex_value[p[1]]))
        return false;
      vec[i] = (signed char) ((hi - 1) << 4 | (lo - 1)) * scale;
    }
  return *p == '\n' || *p == '\0';
}
