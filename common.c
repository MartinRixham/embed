/* common.c -- shared code for embed-index and embed-query.
   Copyright (C) 2026 Martin Rixham

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.  */

#include "common.h"

#include <errno.h>
#include <error.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

bool embed_report_errors = true;

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

/* Return the length of the valid UTF-8 sequence starting at S, which
   has N bytes available, or 0 if it is not valid.  Overlong forms,
   UTF-16 surrogates and values above U+10FFFF are not valid.  */
static size_t
utf8_len (const unsigned char *s, size_t n)
{
  size_t len, i;
  unsigned char lo = 0x80, hi = 0xbf;  /* Range of the second byte.  */

  if (s[0] < 0x80)
    return 1;
  else if (s[0] >= 0xc2 && s[0] <= 0xdf)
    len = 2;
  else if (s[0] >= 0xe0 && s[0] <= 0xef)
    {
      len = 3;
      if (s[0] == 0xe0)
        lo = 0xa0;
      else if (s[0] == 0xed)
        hi = 0x9f;
    }
  else if (s[0] >= 0xf0 && s[0] <= 0xf4)
    {
      len = 4;
      if (s[0] == 0xf0)
        lo = 0x90;
      else if (s[0] == 0xf4)
        hi = 0x8f;
    }
  else
    return 0;

  if (len > n || s[1] < lo || s[1] > hi)
    return 0;
  for (i = 2; i < len; i++)
    if ((s[i] & 0xc0) != 0x80)
      return 0;
  return len;
}

/* Append S to B as a JSON string literal.  Invalid UTF-8 and control
   characters are replaced so that the server always accepts it.  */
static void
buf_json_string (struct buf *b, const char *s)
{
  const unsigned char *p = (const unsigned char *) s;
  size_t n = strlen (s);

  buf_puts (b, "\"");
  while (n > 0)
    {
      size_t len = utf8_len (p, n);

      if (len == 0)
        {
          buf_puts (b, "?");
          len = 1;
        }
      else if (*p == '"')
        buf_puts (b, "\\\"");
      else if (*p == '\\')
        buf_puts (b, "\\\\");
      else if (*p == '\n')
        buf_puts (b, "\\n");
      else if (*p < 0x20 || *p == 0x7f)
        buf_puts (b, " ");
      else
        buf_append (b, (const char *) p, len);
      p += len;
      n -= len;
    }
  buf_puts (b, "\"");
}

/* Return P advanced past any JSON white space.  */
static const char *
skip_space (const char *p)
{
  while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')
    p++;
  return p;
}

/* Find the first vector of the "embeddings" member in the JSON
   response RESP, and return a pointer just past its opening bracket,
   or NULL.  */
static const char *
find_vector (const char *resp)
{
  const char *p = strstr (resp, "\"embeddings\"");

  if (!p)
    return NULL;
  p = skip_space (p + strlen ("\"embeddings\""));
  if (*p++ != ':')
    return NULL;
  p = skip_space (p);
  if (*p++ != '[')
    return NULL;
  p = skip_space (p);
  if (*p++ != '[')
    return NULL;
  return p;
}

static size_t
write_callback (char *ptr, size_t size, size_t nmemb, void *userdata)
{
  buf_append (userdata, ptr, size * nmemb);
  return size * nmemb;
}

void
embedder_init (struct embedder *e, const char *host, const char *model)
{
  struct buf url;

  curl_global_init (CURL_GLOBAL_DEFAULT);
  e->curl = curl_easy_init ();
  if (!e->curl)
    error (EXIT_FAILURE, 0, "cannot initialize libcurl");

  buf_init (&url);
  buf_puts (&url, host);
  buf_puts (&url, "/api/embed");
  e->url = url.data;
  e->model = xstrdup (model);
}

void
embedder_free (struct embedder *e)
{
  curl_easy_cleanup (e->curl);
  curl_global_cleanup ();
  free (e->url);
  free (e->model);
}

bool
embed_text (struct embedder *e, const char *text, long timeout,
            float **vec, size_t *dim)
{
  struct buf req, resp;
  struct curl_slist *headers = NULL;
  CURLcode rc;
  long status = 0;
  bool ok = false;
  const char *p;
  float *v = NULL;
  size_t n = 0, cap = 0;
  double norm = 0;

  buf_init (&req);
  buf_init (&resp);
  buf_puts (&req, "{\"model\":");
  buf_json_string (&req, e->model);
  buf_puts (&req, ",\"keep_alive\":\"10m\",\"truncate\":true,\"input\":");
  buf_json_string (&req, text);
  buf_puts (&req, "}");

  headers = curl_slist_append (headers, "Content-Type: application/json");
  curl_easy_reset (e->curl);
  curl_easy_setopt (e->curl, CURLOPT_URL, e->url);
  curl_easy_setopt (e->curl, CURLOPT_HTTPHEADER, headers);
  curl_easy_setopt (e->curl, CURLOPT_POSTFIELDS, req.data);
  curl_easy_setopt (e->curl, CURLOPT_POSTFIELDSIZE, (long) req.len);
  curl_easy_setopt (e->curl, CURLOPT_WRITEFUNCTION, write_callback);
  curl_easy_setopt (e->curl, CURLOPT_WRITEDATA, &resp);
  curl_easy_setopt (e->curl, CURLOPT_TIMEOUT, timeout);
  curl_easy_setopt (e->curl, CURLOPT_NOSIGNAL, 1L);

  rc = curl_easy_perform (e->curl);
  curl_slist_free_all (headers);
  if (rc != CURLE_OK)
    {
      if (embed_report_errors)
        error (0, 0, "embedding request failed: %s",
               curl_easy_strerror (rc));
      goto out;
    }
  curl_easy_getinfo (e->curl, CURLINFO_RESPONSE_CODE, &status);

  p = find_vector (resp.data);
  if (status != 200 || !p)
    {
      if (embed_report_errors)
        error (0, 0, "embedding server returned %ld: %.200s", status,
               resp.data);
      goto out;
    }

  /* Parse the comma-separated list of numbers in the first vector.  */
  for (p = skip_space (p); *p != ']'; )
    {
      char *end;
      float f = strtof (p, &end);

      if (end == p)
        break;
      if (n == cap)
        {
          cap = cap ? 2 * cap : 1024;
          v = xrealloc (v, cap * sizeof *v);
        }
      v[n++] = f;
      norm += (double) f * f;
      p = skip_space (end);
      if (*p == ',')
        p = skip_space (p + 1);
      else if (*p != ']')
        break;
    }
  if (n == 0 || *p != ']')
    {
      if (embed_report_errors)
        error (0, 0, "malformed embedding response");
      goto out;
    }

  norm = sqrt (norm);
  if (norm > 0)
    for (size_t i = 0; i < n; i++)
      v[i] /= norm;

  *vec = v;
  *dim = n;
  v = NULL;
  ok = true;

 out:
  free (v);
  buf_free (&req);
  buf_free (&resp);
  return ok;
}

const char *
document_prefix (const char *model)
{
  if (strstr (model, "nomic"))
    return "search_document: ";
  return "";
}

const char *
query_prefix (const char *model)
{
  if (strstr (model, "nomic"))
    return "search_query: ";
  if (strstr (model, "qwen3-embedding"))
    return "Instruct: Given a short search query, retrieve the files "
      "on a personal computer that best match it\nQuery: ";
  return "";
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
store_path (void)
{
  const char *home = getenv ("HOME");
  struct buf b;

  if (!home || !*home)
    error (EXIT_FAILURE, 0, "HOME is not set");
  buf_init (&b);
  buf_puts (&b, home);
  buf_puts (&b, "/.local/share/embed/embeddings.txt");
  return b.data;
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
