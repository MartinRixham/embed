/* fetch.c -- download a file and check it.
   Copyright (C) 2026 Martin Rixham

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.  */

#include "fetch.h"
#include "common.h"

#include <curl/curl.h>
#include <openssl/evp.h>
#include <errno.h>
#include <error.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Give up on a download that receives nothing for this many seconds.  */
#ifndef STALL_TIMEOUT
# define STALL_TIMEOUT 60
#endif

struct download
{
  FILE *f;
  EVP_MD_CTX *hash;
  bool failed;                  /* Writing to F failed.  */
  int percent;                  /* Last progress shown, or -1.  */
  const char *name;
};

static size_t
write_callback (char *ptr, size_t size, size_t nmemb, void *userdata)
{
  struct download *d = userdata;
  size_t n = size * nmemb;

  if (fwrite (ptr, 1, n, d->f) != n)
    {
      d->failed = true;
      return 0;
    }
  if (!EVP_DigestUpdate (d->hash, ptr, n))
    {
      d->failed = true;
      return 0;
    }
  return n;
}

static int
progress_callback (void *userdata, curl_off_t total, curl_off_t now,
                   curl_off_t up_total, curl_off_t up_now)
{
  struct download *d = userdata;
  int percent;

  (void) up_total;
  (void) up_now;
  if (total <= 0)
    return 0;
  percent = now * 100 / total;
  if (percent != d->percent)
    {
      fprintf (stderr, "\r%s: %d%% of %.0f MB", d->name, percent,
               total / 1e6);
      d->percent = percent;
    }
  return 0;
}

bool
fetch_file (const char *url, const char *path, const char *sha256)
{
  struct download d = { .percent = -1 };
  struct buf part;
  CURL *curl;
  CURLcode rc;
  long status = 0;
  unsigned char digest[EVP_MAX_MD_SIZE];
  unsigned int digest_len = 0;
  char hex[2 * EVP_MAX_MD_SIZE + 1];
  bool ok = false, show = isatty (STDERR_FILENO);

  if (access (path, F_OK) == 0)
    return true;

  d.name = strrchr (path, '/') ? strrchr (path, '/') + 1 : path;
  buf_init (&part);
  buf_puts (&part, path);
  buf_puts (&part, ".part");
  make_parent_dirs (path);
  d.f = fopen (part.data, "w");
  if (!d.f)
    {
      error (0, errno, "%s", part.data);
      buf_free (&part);
      return false;
    }
  d.hash = EVP_MD_CTX_new ();
  if (!d.hash || !EVP_DigestInit_ex (d.hash, EVP_sha256 (), NULL))
    error (EXIT_FAILURE, 0, "cannot initialize SHA-256");

  curl_global_init (CURL_GLOBAL_DEFAULT);
  curl = curl_easy_init ();
  if (!curl)
    error (EXIT_FAILURE, 0, "cannot initialize libcurl");
  curl_easy_setopt (curl, CURLOPT_URL, url);
  curl_easy_setopt (curl, CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt (curl, CURLOPT_FAILONERROR, 1L);
  curl_easy_setopt (curl, CURLOPT_NOSIGNAL, 1L);
  curl_easy_setopt (curl, CURLOPT_LOW_SPEED_LIMIT, 1L);
  curl_easy_setopt (curl, CURLOPT_LOW_SPEED_TIME, (long) STALL_TIMEOUT);
  curl_easy_setopt (curl, CURLOPT_WRITEFUNCTION, write_callback);
  curl_easy_setopt (curl, CURLOPT_WRITEDATA, &d);
  if (show)
    {
      curl_easy_setopt (curl, CURLOPT_NOPROGRESS, 0L);
      curl_easy_setopt (curl, CURLOPT_XFERINFOFUNCTION, progress_callback);
      curl_easy_setopt (curl, CURLOPT_XFERINFODATA, &d);
    }
  rc = curl_easy_perform (curl);
  curl_easy_getinfo (curl, CURLINFO_RESPONSE_CODE, &status);
  if (show && d.percent >= 0)
    fputc ('\n', stderr);
  curl_easy_cleanup (curl);
  curl_global_cleanup ();

  if (fclose (d.f) != 0)
    d.failed = true;
  if (!d.failed && !EVP_DigestFinal_ex (d.hash, digest, &digest_len))
    error (EXIT_FAILURE, 0, "cannot compute SHA-256");
  EVP_MD_CTX_free (d.hash);
  if (d.failed)
    error (0, errno, "cannot write %s", part.data);
  else if (rc == CURLE_HTTP_RETURNED_ERROR)
    error (0, 0, "cannot download %s: the server answered HTTP %ld", url,
           status);
  else if (rc != CURLE_OK)
    error (0, 0, "cannot download %s: %s", url, curl_easy_strerror (rc));
  else
    {
      for (unsigned int i = 0; i < digest_len; i++)
        snprintf (hex + 2 * i, 3, "%02x", digest[i]);
      if (strcmp (hex, sha256) != 0)
        error (0, 0, "%s: download is corrupt (SHA-256 %s, expected %s)",
               url, hex, sha256);
      else if (rename (part.data, path) != 0)
        error (0, errno, "cannot rename %s", part.data);
      else
        ok = true;
    }
  if (!ok)
    unlink (part.data);
  buf_free (&part);
  return ok;
}
