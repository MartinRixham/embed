/* fake-model.c -- a stand-in for the embedding model, for tests.
   Copyright (C) 2026 Martin Rixham

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.  */

#include "fake-model.h"
#include "../model.h"

#include <error.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>

const char *const fake_keywords[FAKE_DIM] = {
  "sourdough", "invoice", "sonata", "quicksort", "sunset", "secret",
  "alpha", "omega"
};

struct model
{
  bool gpu;
};

static void
log_text (const char *text)
{
  const char *log = getenv ("FAKE_MODEL_LOG");
  FILE *f;

  if (!log || !(f = fopen (log, "a")))
    return;
  fprintf (f, "%s\n----\n", text);
  fclose (f);
}

struct model *
model_open (bool use_gpu)
{
  const char *state = getenv ("FAKE_MODEL");
  struct model *m;

  if (state && strcmp (state, "fail") == 0)
    {
      error (0, 0, "fake model cannot be opened");
      return NULL;
    }
  log_text (use_gpu ? "open gpu=1" : "open gpu=0");
  m = malloc (sizeof *m);
  m->gpu = use_gpu;
  return m;
}

void
model_close (struct model *m)
{
  free (m);
}

const char *
model_device (const struct model *m)
{
  return m->gpu ? "GPU" : "CPU";
}

static size_t
count_ci (const char *hay, const char *needle)
{
  size_t n = 0, m = strlen (needle);

  for (const char *p = hay; *p; p++)
    if (strncasecmp (p, needle, m) == 0)
      n++;
  return n;
}

bool
model_embed (struct model *m, const char *text, float **vec, size_t *dim)
{
  size_t d = strstr (text, "FAKE_DIM3") ? 3 : FAKE_DIM;
  double norm = 0;
  float *v;

  (void) m;
  log_text (text);
  if (strstr (text, "FAKE_SLOW"))
    {
      struct timespec ts = { 3, 0 };
      nanosleep (&ts, NULL);
    }
  if (strstr (text, "FAKE_500"))
    {
      error (0, 0, "fake model failed");
      return false;
    }
  v = malloc (d * sizeof *v);
  for (size_t i = 0; i < d; i++)
    {
      v[i] = strstr (text, "FAKE_ZERO") ? 0
        : 0.05 + count_ci (text, fake_keywords[i]);
      norm += (double) v[i] * v[i];
    }
  norm = sqrt (norm);
  if (norm > 0)
    for (size_t i = 0; i < d; i++)
      v[i] /= norm;
  *vec = v;
  *dim = d;
  return true;
}
