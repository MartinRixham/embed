/* fake-model.c -- a stand-in for the embedding model, for tests.
   Copyright (C) 2026 Martin Rixham

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.  */

#include "fake-model.h"
#include "../model.h"
#include "../common.h"

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

struct vision
{
  int unused;
};

struct vision *
vision_open (bool use_gpu)
{
  const char *state = getenv ("FAKE_VISION");

  (void) use_gpu;
  if (state && strcmp (state, "fail") == 0)
    {
      error (0, 0, "fake vision model cannot be opened");
      return NULL;
    }
  return malloc (sizeof (struct vision));
}

void
vision_close (struct vision *v)
{
  free (v);
}

bool
vision_embed (struct vision *v, const unsigned char *rgb, float **vec,
              size_t *dim)
{
  const size_t area = VISION_SIDE * VISION_SIDE;
  double sum[3] = { 0, 0, 0 }, norm = 0;
  float *p = malloc (FAKE_DIM * sizeof *p);

  (void) v;
  for (size_t i = 0; i < area; i++)
    for (size_t c = 0; c < 3; c++)
      sum[c] += rgb[3 * i + c];
  for (size_t i = 0; i < FAKE_DIM; i++)
    {
      p[i] = 0.05 + (i < 3 ? sum[i] / area / 255 : 0);
      norm += (double) p[i] * p[i];
    }
  norm = sqrt (norm);
  for (size_t i = 0; i < FAKE_DIM; i++)
    p[i] /= norm;
  *vec = p;
  *dim = FAKE_DIM;
  return true;
}

struct speech
{
  int unused;
};

struct speech *
speech_open (bool use_gpu)
{
  const char *state = getenv ("FAKE_SPEECH");

  (void) use_gpu;
  if (state && strcmp (state, "fail") == 0)
    {
      error (0, 0, "fake speech model cannot be opened");
      return NULL;
    }
  return malloc (sizeof (struct speech));
}

void
speech_close (struct speech *s)
{
  free (s);
}

bool
speech_transcribe (struct speech *s, const float *pcm, size_t n,
                   struct buf *out)
{
  char line[64];
  double sum = 0;

  (void) s;
  snprintf (line, sizeof line, "SPEECH %zu", n);
  log_text (line);
  for (size_t i = 0; i < n; i++)
    sum += (double) pcm[i] * pcm[i];
  if (n > 0 && sum / n > 0.001)
    buf_puts (out, " sonata");
  return true;
}
