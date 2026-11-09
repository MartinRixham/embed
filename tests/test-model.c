/* test-model.c -- tests of model.c with the real model.
   Copyright (C) 2026 Martin Rixham

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   These need the model's files in ~/.local/share/embed/models, where
   embed-index downloads them, and are skipped without them; they never
   download anything.  The image and speech models' tests are skipped in
   the same way if just that model is missing.  The speech test needs
   espeak-ng and ffmpeg to make speech to transcribe.  The GPU tests are skipped if OpenVINO finds no
   GPU.  Only a few texts are embedded, to keep the machine usable.  */

#include "../model.c"

#include "testutil.h"

#include <check.h>
#include <math.h>
#include <openvino/c/openvino.h>

static char *real_models;       /* The model directory, if present.  */
static char *real_vision;       /* The image model's, if present.  */
static char *real_speech;       /* The speech model's, if present.  */
static char *home;

static void
setup (void)
{
  char *link;

  home = tmpdir_new ();
  link = path_join (home, ".local/share/embed/models/nomic-embed-text-v1.5");
  make_parent_dirs (link);
  ck_assert_int_eq (symlink (real_models, link), 0);
  free (link);
  link = path_join (home, ".local/share/embed/models/"
                    "nomic-embed-vision-v1.5");
  ck_assert_int_eq (symlink (real_vision, link), 0);
  free (link);
  link = path_join (home, ".local/share/embed/models/whisper-base");
  ck_assert_int_eq (symlink (real_speech, link), 0);
  free (link);
  setenv ("HOME", home, 1);
}

static void
teardown (void)
{
  tmpdir_remove (home);
}

static float *
embed (struct model *m, const char *text)
{
  float *v;
  size_t dim;

  ck_assert (model_embed (m, text, &v, &dim));
  ck_assert_uint_eq (dim, 768);
  return v;
}

static double
dot (const float *a, const float *b)
{
  double s = 0;

  for (size_t i = 0; i < 768; i++)
    s += (double) a[i] * b[i];
  return s;
}

/* A text of about N tokens.  */
static char *
words (size_t n)
{
  static const char *const w[] = {
    "the", "penguin", "colony", "walked", "across", "frozen", "sea", "ice",
    "toward", "open", "water", "where", "fish", "were", "plentiful"
  };
  struct buf b;

  buf_init (&b);
  buf_puts (&b, DOCUMENT_PREFIX);
  for (size_t i = 0; i < n; i++)
    {
      buf_puts (&b, w[i % (sizeof w / sizeof *w)]);
      buf_puts (&b, " ");
    }
  return b.data;
}

START_TEST (test_cpu_embeddings)
{
  struct model *m = model_open (false);
  float *penguins, *invoice, *bread, *query, *again;

  ck_assert_ptr_nonnull (m);
  ck_assert_str_eq (model_device (m), "CPU");
  penguins = embed (m, DOCUMENT_PREFIX "Emperor penguins huddle together "
                    "on the Antarctic ice");
  invoice = embed (m, DOCUMENT_PREFIX "Invoice 42 for consulting services "
                   "in March, payable within 30 days");
  bread = embed (m, DOCUMENT_PREFIX "A recipe for sourdough bread with a "
                 "long, slow fermentation");
  query = embed (m, QUERY_PREFIX "penguins");

  ck_assert_double_eq_tol (dot (query, query), 1.0, 1e-4);
  ck_assert_double_gt (dot (query, penguins), dot (query, invoice) + 0.1);
  ck_assert_double_gt (dot (query, penguins), dot (query, bread) + 0.1);

  /* The same text gives the same vector.  */
  again = embed (m, QUERY_PREFIX "penguins");
  ck_assert_double_eq_tol (dot (query, again), 1.0, 1e-6);
  model_close (m);
}
END_TEST

START_TEST (test_cpu_long_text_truncated)
{
  struct model *m = model_open (false);
  float *v;

  ck_assert_ptr_nonnull (m);
  v = embed (m, words (2000));
  ck_assert_double_eq_tol (dot (v, v), 1.0, 1e-4);
  model_close (m);
}
END_TEST

static bool
have_gpu (void)
{
  ov_core_t *core;
  ov_available_devices_t devices;
  bool found = false;

  if (ov_core_create (&core) != OK)
    return false;
  if (ov_core_get_available_devices (core, &devices) == OK)
    {
      for (size_t i = 0; i < devices.size; i++)
        if (strncmp (devices.devices[i], "GPU", 3) == 0)
          found = true;
      ov_available_devices_free (&devices);
    }
  ov_core_free (core);
  return found;
}

/* The GPU, padding each text to 128, 256 or 512 tokens, agrees with
   the CPU, which does not pad.  */
START_TEST (test_gpu_agrees_with_cpu)
{
  static const size_t lengths[] = { 20, 200, 400, 2000 };
  struct model *gpu, *cpu;

  if (!have_gpu ())
    {
      fprintf (stderr, "test_gpu_agrees_with_cpu: no GPU, skipped\n");
      return;
    }
  gpu = model_open (true);
  cpu = model_open (false);
  ck_assert_ptr_nonnull (gpu);
  ck_assert_ptr_nonnull (cpu);
  ck_assert_str_eq (model_device (gpu), "GPU");
  for (size_t i = 0; i < sizeof lengths / sizeof *lengths; i++)
    {
      char *text = words (lengths[i]);
      double c = dot (embed (gpu, text), embed (cpu, text));

      ck_assert_msg (c > 0.999, "%zu words: cosine %f", lengths[i], c);
      free (text);
    }
  model_close (gpu);
  model_close (cpu);
}
END_TEST

START_TEST (test_gpu_bad_model)
{
  const char *why = NULL;
  char *path = path_join (home, "not-a-model.onnx");

  if (!have_gpu ())
    return;
  write_file (path, "garbage", (size_t) -1);
  ck_assert_ptr_null (ov_embed_load (path, home, &why));
  ck_assert_str_eq (why, "OpenVINO cannot load the model");
  free (path);
}
END_TEST

START_TEST (test_cpu_bad_model)
{
  char *path = path_join (home, "not-a-model.onnx");

  write_file (path, "garbage", (size_t) -1);
  ck_assert_ptr_null (onnx_model_load (path, 1));
  free (path);
}
END_TEST

static bool
have_vision (void)
{
  char *p = path_join (real_vision, vision_files[0].local);
  bool ok = access (p, R_OK) == 0;

  if (!ok)
    fprintf (stderr, "%s is missing; image model test skipped\n", p);
  free (p);
  return ok;
}

/* Return a picture of one colour, freshly allocated.  */
static unsigned char *
plain_picture (unsigned char r, unsigned char g, unsigned char b)
{
  unsigned char *rgb = xmalloc (3 * VISION_SIDE * VISION_SIDE);

  for (size_t i = 0; i < VISION_SIDE * VISION_SIDE; i++)
    {
      rgb[3 * i] = r;
      rgb[3 * i + 1] = g;
      rgb[3 * i + 2] = b;
    }
  return rgb;
}

static float *
embed_picture (struct vision *v, const unsigned char *rgb)
{
  float *vec;
  size_t dim;

  ck_assert (vision_embed (v, rgb, &vec, &dim));
  ck_assert_uint_eq (dim, 768);
  return vec;
}

/* A query vector as compared with pictures: less its mean, at unit
   length, as embed-query does.  */
static float *
picture_query (struct model *m, const char *text)
{
  float *q = embed (m, text);
  double mean = 0, norm = 0;

  for (size_t i = 0; i < 768; i++)
    mean += q[i];
  mean /= 768;
  for (size_t i = 0; i < 768; i++)
    {
      q[i] -= mean;
      norm += (double) q[i] * q[i];
    }
  for (size_t i = 0; i < 768; i++)
    q[i] /= sqrt (norm);
  return q;
}

START_TEST (test_cpu_pictures)
{
  struct model *m;
  struct vision *v;
  float *red, *blue, *again, *q_red, *q_blue;

  if (!have_vision ())
    return;
  m = model_open (false);
  v = vision_open (false);
  ck_assert_ptr_nonnull (m);
  ck_assert_ptr_nonnull (v);
  red = embed_picture (v, plain_picture (230, 20, 20));
  blue = embed_picture (v, plain_picture (20, 20, 230));
  q_red = picture_query (m, QUERY_PREFIX "the colour red");
  q_blue = picture_query (m, QUERY_PREFIX "the colour blue");

  ck_assert_double_eq_tol (dot (red, red), 1.0, 1e-4);
  ck_assert_double_gt (dot (q_red, red), dot (q_red, blue));
  ck_assert_double_gt (dot (q_blue, blue), dot (q_blue, red));
  again = embed_picture (v, plain_picture (230, 20, 20));
  ck_assert_double_eq_tol (dot (red, again), 1.0, 1e-6);
  vision_close (v);
  model_close (m);
}
END_TEST

START_TEST (test_gpu_pictures_agree_with_cpu)
{
  struct vision *gpu, *cpu;
  unsigned char *rgb;

  if (!have_gpu () || !have_vision ())
    {
      fprintf (stderr, "test_gpu_pictures_agree_with_cpu: skipped\n");
      return;
    }
  gpu = vision_open (true);
  cpu = vision_open (false);
  ck_assert_ptr_nonnull (gpu);
  ck_assert_ptr_nonnull (cpu);
  ck_assert_ptr_nonnull (gpu->gpu);
  /* A picture with some detail: a gradient with stripes.  */
  rgb = plain_picture (0, 0, 0);
  for (size_t y = 0; y < VISION_SIDE; y++)
    for (size_t x = 0; x < VISION_SIDE; x++)
      {
        unsigned char *px = rgb + 3 * (y * VISION_SIDE + x);
        px[0] = x;
        px[1] = y;
        px[2] = (x / 16 + y / 16) % 2 ? 200 : 40;
      }
  double c = dot (embed_picture (gpu, rgb), embed_picture (cpu, rgb));
  ck_assert_msg (c > 0.999, "cosine %f", c);
  vision_close (gpu);
  vision_close (cpu);
}
END_TEST

static bool
have_speech (void)
{
  for (size_t i = 0; i < speech_source.n_files; i++)
    {
      char *p = path_join (real_speech, speech_files[i].local);
      bool ok = access (p, R_OK) == 0;

      if (!ok)
        fprintf (stderr, "%s is missing; speech model test skipped\n", p);
      free (p);
      if (!ok)
        return false;
    }
  return true;
}

/* Transcribe the N samples of PCM, returning the text.  */
static char *
hear (struct speech *s, const float *pcm, size_t n)
{
  struct buf b;

  buf_init (&b);
  ck_assert (speech_transcribe (s, pcm, n, &b));
  return b.data ? b.data : xstrdup ("");
}

/* Speak a sentence about penguins into PCM, and return the number of
   samples, or 0 if espeak-ng or ffmpeg is missing.  */
static size_t
speak (float *pcm)
{
  char *wav = path_join (home, "s.wav"), *raw = path_join (home, "s.f32");
  char *cmd;
  size_t n;
  FILE *f;

  if (!have_program ("espeak-ng") || !have_program ("ffmpeg"))
    {
      fprintf (stderr, "no espeak-ng or ffmpeg; speech test skipped\n");
      return 0;
    }
  if (asprintf (&cmd, "espeak-ng -w '%s' 'The emperor penguin is the "
                "tallest and heaviest of all living penguin species. It "
                "breeds during the Antarctic winter.' && ffmpeg -v quiet -i '%s' "
                "-ac 1 -ar %d -f f32le '%s' </dev/null", wav, wav,
                SPEECH_RATE, raw) < 0)
    abort ();
  ck_assert_int_eq (system (cmd), 0);
  f = fopen (raw, "rb");
  n = fread (pcm, sizeof *pcm, SPEECH_SAMPLES, f);
  fclose (f);
  free (cmd);
  free (wav);
  free (raw);
  return n;
}

START_TEST (test_speech)
{
  float *pcm = xmalloc (SPEECH_SAMPLES * sizeof *pcm);
  char *text;
  struct speech *s;
  size_t n;

  if (!have_speech ())
    return;
  s = speech_open (false);
  ck_assert_ptr_nonnull (s);

  /* Silence, and noise, say nothing.  */
  memset (pcm, 0, SPEECH_SAMPLES * sizeof *pcm);
  ck_assert_str_eq (hear (s, pcm, SPEECH_SAMPLES), "");
  ck_assert_str_eq (hear (s, pcm, 0), "");
  srand (1);
  for (size_t i = 0; i < 10 * SPEECH_RATE; i++)
    pcm[i] = 0.05 * (2.0 * rand () / RAND_MAX - 1);
  ck_assert_str_eq (hear (s, pcm, 10 * SPEECH_RATE), "");

  if ((n = speak (pcm)) > 0)
    {
      text = hear (s, pcm, n);
      ck_assert_msg (strcasestr (text, "penguin")
                     && strcasestr (text, "winter"), "heard: %s", text);
    }
  ck_assert_str_eq (whisper_device (s->whisper), "CPU");
  speech_close (s);
  free (pcm);
}
END_TEST

/* The encoder on the GPU hears the same as on the CPU.  */
START_TEST (test_speech_gpu_agrees_with_cpu)
{
  float *pcm = xmalloc (SPEECH_SAMPLES * sizeof *pcm);
  struct speech *gpu, *cpu;
  size_t n;

  if (!have_gpu () || !have_speech () || !(n = speak (pcm)))
    {
      fprintf (stderr, "test_speech_gpu_agrees_with_cpu: skipped\n");
      return;
    }
  gpu = speech_open (true);
  cpu = speech_open (false);
  ck_assert_ptr_nonnull (gpu);
  ck_assert_ptr_nonnull (cpu);
  ck_assert_str_eq (whisper_device (gpu->whisper), "GPU");
  ck_assert_str_eq (hear (gpu, pcm, n), hear (cpu, pcm, n));
  speech_close (gpu);
  speech_close (cpu);
  free (pcm);
}
END_TEST

int
main (void)
{
  const char *real_home = getenv ("HOME");
  Suite *s = suite_create ("model");
  TCase *tc = tcase_create ("model");
  SRunner *sr;
  int failed;

  if (!real_home
      || asprintf (&real_models, "%s/.local/share/embed/models/"
                   "nomic-embed-text-v1.5", real_home) < 0
      || asprintf (&real_vision, "%s/.local/share/embed/models/"
                   "nomic-embed-vision-v1.5", real_home) < 0
      || asprintf (&real_speech, "%s/.local/share/embed/models/"
                   "whisper-base", real_home) < 0)
    return EXIT_FAILURE;
  for (size_t i = 0; i < text_source.n_files; i++)
    {
      char *p = path_join (real_models, text_source.files[i].local);
      if (access (p, R_OK) != 0)
        {
          printf ("test-model: %s is missing; skipped (run embed-index to "
                  "download the model)\n", p);
          return EXIT_SUCCESS;
        }
      free (p);
    }
  /* Share the real cache of compiled GPU kernels, which saves several
     seconds of compiling.  */
  if (!getenv ("XDG_CACHE_HOME"))
    {
      char *cache = path_join (real_home, ".cache");
      setenv ("XDG_CACHE_HOME", cache, 1);
      free (cache);
    }

  tcase_add_checked_fixture (tc, setup, teardown);
  tcase_set_timeout (tc, 120);
  tcase_add_test (tc, test_cpu_embeddings);
  tcase_add_test (tc, test_cpu_long_text_truncated);
  tcase_add_test (tc, test_gpu_agrees_with_cpu);
  tcase_add_test (tc, test_gpu_bad_model);
  tcase_add_test (tc, test_cpu_bad_model);
  tcase_add_test (tc, test_cpu_pictures);
  tcase_add_test (tc, test_gpu_pictures_agree_with_cpu);
  tcase_add_test (tc, test_speech);
  tcase_add_test (tc, test_speech_gpu_agrees_with_cpu);
  suite_add_tcase (s, tc);
  sr = srunner_create (s);
  srunner_run_all (sr, CK_NORMAL);
  failed = srunner_ntests_failed (sr);
  srunner_free (sr);
  return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
