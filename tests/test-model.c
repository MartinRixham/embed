/* test-model.c -- tests of model.c with the real model.
   Copyright (C) 2026 Martin Rixham

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   These need the model's files in ~/.local/share/embed/models, where
   embed-index downloads them, and are skipped without them; they never
   download anything.  The GPU tests are skipped if OpenVINO finds no
   GPU.  Only a few texts are embedded, to keep the machine usable.  */

#include "../model.c"

#include "testutil.h"

#include <check.h>
#include <math.h>
#include <openvino/c/openvino.h>

static char *real_models;       /* The model directory, if present.  */
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
                   "nomic-embed-text-v1.5", real_home) < 0)
    return EXIT_FAILURE;
  for (size_t i = 0; i < sizeof files / sizeof *files; i++)
    {
      char *p = path_join (real_models, files[i].local);
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
  suite_add_tcase (s, tc);
  sr = srunner_create (s);
  srunner_run_all (sr, CK_NORMAL);
  failed = srunner_ntests_failed (sr);
  srunner_free (sr);
  return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
