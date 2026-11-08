/* test-query.c -- unit tests for embed-query.c.
   Copyright (C) 2026 Martin Rixham

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   The file under test is included directly so that its static
   functions can be exercised; its main is renamed so that the tests
   can run it, in a child process, as the real program would run.  */

#define main query_main
#include "../embed-query.c"
#undef main

#include "fake-model.h"
#include "testutil.h"

#include <check.h>
#include <stdarg.h>
#include <sys/stat.h>

#ifndef EMBED_INDEX
# error "EMBED_INDEX must name the built embed-index program"
#endif

struct fixture
{
  char *root;
  char *home;
  char *store;
  char *log;
  char *out;
  char *err;
};

static struct fixture fx;

static void
setup (void)
{
  fx.root = tmpdir_new ();
  fx.home = path_join (fx.root, "home");
  mkdir (fx.home, 0755);
  fx.store = path_join (fx.home, ".local/share/embed/embeddings.txt");
  fx.log = path_join (fx.root, "requests.log");
  fx.out = path_join (fx.root, "stdout");
  fx.err = path_join (fx.root, "stderr");
  setenv ("HOME", fx.home, 1);
  setenv ("FAKE_MODEL_LOG", fx.log, 1);
  unsetenv ("FAKE_MODEL");
}

static void
teardown (void)
{
  tmpdir_remove (fx.root);
}

/* Write a store for MODEL, with records given as pairs of a path and a
   keyword index (or -1 for a vector halfway between keywords 0 and 1),
   ending with NULL.  Each vector is a unit vector along its keyword's
   dimension, as embed-index would store it.  */
static void
make_store (const char *model, const char *path, ...)
{
  struct buf b;
  va_list ap;
  char line[512];

  buf_init (&b);
  snprintf (line, sizeof line, "# embed-index model=%s dim=%d\n", model,
            FAKE_DIM);
  buf_puts (&b, line);
  va_start (ap, path);
  for (const char *p = path; p; p = va_arg (ap, const char *))
    {
      int k = va_arg (ap, int);
      int n = snprintf (line, sizeof line, "1 1\t%s\t", p);

      for (int i = 0; i < FAKE_DIM; i++)
        {
          double v = k == -1 ? (i < 2 ? sqrt (0.5) : 0) : i == k;
          n += snprintf (line + n, sizeof line - n, i ? " %g" : "%g", v);
        }
      buf_puts (&b, line);
      buf_puts (&b, "\n");
    }
  va_end (ap);
  write_file (fx.store, b.data, b.len);
  buf_free (&b);
}

/* Run embed-query with the fake model and the given arguments,
   ending with NULL.  */
static int
run_query (const char *arg, ...)
{
  char *argv[32];
  int argc = 0;
  va_list ap;

  argv[argc++] = "embed-query";
  va_start (ap, arg);
  for (const char *a = arg; a; a = va_arg (ap, const char *))
    argv[argc++] = (char *) a;
  va_end (ap);
  argv[argc] = NULL;
  return run_main (query_main, argv, fx.out, fx.err);
}

static char *
out_text (void)
{
  char *s = read_file (fx.out);
  return s ? s : strdup ("");
}

static char *
err_text (void)
{
  char *s = read_file (fx.err);
  return s ? s : strdup ("");
}

/* insert_match.  */

static void
free_matches (struct match *m, size_t used)
{
  for (size_t i = 0; i < used; i++)
    free (m[i].path);
}

START_TEST (test_insert_sorted)
{
  struct match m[5];
  size_t used = 0;
  static const float scores[] = { 0.3f, 0.9f, 0.1f, 0.5f, 0.7f };
  static const char *const paths[] = { "c", "a", "e", "d", "b" };

  for (int i = 0; i < 5; i++)
    insert_match (m, 5, &used, paths[i], scores[i]);
  ck_assert_uint_eq (used, 5);
  ck_assert_str_eq (m[0].path, "a");
  ck_assert_str_eq (m[1].path, "b");
  ck_assert_str_eq (m[2].path, "d");
  ck_assert_str_eq (m[3].path, "c");
  ck_assert_str_eq (m[4].path, "e");
  free_matches (m, used);
}
END_TEST

START_TEST (test_insert_keeps_best_n)
{
  struct match m[3];
  size_t used = 0;

  for (int i = 0; i < 100; i++)
    {
      char path[16];
      snprintf (path, sizeof path, "p%d", i);
      insert_match (m, 3, &used, path, (float) ((i * 37) % 100));
    }
  ck_assert_uint_eq (used, 3);
  ck_assert_float_eq (m[0].score, 99);
  ck_assert_float_eq (m[1].score, 98);
  ck_assert_float_eq (m[2].score, 97);
  free_matches (m, used);
}
END_TEST

START_TEST (test_insert_rejects_when_full)
{
  struct match m[2];
  size_t used = 0;

  insert_match (m, 2, &used, "a", 0.5f);
  insert_match (m, 2, &used, "b", 0.4f);
  insert_match (m, 2, &used, "tie", 0.4f);
  insert_match (m, 2, &used, "low", 0.1f);
  ck_assert_uint_eq (used, 2);
  ck_assert_str_eq (m[0].path, "a");
  ck_assert_str_eq (m[1].path, "b");
  free_matches (m, used);
}
END_TEST

START_TEST (test_insert_single_slot)
{
  struct match m[1];
  size_t used = 0;

  insert_match (m, 1, &used, "a", 0.2f);
  insert_match (m, 1, &used, "b", 0.6f);
  insert_match (m, 1, &used, "c", 0.4f);
  ck_assert_uint_eq (used, 1);
  ck_assert_str_eq (m[0].path, "b");
  free_matches (m, used);
}
END_TEST

START_TEST (test_insert_negative_scores)
{
  struct match m[3];
  size_t used = 0;

  insert_match (m, 3, &used, "a", -0.5f);
  insert_match (m, 3, &used, "b", -0.1f);
  insert_match (m, 3, &used, "c", -0.9f);
  ck_assert_str_eq (m[0].path, "b");
  ck_assert_str_eq (m[2].path, "c");
  free_matches (m, used);
}
END_TEST

/* The whole program.  */

START_TEST (test_query_ranks)
{
  make_store (EMBED_MODEL,
              "/h/bread.txt", 0, "/h/invoice.pdf", 1, "/h/sonata.mp3", 2,
              "/h/mixed.txt", -1, NULL);
  ck_assert_int_eq (run_query ("sourdough", NULL), 0);
  ck_assert (strncmp (out_text (), "/h/bread.txt\n/h/mixed.txt\n", 26)
             == 0);
  ck_assert_int_eq (run_query ("invoice", NULL), 0);
  ck_assert (strncmp (out_text (), "/h/invoice.pdf\n/h/mixed.txt\n", 28)
             == 0);
  ck_assert_int_eq (run_query ("-n", "1", "sonata", NULL), 0);
  ck_assert_str_eq (out_text (), "/h/sonata.mp3\n");
  ck_assert_str_eq (err_text (), "");
}
END_TEST

START_TEST (test_query_scores)
{
  char *out;
  double s1, s2;
  char p1[64], p2[64];

  make_store (EMBED_MODEL, "/h/a", 0, "/h/b", 1, NULL);
  ck_assert_int_eq (run_query ("-s", "sourdough", NULL), 0);
  out = out_text ();
  ck_assert_int_eq (sscanf (out, "%lf\t%63s\n%lf\t%63s", &s1, p1, &s2, p2),
                    4);
  ck_assert_str_eq (p1, "/h/a");
  ck_assert_str_eq (p2, "/h/b");
  /* The query vector, (1.05, 0.05, 0.05, ...), normalized.  */
  double norm = sqrt (1.05 * 1.05 + 7 * 0.05 * 0.05);
  ck_assert_double_eq_tol (s1, 1.05 / norm, 0.001);
  ck_assert_double_eq_tol (s2, 0.05 / norm, 0.001);
  /* Three decimal places.  */
  ck_assert_ptr_nonnull (strstr (out, "0.992\t"));
}
END_TEST

START_TEST (test_query_default_count)
{
  struct buf b;
  char line[256];

  buf_init (&b);
  snprintf (line, sizeof line, "# embed-index model=" EMBED_MODEL " dim=%d\n", FAKE_DIM);
  buf_puts (&b, line);
  for (int i = 0; i < 25; i++)
    {
      snprintf (line, sizeof line, "1 1\t/f%02d\t1 0 0 0 0 0 0 0\n", i);
      buf_puts (&b, line);
    }
  write_file (fx.store, b.data, b.len);
  ck_assert_int_eq (run_query ("x", NULL), 0);
  ck_assert_uint_eq (count_substr (out_text (), "\n"), 10);
  ck_assert_int_eq (run_query ("-n", "100", "x", NULL), 0);
  ck_assert_uint_eq (count_substr (out_text (), "\n"), 25);
}
END_TEST

START_TEST (test_query_joins_words_and_prefixes)
{
  char *log;

  make_store (EMBED_MODEL, "/h/a", 0, NULL);
  ck_assert_int_eq (run_query ("my", "tax", "receipts", NULL), 0);
  log = read_file (fx.log);
  ck_assert_str_eq (log, "open gpu=0\n----\n"
                    "search_query: my tax receipts\n----\n");
}
END_TEST

START_TEST (test_query_rejects_other_model)
{
  make_store ("qwen3-embedding:0.6b", "/h/a", 0, NULL);
  ck_assert_int_eq (run_query ("x", NULL), EXIT_FAILURE);
  ck_assert_ptr_nonnull (strstr (err_text (), "built with "
                                 "qwen3-embedding:0.6b; run embed-index"));
  /* Without opening the model.  */
  ck_assert_ptr_null (read_file (fx.log));
}
END_TEST

START_TEST (test_query_skips_bad_lines)
{
  char data[2048];

  snprintf (data, sizeof data,
            "# embed-index model=" EMBED_MODEL " dim=%d\n"
            "no tabs at all\n"
            "1 1\tonly one tab\n"
            "1 1\t/short\t1 0 0\n"
            "1 1\t/words\tone two three four five six seven eight\n"
            "1 1\t/good\t1 0 0 0 0 0 0 0\n"
            "\n"
            "1 1\t/truncated\t1 0 0 0 0 0 0 0", FAKE_DIM);
  write_file (fx.store, data, (size_t) -1);
  ck_assert_int_eq (run_query ("sourdough", NULL), 0);
  ck_assert_str_eq (out_text (), "/good\n");
}
END_TEST

START_TEST (test_query_newest_record_wins)
{
  /* /a was re-embedded but the store was not yet compacted.  */
  make_store (EMBED_MODEL, "/a", 0, "/b", 2, "/a", 1, NULL);
  ck_assert_int_eq (run_query ("-s", "sourdough", NULL), 0);
  ck_assert_uint_eq (count_substr (out_text (), "/a\n"), 1);
  ck_assert_int_eq (run_query ("-n", "1", "invoice", NULL), 0);
  ck_assert_str_eq (out_text (), "/a\n");
}
END_TEST

START_TEST (test_query_stale_record_does_not_hide_match)
{
  /* /a scored best, but its newer record scores badly: /c, rejected
     while /a and /b filled the two places, must still be found.  */
  make_store (EMBED_MODEL, "/a", 0, "/b", 0, "/c", -1, "/a", 2, NULL);
  ck_assert_int_eq (run_query ("-n", "2", "sourdough", NULL), 0);
  ck_assert_str_eq (out_text (), "/b\n/c\n");
}
END_TEST

START_TEST (test_query_paths_with_spaces)
{
  make_store (EMBED_MODEL, "/h/My Documents/tax return 2025.pdf", 1, NULL);
  ck_assert_int_eq (run_query ("invoice", NULL), 0);
  ck_assert_str_eq (out_text (), "/h/My Documents/tax return 2025.pdf\n");
}
END_TEST

START_TEST (test_query_empty_store)
{
  make_store (EMBED_MODEL, NULL);
  ck_assert_int_eq (run_query ("anything", NULL), 0);
  ck_assert_str_eq (out_text (), "");
}
END_TEST

START_TEST (test_query_missing_store)
{
  ck_assert_int_eq (run_query ("x", NULL), EXIT_FAILURE);
  ck_assert_ptr_nonnull (strstr (err_text (), "run embed-index first"));
  /* Without even opening the model.  */
  ck_assert_ptr_null (read_file (fx.log));
}
END_TEST

START_TEST (test_query_bad_header)
{
  write_file (fx.store, "1 1\t/a\t1 0\n", (size_t) -1);
  ck_assert_int_eq (run_query ("x", NULL), EXIT_FAILURE);
  ck_assert_ptr_nonnull (strstr (err_text (), "invalid header"));
}
END_TEST

START_TEST (test_query_dimension_mismatch)
{
  write_file (fx.store, "# embed-index model=" EMBED_MODEL " dim=3\n1 1\t/a\t1 0 0\n",
              (size_t) -1);
  ck_assert_int_eq (run_query ("x", NULL), EXIT_FAILURE);
  ck_assert_ptr_nonnull (strstr (err_text (), "dimension 8 but store has "
                                 "3"));
}
END_TEST

START_TEST (test_query_model_unavailable)
{
  make_store (EMBED_MODEL, "/a", 0, NULL);
  setenv ("FAKE_MODEL", "fail", 1);
  ck_assert_int_eq (run_query ("x", NULL), EXIT_FAILURE);
  ck_assert_ptr_nonnull (strstr (err_text (), "cannot open the "
                                 EMBED_MODEL " model"));
  ck_assert_str_eq (out_text (), "");
}
END_TEST

START_TEST (test_query_embedding_fails)
{
  make_store (EMBED_MODEL, "/a", 0, NULL);
  ck_assert_int_eq (run_query ("FAKE_500", NULL), EXIT_FAILURE);
  ck_assert_ptr_nonnull (strstr (err_text (), "cannot embed the query"));
}
END_TEST

START_TEST (test_query_removed_host_option)
{
  make_store (EMBED_MODEL, "/a", 0, NULL);
  ck_assert_int_eq (run_query ("-H", "http://localhost:1", "x", NULL),
                    EXIT_FAILURE);
  ck_assert_ptr_null (read_file (fx.log));
}
END_TEST

START_TEST (test_query_missing_query)
{
  make_store (EMBED_MODEL, "/a", 0, NULL);
  ck_assert_int_eq (run_query (NULL), EXIT_FAILURE);
  ck_assert_ptr_nonnull (strstr (err_text (), "missing query"));
}
END_TEST

START_TEST (test_query_bad_count)
{
  static const char *const bad[] = {
    "0", "-1", "abc", "5x", "", "1000001", "99999999999999999999", NULL
  };

  make_store (EMBED_MODEL, "/a", 0, NULL);
  for (int i = 0; bad[i]; i++)
    ck_assert_msg (run_query ("-n", bad[i], "x", NULL) == EXIT_FAILURE,
                   "count '%s' accepted", bad[i]);
}
END_TEST

START_TEST (test_query_options)
{
  char *help[] = { "embed-query", "--help", NULL };
  char *version[] = { "embed-query", "--version", NULL };
  char *bad[] = { "embed-query", "-z", "x", NULL };

  ck_assert_int_eq (run_main (query_main, help, fx.out, fx.err), 0);
  ck_assert (strncmp (out_text (), "Usage: embed-query", 18) == 0);
  ck_assert_int_eq (run_main (query_main, version, fx.out, fx.err), 0);
  ck_assert_str_eq (out_text (), "embed-query " PACKAGE_VERSION "\n");
  ck_assert_int_eq (run_main (query_main, bad, fx.out, fx.err),
                    EXIT_FAILURE);
}
END_TEST

START_TEST (test_round_trip_with_indexer)
{
  char *cmd;

  /* Build a store with the embed-index program (built with the fake
     model), then search it.  */
  write_file (path_join (fx.home, "Recipes/rye.txt"),
              "A sourdough rye with caraway.", (size_t) -1);
  write_file (path_join (fx.home, "Work/billing.txt"),
              "Invoice 1043, payable in 30 days.", (size_t) -1);
  write_file (path_join (fx.home, "Music/playlist.m3u"),
              "#EXTM3U\nMoonlight sonata.mp3\n", (size_t) -1);
  if (asprintf (&cmd, "'%s' 2>/dev/null", EMBED_INDEX) < 0)
    abort ();
  ck_assert_int_eq (system (cmd), 0);

  ck_assert_int_eq (run_query ("-n", "1", "sourdough", "bread", NULL), 0);
  ck_assert (strstr (out_text (), "/Recipes/rye.txt\n"));
  ck_assert_int_eq (run_query ("-n", "1", "unpaid", "invoice", NULL), 0);
  ck_assert (strstr (out_text (), "/Work/billing.txt\n"));
  ck_assert_int_eq (run_query ("-n", "1", "piano", "sonata", NULL), 0);
  ck_assert (strstr (out_text (), "/Music/playlist.m3u\n"));
}
END_TEST

static Suite *
query_suite (void)
{
  Suite *s = suite_create ("query");
  TCase *tc;

  tc = tcase_create ("matches");
  tcase_add_test (tc, test_insert_sorted);
  tcase_add_test (tc, test_insert_keeps_best_n);
  tcase_add_test (tc, test_insert_rejects_when_full);
  tcase_add_test (tc, test_insert_single_slot);
  tcase_add_test (tc, test_insert_negative_scores);
  suite_add_tcase (s, tc);

  tc = tcase_create ("program");
  tcase_add_checked_fixture (tc, setup, teardown);
  tcase_set_timeout (tc, 30);
  tcase_add_test (tc, test_query_ranks);
  tcase_add_test (tc, test_query_scores);
  tcase_add_test (tc, test_query_default_count);
  tcase_add_test (tc, test_query_joins_words_and_prefixes);
  tcase_add_test (tc, test_query_rejects_other_model);
  tcase_add_test (tc, test_query_skips_bad_lines);
  tcase_add_test (tc, test_query_newest_record_wins);
  tcase_add_test (tc, test_query_stale_record_does_not_hide_match);
  tcase_add_test (tc, test_query_paths_with_spaces);
  tcase_add_test (tc, test_query_empty_store);
  tcase_add_test (tc, test_query_missing_store);
  tcase_add_test (tc, test_query_bad_header);
  tcase_add_test (tc, test_query_dimension_mismatch);
  tcase_add_test (tc, test_query_model_unavailable);
  tcase_add_test (tc, test_query_embedding_fails);
  tcase_add_test (tc, test_query_removed_host_option);
  tcase_add_test (tc, test_query_missing_query);
  tcase_add_test (tc, test_query_bad_count);
  tcase_add_test (tc, test_query_options);
  tcase_add_test (tc, test_round_trip_with_indexer);
  suite_add_tcase (s, tc);

  return s;
}

int
main (void)
{
  SRunner *sr = srunner_create (query_suite ());
  int failed;

  srunner_run_all (sr, CK_NORMAL);
  failed = srunner_ntests_failed (sr);
  srunner_free (sr);
  return failed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
