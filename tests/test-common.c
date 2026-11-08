/* test-common.c -- unit tests for common.c.
   Copyright (C) 2026 Martin Rixham

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   The file under test is included directly so that its static
   functions can be exercised.  */

#include "../common.c"

#include "fake-http.h"
#include "testutil.h"

#include <check.h>
#include <unistd.h>

/* Return the JSON literal that buf_json_string makes of S, freshly
   allocated.  */
static char *
json_of (const char *s)
{
  struct buf b;

  buf_init (&b);
  buf_json_string (&b, s);
  return b.data;
}

static double
norm_of (const float *v, size_t n)
{
  double s = 0;

  for (size_t i = 0; i < n; i++)
    s += (double) v[i] * v[i];
  return sqrt (s);
}

/* Memory helpers.  */

START_TEST (test_xmalloc_zero)
{
  void *p = xmalloc (0);
  ck_assert_ptr_nonnull (p);
  free (p);
}
END_TEST

START_TEST (test_xrealloc_grows_and_keeps)
{
  char *p = xmalloc (4);
  memcpy (p, "abc", 4);
  p = xrealloc (p, 1 << 20);
  ck_assert_str_eq (p, "abc");
  free (p);
}
END_TEST

START_TEST (test_xmalloc_exhausted_exits)
{
  /* Larger than any address space, so it must fail.  */
  xmalloc ((size_t) -1 / 2);
}
END_TEST

START_TEST (test_xstrdup)
{
  char *s = xstrdup ("hello");
  ck_assert_str_eq (s, "hello");
  free (s);
  s = xstrdup ("");
  ck_assert_str_eq (s, "");
  free (s);
}
END_TEST

/* Buffers.  */

START_TEST (test_buf_init_empty)
{
  struct buf b;

  buf_init (&b);
  ck_assert_uint_eq (b.len, 0);
  ck_assert_str_eq (b.data, "");
  buf_free (&b);
  ck_assert_ptr_null (b.data);
  ck_assert_uint_eq (b.len, 0);
}
END_TEST

START_TEST (test_buf_append_and_puts)
{
  struct buf b;

  buf_init (&b);
  buf_puts (&b, "foo");
  buf_append (&b, "barbaz", 3);
  buf_puts (&b, "");
  ck_assert_str_eq (b.data, "foobar");
  ck_assert_uint_eq (b.len, 6);
  buf_free (&b);
}
END_TEST

START_TEST (test_buf_grows_past_capacity)
{
  struct buf b;
  char expect[10001];

  buf_init (&b);
  for (int i = 0; i < 10000; i++)
    {
      buf_append (&b, "x", 1);
      expect[i] = 'x';
    }
  expect[10000] = '\0';
  ck_assert_uint_eq (b.len, 10000);
  ck_assert_uint_ge (b.cap, 10001);
  ck_assert_str_eq (b.data, expect);
  buf_free (&b);
}
END_TEST

START_TEST (test_buf_one_big_append)
{
  struct buf b;
  size_t n = 1000000;
  char *big = xmalloc (n);

  memset (big, 'y', n);
  buf_init (&b);
  buf_append (&b, big, n);
  ck_assert_uint_eq (b.len, n);
  ck_assert_int_eq (b.data[n], '\0');
  ck_assert_int_eq (b.data[n - 1], 'y');
  buf_free (&b);
  free (big);
}
END_TEST

START_TEST (test_buf_embedded_nul)
{
  struct buf b;

  buf_init (&b);
  buf_append (&b, "a\0b", 3);
  ck_assert_uint_eq (b.len, 3);
  ck_assert_int_eq (memcmp (b.data, "a\0b", 4), 0);
  buf_free (&b);
}
END_TEST

/* UTF-8 validation.  */

static size_t
u8 (const char *s, size_t n)
{
  return utf8_len ((const unsigned char *) s, n);
}

START_TEST (test_utf8_valid)
{
  ck_assert_uint_eq (u8 ("a", 1), 1);
  ck_assert_uint_eq (u8 ("\x7f", 1), 1);
  ck_assert_uint_eq (u8 ("\xc2\xa3", 2), 2);           /* U+00A3 */
  ck_assert_uint_eq (u8 ("\xdf\xbf", 2), 2);           /* U+07FF */
  ck_assert_uint_eq (u8 ("\xe0\xa0\x80", 3), 3);       /* U+0800 */
  ck_assert_uint_eq (u8 ("\xe2\x82\xac", 3), 3);       /* U+20AC */
  ck_assert_uint_eq (u8 ("\xef\xbf\xbd", 3), 3);       /* U+FFFD */
  ck_assert_uint_eq (u8 ("\xf0\x90\x80\x80", 4), 4);   /* U+10000 */
  ck_assert_uint_eq (u8 ("\xf0\x9f\x98\x80", 4), 4);   /* U+1F600 */
  ck_assert_uint_eq (u8 ("\xf4\x8f\xbf\xbf", 4), 4);   /* U+10FFFF */
}
END_TEST

START_TEST (test_utf8_bad_lead_bytes)
{
  ck_assert_uint_eq (u8 ("\x80", 1), 0);
  ck_assert_uint_eq (u8 ("\xbf", 1), 0);
  ck_assert_uint_eq (u8 ("\xc0\x80", 2), 0);
  ck_assert_uint_eq (u8 ("\xc1\xbf", 2), 0);
  ck_assert_uint_eq (u8 ("\xf5\x80\x80\x80", 4), 0);
  ck_assert_uint_eq (u8 ("\xf8\x80\x80\x80", 4), 0);
  ck_assert_uint_eq (u8 ("\xff", 1), 0);
}
END_TEST

START_TEST (test_utf8_truncated_and_bad_continuation)
{
  ck_assert_uint_eq (u8 ("\xc2", 1), 0);
  ck_assert_uint_eq (u8 ("\xe2\x82", 2), 0);
  ck_assert_uint_eq (u8 ("\xf0\x9f\x98", 3), 0);
  ck_assert_uint_eq (u8 ("\xc2" "a", 2), 0);
  ck_assert_uint_eq (u8 ("\xe2\x82" "a", 3), 0);
  ck_assert_uint_eq (u8 ("\xf0\x9f\x98\xc0", 4), 0);
}
END_TEST

START_TEST (test_utf8_overlong_and_out_of_range)
{
  /* Overlong encodings of U+0000 and U+FFFF.  */
  ck_assert_uint_eq (u8 ("\xe0\x80\x80", 3), 0);
  ck_assert_uint_eq (u8 ("\xe0\x9f\xbf", 3), 0);
  ck_assert_uint_eq (u8 ("\xf0\x80\x80\x80", 4), 0);
  ck_assert_uint_eq (u8 ("\xf0\x8f\xbf\xbf", 4), 0);
  /* UTF-16 surrogates U+D800 and U+DFFF.  */
  ck_assert_uint_eq (u8 ("\xed\xa0\x80", 3), 0);
  ck_assert_uint_eq (u8 ("\xed\xbf\xbf", 3), 0);
  /* Just past U+10FFFF.  */
  ck_assert_uint_eq (u8 ("\xf4\x90\x80\x80", 4), 0);
}
END_TEST

/* JSON string escaping.  */

START_TEST (test_json_plain)
{
  char *j = json_of ("hello world");
  ck_assert_str_eq (j, "\"hello world\"");
  free (j);
  j = json_of ("");
  ck_assert_str_eq (j, "\"\"");
  free (j);
}
END_TEST

START_TEST (test_json_escapes)
{
  char *j = json_of ("say \"hi\" \\ back\nslash");
  ck_assert_str_eq (j, "\"say \\\"hi\\\" \\\\ back\\nslash\"");
  free (j);
}
END_TEST

START_TEST (test_json_control_characters)
{
  char *j = json_of ("a\tb\rc\x01" "d\x1f" "e\x7f" "f\x1b[0m");
  ck_assert_str_eq (j, "\"a b c d e f [0m\"");
  free (j);
}
END_TEST

START_TEST (test_json_keeps_valid_utf8)
{
  char *j = json_of ("£49.40 €5 \xf0\x9f\x98\x80 日本");
  ck_assert_str_eq (j, "\"£49.40 €5 \xf0\x9f\x98\x80 日本\"");
  free (j);
}
END_TEST

START_TEST (test_json_replaces_invalid_utf8)
{
  char *j = json_of ("a\xff" "b\xc2" "c\xe2\x82" "d\xed\xa0\x80" "e");
  /* Each bad byte becomes one '?'.  */
  ck_assert_str_eq (j, "\"a?b?c??d???e\"");
  free (j);
  /* A sequence cut short at the end of the string.  */
  j = json_of ("x\xe2\x82");
  ck_assert_str_eq (j, "\"x??\"");
  free (j);
}
END_TEST

START_TEST (test_json_output_always_valid)
{
  /* Every byte value, in every position, gives a literal with no raw
     control characters and balanced quotes.  */
  char s[256];
  char *j;

  for (int i = 1; i < 256; i++)
    s[i - 1] = (char) i;
  s[255] = '\0';
  j = json_of (s);
  ck_assert_int_eq (j[0], '"');
  ck_assert_int_eq (j[strlen (j) - 1], '"');
  for (const unsigned char *p = (unsigned char *) j; *p; p++)
    ck_assert_int_ge (*p, 0x20);
  free (j);
}
END_TEST

/* Model prefixes.  */

START_TEST (test_prefixes)
{
  ck_assert_str_eq (document_prefix ("nomic-embed-text"),
                    "search_document: ");
  ck_assert_str_eq (query_prefix ("nomic-embed-text:latest"),
                    "search_query: ");
  ck_assert_str_eq (document_prefix ("qwen3-embedding:0.6b"), "");
  ck_assert_ptr_nonnull (strstr (query_prefix ("qwen3-embedding:0.6b"),
                                 "Query: "));
  ck_assert_str_eq (document_prefix ("all-minilm"), "");
  ck_assert_str_eq (query_prefix ("all-minilm"), "");
}
END_TEST

/* Store path.  */

START_TEST (test_store_path)
{
  char *p;

  setenv ("HOME", "/home/someone", 1);
  p = store_path ();
  ck_assert_str_eq (p, "/home/someone/.local/share/embed/embeddings.txt");
  free (p);
}
END_TEST

START_TEST (test_store_path_no_home)
{
  unsetenv ("HOME");
  store_path ();
}
END_TEST

START_TEST (test_store_path_empty_home)
{
  setenv ("HOME", "", 1);
  store_path ();
}
END_TEST

/* Store header.  */

START_TEST (test_parse_header_valid)
{
  char model[64];
  size_t dim = 0;

  ck_assert (parse_header ("# embed-index model=nomic-embed-text dim=768\n",
                           model, sizeof model, &dim));
  ck_assert_str_eq (model, "nomic-embed-text");
  ck_assert_uint_eq (dim, 768);

  ck_assert (parse_header ("# embed-index model=qwen3-embedding:0.6b "
                           "dim=1024", model, sizeof model, &dim));
  ck_assert_str_eq (model, "qwen3-embedding:0.6b");
  ck_assert_uint_eq (dim, 1024);
}
END_TEST

START_TEST (test_parse_header_invalid)
{
  char model[16];
  size_t dim;

  ck_assert (!parse_header ("embed-index model=m dim=3\n", model,
                            sizeof model, &dim));
  ck_assert (!parse_header ("# embed-index dim=3\n", model, sizeof model,
                            &dim));
  ck_assert (!parse_header ("# embed-index model=m\n", model, sizeof model,
                            &dim));
  ck_assert (!parse_header ("# embed-index model=m dim=0\n", model,
                            sizeof model, &dim));
  ck_assert (!parse_header ("# embed-index model=m dim=x\n", model,
                            sizeof model, &dim));
  ck_assert (!parse_header ("# embed-index model= dim=3\n", model,
                            sizeof model, &dim));
  ck_assert (!parse_header ("", model, sizeof model, &dim));
}
END_TEST

START_TEST (test_parse_header_model_too_long)
{
  char model[8];
  size_t dim;

  ck_assert (parse_header ("# model=1234567 dim=3", model, sizeof model,
                           &dim));
  ck_assert_str_eq (model, "1234567");
  ck_assert (!parse_header ("# model=12345678 dim=3", model, sizeof model,
                            &dim));
}
END_TEST

/* Directory creation.  */

START_TEST (test_make_parent_dirs)
{
  char *dir = tmpdir_new ();
  char *file = path_join (dir, "a/b/c/file.txt");
  char *c = path_join (dir, "a/b/c");
  struct stat st;

  make_parent_dirs (file);
  ck_assert_int_eq (stat (c, &st), 0);
  ck_assert (S_ISDIR (st.st_mode));
  ck_assert_int_ne (stat (file, &st), 0);
  /* Doing it again is harmless.  */
  make_parent_dirs (file);
  tmpdir_remove (dir);
}
END_TEST

START_TEST (test_make_parent_dirs_blocked)
{
  char *dir = tmpdir_new ();
  char *blocker = path_join (dir, "a");
  char *file = path_join (dir, "a/b/file.txt");

  write_file (blocker, "not a directory", (size_t) -1);
  make_parent_dirs (file);
}
END_TEST

/* Embedding requests, against a fake server.  */

struct fixture
{
  char *dir;
  char *log;
  struct fake fake;
  struct embedder emb;
};

static void
fixture_init (struct fixture *fx, const char *model)
{
  fx->dir = tmpdir_new ();
  fx->log = path_join (fx->dir, "requests.log");
  fake_start (&fx->fake, fx->log);
  embedder_init (&fx->emb, fx->fake.url, model);
}

static void
fixture_free (struct fixture *fx)
{
  embedder_free (&fx->emb);
  fake_stop (&fx->fake);
  tmpdir_remove (fx->dir);
}

START_TEST (test_embedder_init_url)
{
  struct embedder e;

  embedder_init (&e, "http://example:1", "m");
  ck_assert_str_eq (e.url, "http://example:1/api/embed");
  ck_assert_str_eq (e.model, "m");
  embedder_free (&e);
}
END_TEST

START_TEST (test_embed_success_normalized)
{
  struct fixture fx;
  float *v;
  size_t dim = 0;

  fixture_init (&fx, "fake-model");
  ck_assert (embed_text (&fx.emb, "sourdough sourdough invoice", 5, &v,
                         &dim));
  ck_assert_uint_eq (dim, FAKE_DIM);
  ck_assert_double_eq_tol (norm_of (v, dim), 1.0, 1e-5);
  /* 2.05 : 1.05 : 0.05 ... before normalization.  */
  ck_assert_double_eq_tol (v[0] / v[1], 2.05 / 1.05, 1e-4);
  ck_assert_double_eq_tol (v[2] / v[1], 0.05 / 1.05, 1e-4);
  free (v);
  fixture_free (&fx);
}
END_TEST

START_TEST (test_embed_request_body)
{
  struct fixture fx;
  float *v;
  size_t dim;
  char *log;

  fixture_init (&fx, "my-model");
  ck_assert (embed_text (&fx.emb, "line one\n\"quoted\" £", 5, &v, &dim));
  free (v);
  log = read_file (fx.log);
  ck_assert_ptr_nonnull (log);
  ck_assert_ptr_nonnull (strstr (log, "\"model\":\"my-model\""));
  ck_assert_ptr_nonnull (strstr (log, "\"truncate\":true"));
  ck_assert_ptr_nonnull (strstr (log, "\"keep_alive\":\"10m\""));
  ck_assert_ptr_nonnull (strstr (log, "\"input\":\"line one\\n\\\"quoted\\\" "
                                 "£\""));
  free (log);
  fixture_free (&fx);
}
END_TEST

START_TEST (test_embed_reuses_connection_handle)
{
  struct fixture fx;
  float *v;
  size_t dim;

  fixture_init (&fx, "m");
  for (int i = 0; i < 20; i++)
    {
      ck_assert (embed_text (&fx.emb, "omega", 5, &v, &dim));
      ck_assert_float_gt (v[7], 0.9);
      free (v);
    }
  ck_assert_uint_eq (count_substr (read_file (fx.log), "----"), 20);
  fixture_free (&fx);
}
END_TEST

START_TEST (test_embed_large_input)
{
  struct fixture fx;
  size_t n = 200000;
  char *text = xmalloc (n + 1);
  float *v;
  size_t dim;

  memset (text, 'z', n);
  memcpy (text + n - 6, "sunset", 6);
  text[n] = '\0';
  fixture_init (&fx, "m");
  ck_assert (embed_text (&fx.emb, text, 5, &v, &dim));
  ck_assert_float_gt (v[4], 0.9);
  free (v);
  free (text);
  fixture_free (&fx);
}
END_TEST

START_TEST (test_embed_tolerates_white_space)
{
  struct fixture fx;
  float *v = NULL;
  size_t dim = 0;

  fixture_init (&fx, "m");
  ck_assert (embed_text (&fx.emb, "FAKE_SPACES alpha", 5, &v, &dim));
  ck_assert_uint_eq (dim, FAKE_DIM);
  ck_assert_float_gt (v[6], 0.9);
  free (v);
  fixture_free (&fx);
}
END_TEST

START_TEST (test_embed_zero_vector)
{
  struct fixture fx;
  float *v;
  size_t dim;

  fixture_init (&fx, "m");
  ck_assert (embed_text (&fx.emb, "FAKE_ZERO", 5, &v, &dim));
  ck_assert_uint_eq (dim, FAKE_DIM);
  for (size_t i = 0; i < dim; i++)
    ck_assert (v[i] == 0 && !isnan (v[i]));
  free (v);
  fixture_free (&fx);
}
END_TEST

START_TEST (test_embed_other_dimension)
{
  struct fixture fx;
  float *v;
  size_t dim;

  fixture_init (&fx, "m");
  ck_assert (embed_text (&fx.emb, "FAKE_DIM3", 5, &v, &dim));
  ck_assert_uint_eq (dim, 3);
  free (v);
  fixture_free (&fx);
}
END_TEST

/* Check that embedding TEXT fails, leaves its outputs alone, and
   reports the failure on standard error containing EXPECT.  */
static void
check_embed_fails (const char *text, const char *expect)
{
  struct fixture fx;
  float *v = (float *) &v;
  size_t dim = 12345;
  char *err_path, *err;

  fixture_init (&fx, "m");
  err_path = path_join (fx.dir, "stderr");
  ck_assert_ptr_nonnull (freopen (err_path, "w", stderr));
  ck_assert (!embed_text (&fx.emb, text, 5, &v, &dim));
  fflush (stderr);
  ck_assert_ptr_eq (v, (float *) &v);
  ck_assert_uint_eq (dim, 12345);
  err = read_file (err_path);
  ck_assert_msg (strstr (err, expect), "stderr was: %s", err);
  free (err);
  fixture_free (&fx);
}

START_TEST (test_embed_http_error)
{
  check_embed_fails ("FAKE_500", "returned 500");
}
END_TEST

START_TEST (test_embed_too_long)
{
  check_embed_fails ("FAKE_TOOLONG", "context length");
}
END_TEST

START_TEST (test_embed_malformed)
{
  check_embed_fails ("FAKE_MALFORMED", "malformed");
}
END_TEST

START_TEST (test_embed_empty_vector)
{
  check_embed_fails ("FAKE_EMPTY", "malformed");
}
END_TEST

START_TEST (test_embed_missing_member)
{
  check_embed_fails ("FAKE_NOVEC", "returned 200");
}
END_TEST

START_TEST (test_embed_bad_separator)
{
  check_embed_fails ("FAKE_BADSEP", "malformed");
}
END_TEST

START_TEST (test_find_vector)
{
  ck_assert_str_eq (find_vector ("{\"embeddings\":[[1,2]]}"), "1,2]]}");
  ck_assert_str_eq (find_vector ("{\"embeddings\" :\n [\t[ 1]]}"), " 1]]}");
  ck_assert_ptr_null (find_vector ("{\"embedding\":[[1]]}"));
  ck_assert_ptr_null (find_vector ("{\"embeddings\" [[1]]}"));
  ck_assert_ptr_null (find_vector ("{\"embeddings\":{[1]]}"));
  ck_assert_ptr_null (find_vector ("{\"embeddings\":[1,2]}"));
  ck_assert_ptr_null (find_vector ("{\"embeddings\":"));
  ck_assert_ptr_null (find_vector (""));
}
END_TEST

START_TEST (test_parse_positive)
{
  long v = -7;

  ck_assert (parse_positive ("1", 10, &v) && v == 1);
  ck_assert (parse_positive ("10", 10, &v) && v == 10);
  ck_assert (parse_positive ("007", 10, &v) && v == 7);
  v = -7;
  ck_assert (!parse_positive ("11", 10, &v));
  ck_assert (!parse_positive ("0", 10, &v));
  ck_assert (!parse_positive ("-1", 10, &v));
  ck_assert (!parse_positive ("+1", 10, &v));
  ck_assert (!parse_positive (" 1", 10, &v));
  ck_assert (!parse_positive ("1 ", 10, &v));
  ck_assert (!parse_positive ("1.5", 10, &v));
  ck_assert (!parse_positive ("0x5", 10, &v));
  ck_assert (!parse_positive ("", 10, &v));
  ck_assert (!parse_positive ("99999999999999999999999", 10, &v));
  ck_assert_int_eq (v, -7);
}
END_TEST

START_TEST (test_embed_quiet)
{
  struct fixture fx;
  float *v;
  size_t dim;
  char *err_path;

  fixture_init (&fx, "m");
  err_path = path_join (fx.dir, "stderr");
  ck_assert_ptr_nonnull (freopen (err_path, "w", stderr));
  embed_report_errors = false;
  ck_assert (!embed_text (&fx.emb, "FAKE_500", 5, &v, &dim));
  ck_assert (!embed_text (&fx.emb, "FAKE_MALFORMED", 5, &v, &dim));
  fflush (stderr);
  ck_assert_str_eq (read_file (err_path), "");
  fixture_free (&fx);
}
END_TEST

START_TEST (test_embed_timeout)
{
  struct fixture fx;
  float *v;
  size_t dim;
  double start;

  fixture_init (&fx, "m");
  embed_report_errors = false;
  start = now ();
  ck_assert (!embed_text (&fx.emb, "FAKE_SLOW", 1, &v, &dim));
  ck_assert_double_lt (now () - start, 2.5);
  /* The handle still works afterwards.  */
  ck_assert (embed_text (&fx.emb, "alpha", 5, &v, &dim));
  free (v);
  fixture_free (&fx);
}
END_TEST

START_TEST (test_embed_connection_refused)
{
  struct embedder e;
  float *v;
  size_t dim;

  embed_report_errors = false;
  embedder_init (&e, dead_url (), "m");
  ck_assert (!embed_text (&e, "alpha", 2, &v, &dim));
  embedder_free (&e);
}
END_TEST

static Suite *
common_suite (void)
{
  Suite *s = suite_create ("common");
  TCase *tc;

  tc = tcase_create ("memory");
  tcase_add_test (tc, test_xmalloc_zero);
  tcase_add_test (tc, test_xrealloc_grows_and_keeps);
  tcase_add_exit_test (tc, test_xmalloc_exhausted_exits, EXIT_FAILURE);
  tcase_add_test (tc, test_xstrdup);
  suite_add_tcase (s, tc);

  tc = tcase_create ("buf");
  tcase_add_test (tc, test_buf_init_empty);
  tcase_add_test (tc, test_buf_append_and_puts);
  tcase_add_test (tc, test_buf_grows_past_capacity);
  tcase_add_test (tc, test_buf_one_big_append);
  tcase_add_test (tc, test_buf_embedded_nul);
  suite_add_tcase (s, tc);

  tc = tcase_create ("utf8");
  tcase_add_test (tc, test_utf8_valid);
  tcase_add_test (tc, test_utf8_bad_lead_bytes);
  tcase_add_test (tc, test_utf8_truncated_and_bad_continuation);
  tcase_add_test (tc, test_utf8_overlong_and_out_of_range);
  suite_add_tcase (s, tc);

  tc = tcase_create ("json");
  tcase_add_test (tc, test_json_plain);
  tcase_add_test (tc, test_json_escapes);
  tcase_add_test (tc, test_json_control_characters);
  tcase_add_test (tc, test_json_keeps_valid_utf8);
  tcase_add_test (tc, test_json_replaces_invalid_utf8);
  tcase_add_test (tc, test_json_output_always_valid);
  suite_add_tcase (s, tc);

  tc = tcase_create ("store");
  tcase_add_test (tc, test_prefixes);
  tcase_add_test (tc, test_store_path);
  tcase_add_exit_test (tc, test_store_path_no_home, EXIT_FAILURE);
  tcase_add_exit_test (tc, test_store_path_empty_home, EXIT_FAILURE);
  tcase_add_test (tc, test_parse_header_valid);
  tcase_add_test (tc, test_parse_header_invalid);
  tcase_add_test (tc, test_parse_header_model_too_long);
  tcase_add_test (tc, test_find_vector);
  tcase_add_test (tc, test_parse_positive);
  tcase_add_test (tc, test_make_parent_dirs);
  tcase_add_exit_test (tc, test_make_parent_dirs_blocked, EXIT_FAILURE);
  suite_add_tcase (s, tc);

  tc = tcase_create ("embed");
  tcase_set_timeout (tc, 20);
  tcase_add_test (tc, test_embedder_init_url);
  tcase_add_test (tc, test_embed_success_normalized);
  tcase_add_test (tc, test_embed_request_body);
  tcase_add_test (tc, test_embed_reuses_connection_handle);
  tcase_add_test (tc, test_embed_large_input);
  tcase_add_test (tc, test_embed_tolerates_white_space);
  tcase_add_test (tc, test_embed_zero_vector);
  tcase_add_test (tc, test_embed_other_dimension);
  tcase_add_test (tc, test_embed_http_error);
  tcase_add_test (tc, test_embed_too_long);
  tcase_add_test (tc, test_embed_malformed);
  tcase_add_test (tc, test_embed_empty_vector);
  tcase_add_test (tc, test_embed_missing_member);
  tcase_add_test (tc, test_embed_bad_separator);
  tcase_add_test (tc, test_embed_quiet);
  tcase_add_test (tc, test_embed_timeout);
  tcase_add_test (tc, test_embed_connection_refused);
  suite_add_tcase (s, tc);

  return s;
}

int
main (void)
{
  SRunner *sr = srunner_create (common_suite ());
  int failed;

  srunner_run_all (sr, CK_NORMAL);
  failed = srunner_ntests_failed (sr);
  srunner_free (sr);
  return failed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
