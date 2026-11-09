/* test-common.c -- unit tests for common.c.
   Copyright (C) 2026 Martin Rixham

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   The file under test is included directly so that its static
   functions can be exercised.  */

#include "../src/common.c"

#include "testutil.h"

#include <check.h>
#include <unistd.h>

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

START_TEST (test_data_path)
{
  char *p;

  setenv ("HOME", "/home/someone", 1);
  p = data_path ("models/x/");
  ck_assert_str_eq (p, "/home/someone/.local/share/embed/models/x/");
  free (p);
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

START_TEST (test_hex_vector_round_trip)
{
  float in[] = { 0.5f, -0.25f, 1e-7f, 0.0f, -0.0f, 1.0f / 3, 3.4e38f,
                 1e-45f };
  size_t n = sizeof in / sizeof *in;
  float out[sizeof in / sizeof *in];
  char *text;
  size_t len;
  FILE *f = open_memstream (&text, &len);

  write_hex_vector (f, in, n);
  fclose (f);
  ck_assert_uint_eq (len, 8 * n);
  ck_assert (strncmp (text, "3f000000be80000033d6bf9500000000"
                      "80000000", 40) == 0);
  ck_assert (parse_hex_vector (text, out, n));
  /* Bit for bit, including the sign of zero.  */
  ck_assert (memcmp (in, out, sizeof in) == 0);
  free (text);
}
END_TEST

START_TEST (test_hex_vector_accepts_case_and_newline)
{
  float v[2];

  ck_assert (parse_hex_vector ("3F800000BE800000\n", v, 2));
  ck_assert_float_eq (v[0], 1.0f);
  ck_assert_float_eq (v[1], -0.25f);
  ck_assert (parse_hex_vector ("3f800000be800000", v, 2));
  ck_assert (parse_hex_vector ("\n", v, 0));
}
END_TEST

START_TEST (test_hex_vector_rejects)
{
  float v[2];

  ck_assert (!parse_hex_vector ("3f800000", v, 2));             /* short */
  ck_assert (!parse_hex_vector ("3f800000be8000", v, 2));       /* cut */
  ck_assert (!parse_hex_vector ("3f800000be80000000", v, 2));   /* long */
  ck_assert (!parse_hex_vector ("3f800000 be800000", v, 2));    /* space */
  ck_assert (!parse_hex_vector ("3f80000gbe800000", v, 2));     /* digit */
  ck_assert (!parse_hex_vector ("0x800000be800000", v, 2));
  ck_assert (!parse_hex_vector ("3f800000be800000\t", v, 2));
  ck_assert (!parse_hex_vector ("", v, 2));
}
END_TEST

START_TEST (test_q8_vector_round_trip)
{
  float in[] = { 0.5f, -0.25f, 0.001f, 0.0f, -0.5f, 1.0f / 3 };
  size_t n = sizeof in / sizeof *in;
  float out[sizeof in / sizeof *in];
  char *text;
  size_t len;
  FILE *f = open_memstream (&text, &len);

  write_q8_vector (f, in, n);
  fclose (f);
  ck_assert_uint_eq (len, 8 + 2 * n);
  /* The largest component is exactly 127 steps.  */
  ck_assert (strncmp (text + 8, "7fc0", 4) == 0);
  ck_assert (parse_q8_vector (text, out, n));
  for (size_t i = 0; i < n; i++)
    ck_assert_double_eq_tol (out[i], in[i], 0.5 / 127 * 0.5 + 1e-6);
  free (text);
}
END_TEST

START_TEST (test_q8_vector_zero_and_rejects)
{
  float zero[2] = { 0, 0 }, v[2];
  char *text;
  size_t len;
  FILE *f = open_memstream (&text, &len);

  write_q8_vector (f, zero, 2);
  fclose (f);
  ck_assert (parse_q8_vector (text, v, 2));
  ck_assert_float_eq (v[0], 0);
  ck_assert_float_eq (v[1], 0);
  ck_assert (parse_q8_vector ("3c0102047f81\n", v, 2));
  ck_assert_float_eq_tol (v[0], 1, 1e-3);
  ck_assert_float_eq_tol (v[1], -1, 1e-3);
  ck_assert (!parse_q8_vector ("3c0102047f", v, 2));
  ck_assert (!parse_q8_vector ("3c0102047f8100", v, 2));
  ck_assert (!parse_q8_vector ("3c0102047fzz", v, 2));
  ck_assert (!parse_q8_vector ("-", v, 2));
  free (text);
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



  tc = tcase_create ("store");
  tcase_add_test (tc, test_store_path);
  tcase_add_exit_test (tc, test_store_path_no_home, EXIT_FAILURE);
  tcase_add_exit_test (tc, test_store_path_empty_home, EXIT_FAILURE);
  tcase_add_test (tc, test_data_path);
  tcase_add_test (tc, test_parse_header_valid);
  tcase_add_test (tc, test_parse_header_invalid);
  tcase_add_test (tc, test_parse_header_model_too_long);
  tcase_add_test (tc, test_parse_positive);
  tcase_add_test (tc, test_hex_vector_round_trip);
  tcase_add_test (tc, test_hex_vector_accepts_case_and_newline);
  tcase_add_test (tc, test_hex_vector_rejects);
  tcase_add_test (tc, test_q8_vector_round_trip);
  tcase_add_test (tc, test_q8_vector_zero_and_rejects);
  tcase_add_test (tc, test_make_parent_dirs);
  tcase_add_exit_test (tc, test_make_parent_dirs_blocked, EXIT_FAILURE);
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
