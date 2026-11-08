/* test-fetch.c -- unit tests for fetch.c.
   Copyright (C) 2026 Martin Rixham

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   Downloads come from a fake file server on the loopback interface.  */

/* Give up on a stalled download quickly.  */
#define STALL_TIMEOUT 2

#include "../fetch.c"

#include "testutil.h"

#include <check.h>
#include <sys/stat.h>

/* SHA-256 of the 100000 byte file the fixture serves.  */
static char good_hash[65];

struct fixture
{
  char *dir;                    /* Served by the fake server.  */
  char *out;                    /* Where downloads go.  */
  char *err;
  struct fake fake;
};

static struct fixture fx;

static void
setup (void)
{
  char *data = xmalloc (100000), *path;
  unsigned char digest[32];

  for (size_t i = 0; i < 100000; i++)
    data[i] = (char) (i * 7 + i / 251);
  fx.dir = tmpdir_new ();
  fx.out = tmpdir_new ();
  fx.err = path_join (fx.out, "stderr");
  path = path_join (fx.dir, "model.bin");
  write_file (path, data, 100000);
  ck_assert (EVP_Digest (data, 100000, digest, NULL, EVP_sha256 (), NULL));
  for (int i = 0; i < 32; i++)
    snprintf (good_hash + 2 * i, 3, "%02x", digest[i]);
  free (path);
  free (data);
  fake_start (&fx.fake, fx.dir);
}

static void
teardown (void)
{
  fake_stop (&fx.fake);
  tmpdir_remove (fx.dir);
  tmpdir_remove (fx.out);
}

/* Fetch NAME from the fake server into OUT/sub/dir/model.bin, checking
   it against HASH, with standard error going to fx.err.  */
static bool
fetch (const char *name, const char *hash)
{
  char *url, *path = path_join (fx.out, "sub/dir/model.bin");
  bool ok;

  ck_assert_int_ge (asprintf (&url, "%s/%s", fx.fake.url, name), 0);
  fflush (stderr);
  ck_assert_ptr_nonnull (freopen (fx.err, "w", stderr));
  ok = fetch_file (url, path, hash);
  fflush (stderr);
  free (url);
  free (path);
  return ok;
}

static char *
err_text (void)
{
  char *s = read_file (fx.err);
  return s ? s : xstrdup ("");
}

static bool
exists (const char *name)
{
  char *path = path_join (fx.out, name);
  bool there = access (path, F_OK) == 0;

  free (path);
  return there;
}

/* Check that a failed fetch left nothing behind.  */
static void
check_nothing_left (void)
{
  ck_assert (!exists ("sub/dir/model.bin"));
  ck_assert (!exists ("sub/dir/model.bin.part"));
}

START_TEST (test_fetch_success)
{
  char *path = path_join (fx.out, "sub/dir/model.bin");
  struct stat st;

  ck_assert (fetch ("model.bin", good_hash));
  ck_assert_int_eq (stat (path, &st), 0);
  ck_assert_int_eq (st.st_size, 100000);
  ck_assert (!exists ("sub/dir/model.bin.part"));
  /* Not a terminal, so no progress.  */
  ck_assert_str_eq (err_text (), "");
}
END_TEST

START_TEST (test_fetch_follows_redirect)
{
  ck_assert (fetch ("redirect/model.bin", good_hash));
  ck_assert (exists ("sub/dir/model.bin"));
}
END_TEST

START_TEST (test_fetch_existing_file_kept)
{
  char *path = path_join (fx.out, "sub/dir/model.bin");
  char *log = path_join (fx.dir, "requests.log");

  write_file (path, "already here", (size_t) -1);
  ck_assert (fetch ("model.bin", good_hash));
  ck_assert_str_eq (read_file (path), "already here");
  /* Without asking the server.  */
  ck_assert_ptr_null (read_file (log));
}
END_TEST

START_TEST (test_fetch_bad_hash)
{
  ck_assert (!fetch ("model.bin", "0123456789abcdef0123456789abcdef"
                     "0123456789abcdef0123456789abcdef"));
  ck_assert_ptr_nonnull (strstr (err_text (), "download is corrupt"));
  ck_assert_ptr_nonnull (strstr (err_text (), good_hash));
  check_nothing_left ();
}
END_TEST

START_TEST (test_fetch_not_found)
{
  ck_assert (!fetch ("missing.bin", good_hash));
  ck_assert_ptr_nonnull (strstr (err_text (), "cannot download"));
  ck_assert_ptr_nonnull (strstr (err_text (), "404"));
  check_nothing_left ();
}
END_TEST

START_TEST (test_fetch_truncated)
{
  ck_assert (!fetch ("truncated/model.bin", good_hash));
  ck_assert_ptr_nonnull (strstr (err_text (), "cannot download"));
  check_nothing_left ();
}
END_TEST

START_TEST (test_fetch_stalled)
{
  double start = now ();

  ck_assert (!fetch ("stall/model.bin", good_hash));
  ck_assert_double_lt (now () - start, 10);
  ck_assert_ptr_nonnull (strstr (err_text (), "cannot download"));
  check_nothing_left ();
}
END_TEST

START_TEST (test_fetch_server_down)
{
  char *url, *path = path_join (fx.out, "model.bin");

  ck_assert_int_ge (asprintf (&url, "%s/model.bin", dead_url ()), 0);
  ck_assert_ptr_nonnull (freopen (fx.err, "w", stderr));
  ck_assert (!fetch_file (url, path, good_hash));
  fflush (stderr);
  ck_assert_ptr_nonnull (strstr (err_text (), "cannot download"));
  ck_assert (!exists ("model.bin.part"));
}
END_TEST

START_TEST (test_fetch_unwritable)
{
  char *dir = path_join (fx.out, "sub/dir");
  char *path = path_join (dir, "model.bin");

  make_parent_dirs (path);
  chmod (dir, 0555);
  if (access (dir, W_OK) == 0)
    return;                     /* Running as root.  */
  ck_assert (!fetch ("model.bin", good_hash));
  ck_assert_ptr_nonnull (strstr (err_text (), "model.bin.part"));
  chmod (dir, 0755);
}
END_TEST

int
main (void)
{
  Suite *s = suite_create ("fetch");
  TCase *tc = tcase_create ("fetch");
  SRunner *sr;
  int failed;

  tcase_add_checked_fixture (tc, setup, teardown);
  tcase_set_timeout (tc, 30);
  tcase_add_test (tc, test_fetch_success);
  tcase_add_test (tc, test_fetch_follows_redirect);
  tcase_add_test (tc, test_fetch_existing_file_kept);
  tcase_add_test (tc, test_fetch_bad_hash);
  tcase_add_test (tc, test_fetch_not_found);
  tcase_add_test (tc, test_fetch_truncated);
  tcase_add_test (tc, test_fetch_stalled);
  tcase_add_test (tc, test_fetch_server_down);
  tcase_add_test (tc, test_fetch_unwritable);
  suite_add_tcase (s, tc);
  sr = srunner_create (s);
  srunner_run_all (sr, CK_NORMAL);
  failed = srunner_ntests_failed (sr);
  srunner_free (sr);
  return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
