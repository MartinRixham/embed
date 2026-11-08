/* test-tokenizer.c -- unit tests for tokenizer.c.
   Copyright (C) 2026 Martin Rixham

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   The reference cases in data/tokenizer-cases.txt come from the
   Hugging Face tokenizers library, with nomic-embed-text's
   tokenizer.json; see tools/gen-tokenizer-cases.py.  */

#include "../tokenizer.c"

#include "testutil.h"

#include <check.h>
#include <errno.h>

#ifndef TEST_DATA
# error "TEST_DATA must name the test data directory"
#endif

static struct tokenizer real;

static void
load_real (void)
{
  ck_assert (tokenizer_load (&real, TEST_DATA "/vocab.txt"));
}

static void
free_real (void)
{
  tokenizer_free (&real);
}

/* Tokenize TEXT with T and return the ids as a string like "101 7592
   102", freshly allocated.  */
static char *
ids_of (const struct tokenizer *t, const char *text, size_t max)
{
  int64_t *ids = xmalloc (max * sizeof *ids);
  size_t n = tokenize (t, text, ids, max);
  struct buf b;

  buf_init (&b);
  for (size_t i = 0; i < n; i++)
    {
      char num[32];
      snprintf (num, sizeof num, i ? " %ld" : "%ld", (long) ids[i]);
      buf_puts (&b, num);
    }
  free (ids);
  return b.data;
}

/* Tokenize TEXT with T and return the tokens as a string like "[CLS]
   un ##aff ##able [SEP]", freshly allocated.  */
static char *
tokens_of (const struct tokenizer *t, const char *text)
{
  int64_t ids[256];
  size_t n = tokenize (t, text, ids, 256);
  struct buf b;

  buf_init (&b);
  for (size_t i = 0; i < n; i++)
    {
      if (i)
        buf_puts (&b, " ");
      buf_puts (&b, t->words[ids[i]]);
    }
  return b.data;
}

START_TEST (test_special_ids)
{
  ck_assert_uint_eq (real.n_words, 30522);
  ck_assert_int_eq (real.unk, 100);
  ck_assert_int_eq (real.cls, 101);
  ck_assert_int_eq (real.sep, 102);
  ck_assert_int_eq (tokenizer_lookup (&real, "[PAD]", 5), 0);
  ck_assert_int_eq (tokenizer_lookup (&real, "the", 3), 1996);
  ck_assert_int_eq (tokenizer_lookup (&real, "##ing", 5), 2075);
  ck_assert_int_eq (tokenizer_lookup (&real, "zzzzzz", 6), -1);
  /* A prefix of a token is not that token.  */
  ck_assert_int_eq (tokenizer_lookup (&real, "th", 2), 16215);
  ck_assert_int_eq (tokenizer_lookup (&real, "thereby", 3), 1996);
}
END_TEST

START_TEST (test_reference_cases)
{
  FILE *f = fopen (TEST_DATA "/tokenizer-cases.txt", "r");
  char *line = NULL;
  size_t size = 0, n_cases = 0, n_failed = 0;
  ssize_t len;

  ck_assert_ptr_nonnull (f);
  while ((len = getline (&line, &size, f)) > 0)
    {
      char *tab = strchr (line, '\t');
      size_t hex_len;
      char *text, *got;

      ck_assert_ptr_nonnull (tab);
      line[len - 1] = '\0';
      hex_len = tab - line;
      text = xmalloc (hex_len / 2 + 1);
      for (size_t i = 0; i < hex_len / 2; i++)
        sscanf (line + 2 * i, "%2hhx", (unsigned char *) &text[i]);
      text[hex_len / 2] = '\0';
      got = ids_of (&real, text, strlen (text) + 2);
      if (strcmp (got, tab + 1) != 0)
        {
          if (n_failed < 5)
            fprintf (stderr, "case %zu: text %.*s\n  want %s\n  got  %s\n",
                     n_cases + 1, (int) hex_len, line, tab + 1, got);
          n_failed++;
        }
      free (got);
      free (text);
      n_cases++;
    }
  free (line);
  fclose (f);
  ck_assert_uint_gt (n_cases, 600);
  ck_assert_msg (n_failed == 0, "%zu of %zu cases differ", n_failed,
                 n_cases);
}
END_TEST

START_TEST (test_basic_words)
{
  ck_assert_str_eq (tokens_of (&real, "Hello, World!"),
                    "[CLS] hello , world ! [SEP]");
  ck_assert_str_eq (tokens_of (&real, "unaffable"),
                    "[CLS] una ##ffa ##ble [SEP]");
  ck_assert_str_eq (tokens_of (&real, "  \t\n "), "[CLS] [SEP]");
  ck_assert_str_eq (tokens_of (&real, ""), "[CLS] [SEP]");
}
END_TEST

START_TEST (test_accents_and_case)
{
  ck_assert_str_eq (tokens_of (&real, "Café NAÏVE"),
                    "[CLS] cafe naive [SEP]");
  /* Decomposed and composed forms are the same.  */
  ck_assert_str_eq (ids_of (&real, "e\xcc\x81", 8),
                    ids_of (&real, "\xc3\xa9", 8));
}
END_TEST

START_TEST (test_chinese_split)
{
  ck_assert_str_eq (tokens_of (&real, "ab日本cd"),
                    "[CLS] ab 日 本 cd [SEP]");
}
END_TEST

START_TEST (test_special_tokens_in_text)
{
  ck_assert_str_eq (tokens_of (&real, "a[SEP]b [MASK]"),
                    "[CLS] a [SEP] b [MASK] [SEP]");
  ck_assert_str_eq (tokens_of (&real, "[sep]"),
                    "[CLS] [ sep ] [SEP]");
}
END_TEST

START_TEST (test_controls_removed)
{
  ck_assert_str_eq (ids_of (&real, "ab\x01" "c\x7f" "d\xef\xbf\xbd", 16),
                    ids_of (&real, "abcd", 16));
}
END_TEST

START_TEST (test_invalid_utf8_skipped)
{
  ck_assert_str_eq (ids_of (&real, "ab\xff" "cd\xc3", 16),
                    ids_of (&real, "abcd", 16));
  ck_assert_str_eq (ids_of (&real, "\xe2\x82", 16),
                    ids_of (&real, "", 16));
}
END_TEST

START_TEST (test_long_word_unknown)
{
  char word[102];

  memset (word, 'a', 100);
  word[100] = '\0';
  ck_assert_ptr_null (strstr (tokens_of (&real, word), "[UNK]"));
  word[100] = 'a';
  word[101] = '\0';
  ck_assert_str_eq (tokens_of (&real, word), "[CLS] [UNK] [SEP]");
}
END_TEST

START_TEST (test_truncation)
{
  int64_t ids[5];
  size_t n = tokenize (&real, "one two three four five six", ids, 5);

  ck_assert_uint_eq (n, 5);
  ck_assert_int_eq (ids[0], real.cls);
  ck_assert_int_eq (ids[1], tokenizer_lookup (&real, "one", 3));
  ck_assert_int_eq (ids[3], tokenizer_lookup (&real, "three", 5));
  ck_assert_int_eq (ids[4], real.sep);

  n = tokenize (&real, "one two", ids, 2);
  ck_assert_uint_eq (n, 2);
  ck_assert_int_eq (ids[1], real.sep);
}
END_TEST

/* A small vocabulary, for the WordPiece algorithm.  */

static struct tokenizer small;
static char *small_dir;

static void
load_small (void)
{
  char *path;

  small_dir = tmpdir_new ();
  path = path_join (small_dir, "vocab.txt");
  write_file (path, "[PAD]\n[UNK]\r\n[CLS]\n[SEP]\nun\n##aff\n##able\n"
              "una\nab\n##le\n##b\na\n##a\n,\nun\n", (size_t) -1);
  ck_assert (tokenizer_load (&small, path));
  free (path);
}

static void
free_small (void)
{
  tokenizer_free (&small);
  tmpdir_remove (small_dir);
}

START_TEST (test_wordpiece_longest_match)
{
  /* "una" beats "un", then "##ff" is missing: the word is unknown.  */
  ck_assert_str_eq (tokens_of (&small, "unaffable"), "[CLS] [UNK] [SEP]");
  ck_assert_str_eq (tokens_of (&small, "able"), "[CLS] ab ##le [SEP]");
  ck_assert_str_eq (tokens_of (&small, "abc"), "[CLS] [UNK] [SEP]");
  ck_assert_str_eq (tokens_of (&small, "ab"), "[CLS] ab [SEP]");
  ck_assert_str_eq (tokens_of (&small, "abb"), "[CLS] ab ##b [SEP]");
  ck_assert_str_eq (tokens_of (&small, "aaa,a"),
                    "[CLS] a ##a ##a , a [SEP]");
  /* An unknown word does not swallow its neighbours.  */
  ck_assert_str_eq (tokens_of (&small, "ab zz ab"),
                    "[CLS] ab [UNK] ab [SEP]");
}
END_TEST

START_TEST (test_wordpiece_multibyte)
{
  char *path = path_join (small_dir, "vocab2.txt");
  struct tokenizer t;

  /* "é" is two bytes; a piece must not end in the middle of it.  */
  write_file (path, "[UNK]\n[CLS]\n[SEP]\nx\n##\xc3\n##\xc3\xa9\n",
              (size_t) -1);
  ck_assert (tokenizer_load (&t, path));
  /* Lowercase only: accents are stripped, so use a character that
     does not decompose.  */
  ck_assert_str_eq (tokens_of (&t, "x\xc3\xa9"), "[CLS] [UNK] [SEP]");
  tokenizer_free (&t);
  free (path);
}
END_TEST

START_TEST (test_duplicate_first_wins)
{
  ck_assert_int_eq (tokenizer_lookup (&small, "un", 2), 4);
  /* The CR of a CRLF line is not part of the token.  */
  ck_assert_int_eq (tokenizer_lookup (&small, "[UNK]", 5), 1);
  ck_assert_int_eq (small.unk, 1);
}
END_TEST

START_TEST (test_load_failures)
{
  struct tokenizer t;
  char *path = path_join (small_dir, "bad.txt");

  errno = 0;
  ck_assert (!tokenizer_load (&t, "/nonexistent/vocab.txt"));
  ck_assert_int_eq (errno, ENOENT);

  write_file (path, "[UNK]\n[CLS]\nhello\n", (size_t) -1);
  errno = 0;
  ck_assert (!tokenizer_load (&t, path));
  ck_assert_int_eq (errno, EINVAL);
  ck_assert_ptr_null (t.words);
  free (path);
}
END_TEST

static Suite *
tokenizer_suite (void)
{
  Suite *s = suite_create ("tokenizer");
  TCase *tc;

  tc = tcase_create ("real");
  tcase_add_checked_fixture (tc, load_real, free_real);
  tcase_add_test (tc, test_special_ids);
  tcase_add_test (tc, test_reference_cases);
  tcase_add_test (tc, test_basic_words);
  tcase_add_test (tc, test_accents_and_case);
  tcase_add_test (tc, test_chinese_split);
  tcase_add_test (tc, test_special_tokens_in_text);
  tcase_add_test (tc, test_controls_removed);
  tcase_add_test (tc, test_invalid_utf8_skipped);
  tcase_add_test (tc, test_long_word_unknown);
  tcase_add_test (tc, test_truncation);
  suite_add_tcase (s, tc);

  tc = tcase_create ("small");
  tcase_add_checked_fixture (tc, load_small, free_small);
  tcase_add_test (tc, test_wordpiece_longest_match);
  tcase_add_test (tc, test_wordpiece_multibyte);
  tcase_add_test (tc, test_duplicate_first_wins);
  tcase_add_test (tc, test_load_failures);
  suite_add_tcase (s, tc);
  return s;
}

int
main (void)
{
  SRunner *sr = srunner_create (tokenizer_suite ());
  int failed;

  srunner_run_all (sr, CK_NORMAL);
  failed = srunner_ntests_failed (sr);
  srunner_free (sr);
  return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
