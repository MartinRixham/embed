/* test-whisper.c -- unit tests for whisper.c.
   Copyright (C) 2026 Martin Rixham

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   These test the parts that need no model: the spectrogram, reading
   the vocabulary and finding loops.  test-model runs the model.  */

#include "../src/whisper.c"

#include "testutil.h"

#include <check.h>

static float *
spectrogram (const float *pcm, size_t n)
{
  float *mel = xmalloc (WHISPER_MELS * WHISPER_FRAMES * sizeof *mel);

  whisper_log_mel (pcm, n, mel);
  return mel;
}

START_TEST (test_mel_scale)
{
  /* Linear to 1 kHz, logarithmic above, and continuous.  */
  ck_assert_double_eq_tol (hz_to_mel (0), 0, 1e-9);
  ck_assert_double_eq_tol (hz_to_mel (1000), 15, 1e-9);
  ck_assert_double_eq_tol (hz_to_mel (6400), 42, 1e-9);
  for (double hz = 50; hz < 8000; hz *= 1.7)
    ck_assert_double_eq_tol (mel_to_hz (hz_to_mel (hz)), hz, 1e-6);
}
END_TEST

START_TEST (test_mel_filters)
{
  const float *f = mel_filters ();

  /* Each is a triangle with unit area (in hertz), and so the higher
     ones, which are wider, are lower.  */
  for (int m = 0; m < WHISPER_MELS; m++)
    {
      double area = 0, top = 0;

      for (int b = 0; b < N_BINS; b++)
        {
          area += f[m * N_BINS + b] * (WHISPER_RATE / 2.0) / (N_BINS - 1);
          if (f[m * N_BINS + b] > top)
            top = f[m * N_BINS + b];
        }
      ck_assert_msg (top > 0, "filter %d is empty", m);
      if (m > 10)
        ck_assert_double_eq_tol (area, 1, 0.1);
    }
  ck_assert_float_gt (f[0 * N_BINS + 1], f[79 * N_BINS + 190]);
}
END_TEST

START_TEST (test_fft_matches_dft)
{
  double complex in[N_FFT], out[N_FFT], tw[N_FFT];

  for (int k = 0; k < N_FFT; k++)
    {
      tw[k] = cexp (-2 * M_PI * I * k / N_FFT);
      in[k] = sin (k * 0.37) + 0.5 * cos (k * 1.9) + (k % 7) * 0.1;
    }
  fft (in, 1, N_FFT, out, tw);
  for (int k = 0; k < N_FFT; k += 13)
    {
      double complex sum = 0;

      for (int n = 0; n < N_FFT; n++)
        sum += in[n] * cexp (-2 * M_PI * I * k * n / N_FFT);
      ck_assert_double_eq_tol (cabs (out[k] - sum), 0, 1e-9);
    }
}
END_TEST

START_TEST (test_log_mel_silence)
{
  float *mel = spectrogram (NULL, 0);

  /* Everything at the floor, 1e-10, scaled.  */
  for (size_t i = 0; i < WHISPER_MELS * WHISPER_FRAMES; i++)
    ck_assert_float_eq_tol (mel[i], (-10 + 4) / 4.0, 1e-6);
  free (mel);
}
END_TEST

START_TEST (test_log_mel_tone)
{
  size_t n = 2 * WHISPER_RATE;
  float *pcm = xmalloc (n * sizeof *pcm), *mel;
  int best = 0;

  /* Two seconds at 1 kHz, which is 15 on the mel scale.  */
  for (size_t i = 0; i < n; i++)
    pcm[i] = 0.5 * sin (2 * M_PI * 1000 * i / WHISPER_RATE);
  mel = spectrogram (pcm, n);
  for (int m = 0; m < WHISPER_MELS; m++)
    if (mel[m * WHISPER_FRAMES + 100] > mel[best * WHISPER_FRAMES + 100])
      best = m;
  ck_assert_int_eq (best, (int) round (15 / hz_to_mel (8000) * 81) - 1);
  /* After the tone, everything is 80 dB down, the most allowed.  */
  ck_assert_float_gt (mel[best * WHISPER_FRAMES + 100], 0.5);
  ck_assert_float_eq_tol (mel[best * WHISPER_FRAMES + 2000],
                          mel[best * WHISPER_FRAMES + 100] - 2, 1e-5);
  free (pcm);
  free (mel);
}
END_TEST

START_TEST (test_log_mel_too_long)
{
  size_t n = WHISPER_SAMPLES + 1000;
  float *pcm = xmalloc (n * sizeof *pcm), *mel;

  for (size_t i = 0; i < n; i++)
    pcm[i] = i % 100 < 50 ? 0.1 : -0.1;
  /* The excess is ignored, and it does not run off the end.  */
  mel = spectrogram (pcm, n);
  free (mel);
  free (pcm);
}
END_TEST

START_TEST (test_cut_loop)
{
  int64_t a[] = { 1, 2, 3, 4, 5 };
  int64_t b[] = { 1, 7, 7, 7 };
  int64_t c[] = { 9, 1, 2, 1, 2, 1, 2 };
  int64_t d[] = { 1, 2, 1, 2 };

  ck_assert_uint_eq (cut_loop (a, 5), 5);
  ck_assert_uint_eq (cut_loop (b, 4), 2);
  ck_assert_uint_eq (cut_loop (c, 7), 3);
  /* Twice is not yet a loop.  */
  ck_assert_uint_eq (cut_loop (d, 4), 4);
}
END_TEST

static char *
json (const char *s)
{
  struct buf b;

  buf_init (&b);
  if (!json_string (&s, &b))
    {
      buf_free (&b);
      return NULL;
    }
  return b.data ? b.data : xstrdup ("");
}

START_TEST (test_json_string)
{
  ck_assert_str_eq (json ("\"abc\" rest"), "abc");
  ck_assert_str_eq (json ("\"\""), "");
  ck_assert_str_eq (json ("\"a\\\"b\\\\c\\/d\\n\""), "a\"b\\c/d\n");
  ck_assert_str_eq (json ("\"\\u0120x\""), "\xc4\xa0x");
  ck_assert_str_eq (json ("\"\\ud83d\\ude00\""), "\xf0\x9f\x98\x80");
  ck_assert_ptr_null (json ("\"unterminated"));
  ck_assert_ptr_null (json ("\"bad \\u12g4\""));
  ck_assert_ptr_null (json ("no quote"));
}
END_TEST

START_TEST (test_token_bytes)
{
  char space[] = "\xc4\xa0the";         /* U+0120, for a space.  */
  char newline[] = "\xc4\x8a";          /* U+010A, for a newline.  */
  char accented[] = "\xc3\x83\xc2\xa9"; /* "é" in UTF-8, byte by byte.  */

  ck_assert_uint_eq (token_bytes (space, strlen (space)), 4);
  ck_assert (memcmp (space, " the", 4) == 0);
  ck_assert_uint_eq (token_bytes (newline, strlen (newline)), 1);
  ck_assert_int_eq (newline[0], '\n');
  ck_assert_uint_eq (token_bytes (accented, strlen (accented)), 2);
  ck_assert (memcmp (accented, "\xc3\xa9", 2) == 0);
}
END_TEST

START_TEST (test_load_vocab_incomplete)
{
  char *dir = tmpdir_new (), *p = path_join (dir, "vocab.json");
  struct whisper w;

  memset (&w, 0, sizeof w);
  write_file (p, "{\"a\": 0, \"\\u0120b\": 1}", (size_t) -1);
  ck_assert (!load_vocab (&w, p));
  for (size_t i = 0; i < w.n_tokens; i++)
    free (w.tokens[i]);
  free (w.tokens);
  free (w.token_len);
  memset (&w, 0, sizeof w);
  write_file (p, "[1, 2]", (size_t) -1);
  ck_assert (!load_vocab (&w, p));
  free (w.tokens);
  free (w.token_len);
  memset (&w, 0, sizeof w);
  ck_assert (!load_vocab (&w, "/nonexistent/vocab.json"));
  tmpdir_remove (dir);
}
END_TEST

START_TEST (test_load_vocab_complete)
{
  char *dir = tmpdir_new (), *p = path_join (dir, "vocab.json");
  struct whisper w;
  struct buf b;
  char entry[64];

  buf_init (&b);
  buf_puts (&b, "{\n");
  for (int i = 0; i < TOKEN_EOT; i++)
    {
      snprintf (entry, sizeof entry, "%s\"t%d\": %d", i ? ",\n" : "", i, i);
      buf_puts (&b, entry);
    }
  /* Special tokens beyond the ordinary ones are ignored.  */
  buf_puts (&b, ",\n\"<|endoftext|>\": 50257,\n\"\\u0120x\": 5\n}\n");
  write_file (p, b.data, b.len);
  memset (&w, 0, sizeof w);
  ck_assert (load_vocab (&w, p));
  ck_assert_uint_eq (w.token_len[42], 3);
  ck_assert (memcmp (w.tokens[42], "t42", 3) == 0);
  /* The first spelling of an id wins.  */
  ck_assert (memcmp (w.tokens[5], "t5", 2) == 0);
  for (size_t i = 0; i < w.n_tokens; i++)
    free (w.tokens[i]);
  free (w.tokens);
  free (w.token_len);
  buf_free (&b);
  tmpdir_remove (dir);
}
END_TEST

int
main (void)
{
  Suite *s = suite_create ("whisper");
  TCase *tc = tcase_create ("whisper");
  SRunner *sr;
  int failed;

  tcase_set_timeout (tc, 30);
  tcase_add_test (tc, test_mel_scale);
  tcase_add_test (tc, test_mel_filters);
  tcase_add_test (tc, test_fft_matches_dft);
  tcase_add_test (tc, test_log_mel_silence);
  tcase_add_test (tc, test_log_mel_tone);
  tcase_add_test (tc, test_log_mel_too_long);
  tcase_add_test (tc, test_cut_loop);
  tcase_add_test (tc, test_json_string);
  tcase_add_test (tc, test_token_bytes);
  tcase_add_test (tc, test_load_vocab_incomplete);
  tcase_add_test (tc, test_load_vocab_complete);
  suite_add_tcase (s, tc);
  sr = srunner_create (s);
  srunner_run_all (sr, CK_NORMAL);
  failed = srunner_ntests_failed (sr);
  srunner_free (sr);
  return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
