/* test-index.c -- unit tests for embed-index.c.
   Copyright (C) 2026 Martin Rixham

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   The file under test is included directly so that its static
   functions can be exercised; its main is renamed so that the tests
   can run it, in a child process, as the real program would run.

   Tests of media extraction generate their sample files with ffmpeg,
   zip and a hand-written PDF, and are skipped when a tool they need
   is missing.  */

#define main index_main
#include "../embed-index.c"
#undef main

#include "fake-model.h"
#include "testutil.h"

#include <check.h>
#include <poll.h>
#include <pty.h>
#include <stdarg.h>
#include <sys/time.h>

/* A temporary home directory, with the fake model logging what it
   embeds.  */
struct fixture
{
  char *root;                   /* Holds HOME and the test's own files.  */
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

/* Return HOME/NAME, freshly allocated.  */
static char *
home_file (const char *name)
{
  return path_join (fx.home, name);
}

static void
put (const char *name, const char *content)
{
  char *p = home_file (name);
  write_file (p, content, (size_t) -1);
  free (p);
}

/* Run embed-index with the fake model and the given extra
   arguments, ending with NULL.  Return its exit status.  */
static int
run_index (const char *arg, ...)
{
  char *argv[32];
  int argc = 0;
  va_list ap;

  argv[argc++] = "embed-index";
  va_start (ap, arg);
  for (const char *a = arg; a; a = va_arg (ap, const char *))
    argv[argc++] = (char *) a;
  va_end (ap);
  argv[argc] = NULL;
  return run_main (index_main, argv, fx.out, fx.err);
}

static char *
err_text (void)
{
  char *s = read_file (fx.err);
  return s ? s : strdup ("");
}

/* Return the number of texts embedded, not counting opening the
   model.  */
static size_t
requests (void)
{
  char *log = read_file (fx.log);

  return count_substr (log, "\n----\n") - count_substr (log, "open gpu=");
}

/* Return the number of records for HOME/NAME in the store.  */
static size_t
records_for (const char *name)
{
  char *needle;
  size_t n;

  if (asprintf (&needle, "\t%s/%s\t", fx.home, name) < 0)
    abort ();
  n = count_substr (read_file (fx.store), needle);
  free (needle);
  return n;
}

static size_t
record_count (void)
{
  char *s = read_file (fx.store);
  size_t n = count_substr (s, "\n");

  /* Less the header.  */
  return s && s[0] == '#' ? n - 1 : n;
}

/* Set the modification time of HOME/NAME to SECONDS from now.  */
static void
touch_by (const char *name, int seconds)
{
  struct timeval tv[2];
  char *p = home_file (name);

  gettimeofday (&tv[0], NULL);
  tv[0].tv_sec += seconds;
  tv[1] = tv[0];
  ck_assert_int_eq (utimes (p, tv), 0);
  free (p);
}

/* Simple predicates.  */

START_TEST (test_starts_ends_with)
{
  ck_assert (starts_with ("text/plain", "text/"));
  ck_assert (starts_with ("abc", ""));
  ck_assert (!starts_with ("tex", "text/"));
  ck_assert (ends_with ("image/svg+xml", "+xml"));
  ck_assert (ends_with ("abc", ""));
  ck_assert (!ends_with ("xml", "+xml"));
}
END_TEST

START_TEST (test_textual_mime)
{
  static const char *const yes[] = {
    "text/plain", "text/x-c", "text/html", "application/json",
    "application/javascript", "application/xml", "image/svg+xml",
    "application/ld+json", "application/x-shellscript", "message/rfc822",
    "application/toml", NULL
  };
  static const char *const no[] = {
    "application/pdf", "image/png", "audio/mpeg", "video/mp4",
    "application/octet-stream", "application/zip", "inode/x-empty",
    "application/x-executable", "", NULL
  };

  for (int i = 0; yes[i]; i++)
    ck_assert_msg (textual_mime (yes[i]), "%s", yes[i]);
  for (int i = 0; no[i]; i++)
    ck_assert_msg (!textual_mime (no[i]), "%s", no[i]);
}
END_TEST

START_TEST (test_under_root)
{
  char *roots[] = { "/home/a", "/srv/data/", NULL };
  char *slash[] = { "/", NULL };

  ck_assert (under_root ("/home/a", roots));
  ck_assert (under_root ("/home/a/x/y", roots));
  ck_assert (!under_root ("/home/ab/x", roots));
  ck_assert (!under_root ("/home", roots));
  ck_assert (under_root ("/srv/data/file", roots));
  ck_assert (!under_root ("/srv/database", roots));
  ck_assert (under_root ("/anything", slash));
}
END_TEST

/* Markup stripping.  */

static char *
stripped (const char *s)
{
  struct buf b;

  buf_init (&b);
  buf_puts (&b, s);
  strip_markup (&b);
  ck_assert_uint_eq (b.len, strlen (b.data));
  return b.data;
}

START_TEST (test_strip_markup)
{
  ck_assert_str_eq (stripped ("<p>Hello <b>world</b></p>"), "Hello world ");
  ck_assert_str_eq (stripped ("  a \n\n\t b  "), "a b ");
  ck_assert_str_eq (stripped ("<w:t>one</w:t><w:t>two</w:t>"), "one two ");
  ck_assert_str_eq (stripped ("x &amp; y"), "x &amp; y");
  ck_assert_str_eq (stripped ("before <unclosed tag"), "before ");
  ck_assert_str_eq (stripped (""), "");
  ck_assert_str_eq (stripped ("<a><b><c>"), "");
  ck_assert_str_eq (stripped ("1 > 0"), "1 > 0");
}
END_TEST

/* Reading files and running helpers.  */

START_TEST (test_read_head)
{
  struct buf b;
  char *p = home_file ("f.txt");

  put ("f.txt", "0123456789");
  buf_init (&b);
  read_head (p, &b, 4);
  ck_assert_str_eq (b.data, "0123");
  b.len = 0;
  read_head (p, &b, 100);
  ck_assert_str_eq (b.data, "0123456789");
  buf_free (&b);
}
END_TEST

START_TEST (test_read_head_missing_and_unreadable)
{
  struct buf b;
  char *p = home_file ("secret.txt");

  buf_init (&b);
  read_head ("/nonexistent/file", &b, 100);
  ck_assert_uint_eq (b.len, 0);
  put ("secret.txt", "hidden");
  chmod (p, 0);
  read_head (p, &b, 100);
  if (geteuid () != 0)
    ck_assert_uint_eq (b.len, 0);
  buf_free (&b);
}
END_TEST

START_TEST (test_run_capture_output)
{
  struct buf b;
  const char *argv[] = { "printf", "%s|%s", "a b", "$(rm -rf x);*", NULL };

  buf_init (&b);
  run_capture (argv, &b, 1000);
  /* Arguments reach the program literally, without a shell.  */
  ck_assert_str_eq (b.data, "a b|$(rm -rf x);*");
  buf_free (&b);
}
END_TEST

START_TEST (test_run_capture_ignores_stderr)
{
  struct buf b;
  const char *argv[] = { "sh", "-c", "echo out; echo err >&2", NULL };

  buf_init (&b);
  run_capture (argv, &b, 1000);
  ck_assert_str_eq (b.data, "out\n");
  buf_free (&b);
}
END_TEST

START_TEST (test_run_capture_limits_output)
{
  struct buf b;
  const char *argv[] = { "yes", "spam", NULL };
  double start = now ();

  buf_init (&b);
  run_capture (argv, &b, 1001);
  ck_assert_uint_eq (b.len, 1001);
  /* yes(1) dies of SIGPIPE rather than running into the time limit.  */
  ck_assert_double_lt (now () - start, 1.0);
  buf_free (&b);
}
END_TEST

START_TEST (test_run_capture_time_limit)
{
  struct buf b;
  const char *argv[] = { "sh", "-c", "echo partial; exec sleep 30", NULL };
  double start = now ();

  helper_timeout = 1;
  buf_init (&b);
  run_capture (argv, &b, 1000);
  ck_assert_str_eq (b.data, "partial\n");
  ck_assert_double_lt (now () - start, 3.5);
  buf_free (&b);
}
END_TEST

START_TEST (test_run_capture_stdin_is_empty)
{
  struct buf b;
  const char *argv[] = { "cat", NULL };
  double start = now ();

  buf_init (&b);
  run_capture (argv, &b, 1000);
  ck_assert_str_eq (b.data, "");
  ck_assert_double_lt (now () - start, 1.0);
  buf_free (&b);
}
END_TEST

START_TEST (test_run_capture_missing_program)
{
  struct buf b;
  const char *argv[] = { "no-such-program-xyz", "arg", NULL };

  buf_init (&b);
  run_capture (argv, &b, 1000);
  ck_assert_uint_eq (b.len, 0);
  buf_free (&b);
}
END_TEST

START_TEST (test_run_capture_many_calls)
{
  /* No file descriptors or zombies leak between calls.  */
  const char *argv[] = { "true", NULL };
  struct buf b;
  int before, after;

  buf_init (&b);
  before = dup (0);
  close (before);
  for (int i = 0; i < 300; i++)
    run_capture (argv, &b, 10);
  after = dup (0);
  close (after);
  ck_assert_int_eq (before, after);
  /* Nothing left to reap.  */
  ck_assert_int_eq (waitpid (-1, NULL, WNOHANG), -1);
  ck_assert_int_eq (errno, ECHILD);
  buf_free (&b);
}
END_TEST

/* Content extraction by type.  */

static char *
extract (const char *name, const char *mime)
{
  struct buf b;
  char *p = home_file (name);

  buf_init (&b);
  extract_content (p, mime, &b);
  free (p);
  return b.data;
}

START_TEST (test_extract_text)
{
  char big[MAX_CONTENT * 3];
  char *s;

  put ("t.txt", "plain words here");
  ck_assert_str_eq (extract ("t.txt", "text/plain"), "plain words here");

  memset (big, 'q', sizeof big - 1);
  big[sizeof big - 1] = '\0';
  put ("big.txt", big);
  s = extract ("big.txt", "text/plain");
  ck_assert_uint_eq (strlen (s), MAX_CONTENT);
}
END_TEST

START_TEST (test_extract_html_and_xml)
{
  put ("p.html", "<html><body><h1>Title</h1><p>Body text</p></body></html>");
  ck_assert_str_eq (extract ("p.html", "text/html"), "Title Body text ");
  put ("d.xml", "<root><item>value</item></root>");
  ck_assert_str_eq (extract ("d.xml", "text/xml"), "value ");
  put ("i.svg", "<svg><title>Logo</title></svg>");
  ck_assert_str_eq (extract ("i.svg", "image/svg+xml"), "Logo ");
}
END_TEST

START_TEST (test_extract_json_kept_verbatim)
{
  put ("c.json", "{\"key\": \"<value>\"}");
  ck_assert_str_eq (extract ("c.json", "application/json"),
                    "{\"key\": \"<value>\"}");
}
END_TEST

START_TEST (test_extract_unknown_type)
{
  put ("blob.bin", "\x01\x02\x03 sourdough");
  ck_assert_str_eq (extract ("blob.bin", "application/octet-stream"), "");
  ck_assert_str_eq (extract ("blob.bin", "application/x-executable"), "");
}
END_TEST

/* Write a one-page PDF showing TEXT to HOME/NAME.  */
static void
make_pdf (const char *name, const char *text)
{
  struct buf pdf;
  size_t offsets[6];
  char stream[256], tmp[512];

  snprintf (stream, sizeof stream, "BT /F1 18 Tf 10 50 Td (%s) Tj ET",
            text);
  buf_init (&pdf);
  buf_puts (&pdf, "%PDF-1.4\n");
  offsets[1] = pdf.len;
  buf_puts (&pdf, "1 0 obj <</Type /Catalog /Pages 2 0 R>> endobj\n");
  offsets[2] = pdf.len;
  buf_puts (&pdf, "2 0 obj <</Type /Pages /Kids [3 0 R] /Count 1>> "
            "endobj\n");
  offsets[3] = pdf.len;
  buf_puts (&pdf, "3 0 obj <</Type /Page /Parent 2 0 R /MediaBox "
            "[0 0 400 100] /Contents 4 0 R /Resources <</Font <</F1 5 0 R"
            ">> >> >> endobj\n");
  offsets[4] = pdf.len;
  snprintf (tmp, sizeof tmp, "4 0 obj <</Length %zu>> stream\n%s\n"
            "endstream endobj\n", strlen (stream), stream);
  buf_puts (&pdf, tmp);
  offsets[5] = pdf.len;
  buf_puts (&pdf, "5 0 obj <</Type /Font /Subtype /Type1 /BaseFont "
            "/Helvetica>> endobj\n");
  size_t xref = pdf.len;
  buf_puts (&pdf, "xref\n0 6\n0000000000 65535 f \n");
  for (int i = 1; i < 6; i++)
    {
      snprintf (tmp, sizeof tmp, "%010zu 00000 n \n", offsets[i]);
      buf_puts (&pdf, tmp);
    }
  snprintf (tmp, sizeof tmp, "trailer <</Size 6 /Root 1 0 R>>\n"
            "startxref\n%zu\n%%%%EOF\n", xref);
  buf_puts (&pdf, tmp);

  char *p = home_file (name);
  write_file (p, pdf.data, pdf.len);
  free (p);
  buf_free (&pdf);
}

/* Run the shell command FMT in HOME, failing the test if it fails.
   Standard input is /dev/null: Check runs each test in its own
   process group, so when make check is run from a terminal, a
   command that touches it (as ffmpeg does, to read keystrokes) is
   stopped by SIGTTIN or SIGTTOU until the test times out.  */
static void
shell (const char *fmt, ...)
{
  char *cmd, *full;
  va_list ap;

  va_start (ap, fmt);
  if (vasprintf (&cmd, fmt, ap) < 0)
    abort ();
  va_end (ap);
  if (asprintf (&full, "cd '%s' && { %s; } </dev/null", fx.home, cmd) < 0)
    abort ();
  ck_assert_msg (system (full) == 0, "command failed: %s", cmd);
  free (cmd);
  free (full);
}

START_TEST (test_extract_pdf)
{
  if (!have_program ("pdftotext"))
    return;
  make_pdf ("doc.pdf", "Quarterly invoice from Acme");
  ck_assert_ptr_nonnull (strstr (extract ("doc.pdf", "application/pdf"),
                                 "Quarterly invoice from Acme"));
}
END_TEST

START_TEST (test_extract_broken_pdf)
{
  put ("bad.pdf", "%PDF-1.4\ngarbage");
  /* No crash, no hang, and nothing sensible to say.  */
  ck_assert_str_eq (extract ("bad.pdf", "application/pdf"), "");
}
END_TEST

START_TEST (test_extract_image_ocr)
{
  char *text;

  if (!have_program ("tesseract") || !have_program ("ffmpeg"))
    return;
  shell ("ffmpeg -v quiet -f lavfi -i color=white:s=640x120 -frames:v 1 "
         "-vf \"drawtext=text='INVOICE TOTAL DUE':fontsize=48:x=20:y=35:"
         "fontcolor=black\" scan.png");
  text = extract ("scan.png", "image/png");
  ck_assert_msg (strstr (text, "INVOICE TOTAL DUE"), "OCR gave: %s", text);
}
END_TEST

START_TEST (test_extract_audio_tags)
{
  char *text;

  if (!have_program ("ffprobe") || !have_program ("ffmpeg"))
    return;
  shell ("ffmpeg -v quiet -f lavfi -i sine=frequency=440:duration=1 "
         "-metadata title='Moonlight Sonata' -metadata artist=Beethoven "
         "song.mp3");
  text = extract ("song.mp3", "audio/mpeg");
  ck_assert_ptr_nonnull (strstr (text, "TAG:title=Moonlight Sonata"));
  ck_assert_ptr_nonnull (strstr (text, "TAG:artist=Beethoven"));
  ck_assert_ptr_nonnull (strstr (text, "duration="));
}
END_TEST

START_TEST (test_extract_video_streams)
{
  char *text;

  if (!have_program ("ffprobe") || !have_program ("ffmpeg"))
    return;
  shell ("ffmpeg -v quiet -f lavfi -i testsrc=s=320x240:d=1 "
         "-metadata title='Sunset timelapse' clip.mp4");
  text = extract ("clip.mp4", "video/mp4");
  ck_assert_ptr_nonnull (strstr (text, "TAG:title=Sunset timelapse"));
  ck_assert_ptr_nonnull (strstr (text, "width=320"));
  ck_assert_ptr_nonnull (strstr (text, "height=240"));
}
END_TEST

START_TEST (test_extract_docx)
{
  char *text;

  if (!have_program ("zip") || !have_program ("unzip"))
    return;
  put ("d/[Content_Types].xml", "<Types/>");
  put ("d/word/document.xml",
       "<w:document><w:body><w:p><w:t>Minutes of the</w:t><w:t>board "
       "meeting</w:t></w:p></w:body></w:document>");
  shell ("cd d && zip -q -X -r ../report.docx '[Content_Types].xml' word");
  text = extract ("report.docx", "application/vnd.openxmlformats-"
                  "officedocument.wordprocessingml.document");
  ck_assert_str_eq (text, "Minutes of the board meeting ");
}
END_TEST

START_TEST (test_extract_odt)
{
  char *text;

  if (!have_program ("zip") || !have_program ("unzip"))
    return;
  put ("o/mimetype", "application/vnd.oasis.opendocument.text");
  put ("o/content.xml", "<office:document-content><text:p>Holiday "
       "packing list</text:p></office:document-content>");
  shell ("cd o && zip -q -X -0 ../notes.odt mimetype "
         "&& zip -q -X ../notes.odt content.xml");
  text = extract ("notes.odt", "application/vnd.oasis.opendocument.text");
  ck_assert_str_eq (text, "Holiday packing list ");
}
END_TEST

START_TEST (test_extract_pptx_all_slides)
{
  char *text;

  if (!have_program ("zip") || !have_program ("unzip"))
    return;
  put ("s/ppt/slides/slide1.xml", "<p:sld><a:t>First slide</a:t></p:sld>");
  put ("s/ppt/slides/slide2.xml", "<p:sld><a:t>Second slide</a:t></p:sld>");
  shell ("cd s && zip -q -X -r ../deck.pptx ppt");
  text = extract ("deck.pptx", "application/vnd.openxmlformats-"
                  "officedocument.presentationml.presentation");
  ck_assert_ptr_nonnull (strstr (text, "First slide"));
  ck_assert_ptr_nonnull (strstr (text, "Second slide"));
}
END_TEST

START_TEST (test_extract_xlsx_strings)
{
  char *text;

  if (!have_program ("zip") || !have_program ("unzip"))
    return;
  put ("x/xl/sharedStrings.xml", "<sst><si><t>Budget</t></si><si><t>2026"
       "</t></si></sst>");
  shell ("cd x && zip -q -X -r ../sheet.xlsx xl");
  text = extract ("sheet.xlsx", "application/vnd.openxmlformats-"
                  "officedocument.spreadsheetml.sheet");
  ck_assert_str_eq (text, "Budget 2026 ");
}
END_TEST

/* Describing files.  */

static magic_t mime_cookie, desc_cookie;

static void
open_magic (void)
{
  mime_cookie = magic_open (MAGIC_MIME_TYPE | MAGIC_ERROR);
  desc_cookie = magic_open (MAGIC_ERROR);
  ck_assert_int_eq (magic_load (mime_cookie, NULL), 0);
  ck_assert_int_eq (magic_load (desc_cookie, NULL), 0);
}

static char *
describe (const char *path, bool with_content)
{
  struct buf b;

  buf_init (&b);
  describe_file (mime_cookie, desc_cookie, path, fx.home, with_content,
                 &b);
  return b.data;
}

START_TEST (test_describe_text_file)
{
  char *d;

  open_magic ();
  put ("Recipes/bread.txt", "Sourdough starter notes\n");
  d = describe (home_file ("Recipes/bread.txt"), true);
  ck_assert_str_eq (d, "search_document: File name: bread.txt\n"
                    "Location: ~/Recipes\nType: ASCII text\n\n"
                    "Sourdough starter notes\n");
}
END_TEST

START_TEST (test_describe_without_content)
{
  char *d;

  open_magic ();
  put ("note.txt", "Sourdough starter notes\n");
  d = describe (home_file ("note.txt"), false);
  ck_assert_str_eq (d, "search_document: File name: note.txt\n"
                    "Location: ~\nType: ASCII text\n\n");
}
END_TEST

START_TEST (test_describe_outside_home)
{
  char *dir = tmpdir_new ();
  char *p = path_join (dir, "x.txt");
  char *d, *expect;

  open_magic ();
  write_file (p, "hello\n", (size_t) -1);
  d = describe (p, false);
  if (asprintf (&expect, "search_document: File name: x.txt\n"
                "Location: %s\n", dir) < 0)
    abort ();
  ck_assert_msg (starts_with (d, expect), "got: %s", d);
  tmpdir_remove (dir);
}
END_TEST

START_TEST (test_describe_home_prefix_sibling)
{
  /* A directory whose name merely starts with HOME is not under it.  */
  char *sibling, *p, *d;

  open_magic ();
  if (asprintf (&sibling, "%s-other", fx.home) < 0)
    abort ();
  p = path_join (sibling, "y.txt");
  write_file (p, "hello\n", (size_t) -1);
  d = describe (p, false);
  ck_assert_ptr_null (strstr (d, "Location: ~"));
  tmpdir_remove (sibling);
}
END_TEST

START_TEST (test_describe_unreadable_file)
{
  char *p = home_file ("locked.txt");
  char *d;

  if (geteuid () == 0)
    return;
  open_magic ();
  put ("locked.txt", "sourdough");
  chmod (p, 0);
  d = describe (p, true);
  ck_assert_ptr_nonnull (strstr (d, "File name: locked.txt"));
  ck_assert_ptr_null (strstr (d, "sourdough"));
}
END_TEST

START_TEST (test_describe_pdf_type)
{
  char *d;

  if (!have_program ("pdftotext"))
    return;
  open_magic ();
  make_pdf ("Taxes/receipt.pdf", "Order receipt total 12 EUR");
  d = describe (home_file ("Taxes/receipt.pdf"), true);
  ck_assert_ptr_nonnull (strstr (d, "Type: PDF document"));
  ck_assert_ptr_nonnull (strstr (d, "Order receipt total 12 EUR"));
}
END_TEST

/* The store.  */

START_TEST (test_keep_push_grows)
{
  for (int i = 0; i < 10000; i++)
    keep_push (i % 3 == 0);
  ck_assert_uint_eq (n_lines, 10000);
  ck_assert_int_eq (keep[0], 1);
  ck_assert_int_eq (keep[9999], 1);
  ck_assert_int_eq (keep[9998], 0);
}
END_TEST

START_TEST (test_write_record_format)
{
  char *p = home_file ("rec");
  FILE *f = fopen (p, "w");
  struct stat st = { .st_mtime = 1700000000, .st_size = 42 };
  float v[3] = { 0.5f, -0.25f, 1e-7f };

  ck_assert (write_record (f, "/a/b c", &st, v, 3));
  fclose (f);
  ck_assert_str_eq (read_file (p), "1700000000 42\t/a/b c\t"
                    "3f000000be80000033d6bf95\n");
}
END_TEST

START_TEST (test_write_record_full_disk)
{
  FILE *f = fopen ("/dev/full", "w");
  struct stat st = { 0 };
  float v[1] = { 1 };

  if (!f)
    return;
  ck_assert (!write_record (f, "/x", &st, v, 1));
  fclose (f);
}
END_TEST

START_TEST (test_load_store_missing)
{
  char model[64];

  ck_assert (!load_store (fx.store, model, sizeof model));
  ck_assert_uint_eq (n_entries, 0);
}
END_TEST

START_TEST (test_load_store_bad_header)
{
  char model[64];

  write_file (fx.store, "not a header\n", (size_t) -1);
  load_store (fx.store, model, sizeof model);
}
END_TEST

START_TEST (test_load_store_empty_file)
{
  char model[64];

  write_file (fx.store, "", 0);
  load_store (fx.store, model, sizeof model);
}
END_TEST

START_TEST (test_load_store_header_only)
{
  char model[64];

  write_file (fx.store, "# embed-index model=m dim=3\n", (size_t) -1);
  ck_assert (load_store (fx.store, model, sizeof model));
  ck_assert_uint_eq (n_entries, 0);
  ck_assert_uint_eq (n_lines, 1);
  ck_assert_ptr_null (find_entry ("/a"));
}
END_TEST

START_TEST (test_load_store_dedupes_and_skips_junk)
{
  char model[64];
  struct entry *e;

  write_file (fx.store,
              "# embed-index model=mod dim=2\n"   /* line 0 */
              "10 1\t/b\t0 1\n"                   /* 1: superseded */
              "20 2\t/a\t1 0\n"                   /* 2 */
              "garbage line\n"                    /* 3: junk */
              "30 3\t/b\t1 1\n"                   /* 4: newest /b */
              "40 4\tno second tab\n"             /* 5: junk */
              "50 5\t/c\t0 0",                    /* 6: truncated */
              (size_t) -1);
  ck_assert (load_store (fx.store, model, sizeof model));
  ck_assert_str_eq (model, "mod");
  ck_assert_uint_eq (n_lines, 7);
  ck_assert_uint_eq (n_entries, 2);
  ck_assert_str_eq (entries[0].path, "/a");
  ck_assert_str_eq (entries[1].path, "/b");

  e = find_entry ("/b");
  ck_assert_ptr_nonnull (e);
  ck_assert_int_eq (e->mtime, 30);
  ck_assert_int_eq (e->size, 3);
  ck_assert_uint_eq (e->line, 4);
  ck_assert (!e->seen);
  ck_assert_ptr_null (find_entry ("/c"));
  ck_assert_ptr_null (find_entry ("/"));
  ck_assert_ptr_null (find_entry ("/zzz"));

  const char expect[] = { 1, 0, 1, 0, 1, 0, 0 };
  for (int i = 0; i < 7; i++)
    ck_assert_msg (keep[i] == expect[i], "line %d", i);
}
END_TEST

START_TEST (test_load_store_many)
{
  char model[64];
  FILE *f;

  make_parent_dirs (fx.store);
  f = fopen (fx.store, "w");
  fprintf (f, "# embed-index model=m dim=1\n");
  /* Written in reverse order, with every path twice.  */
  for (int round = 0; round < 2; round++)
    for (int i = 9999; i >= 0; i--)
      fprintf (f, "%d 0\t/f%05d\t1\n", round, i);
  fclose (f);

  ck_assert (load_store (fx.store, model, sizeof model));
  ck_assert_uint_eq (n_entries, 10000);
  for (int i = 0; i < 10000; i += 997)
    {
      char path[16];
      snprintf (path, sizeof path, "/f%05d", i);
      struct entry *e = find_entry (path);
      ck_assert_ptr_nonnull (e);
      ck_assert_int_eq (e->mtime, 1);
    }
}
END_TEST

START_TEST (test_compact_store)
{
  char *tmp;
  struct stat st;

  write_file (fx.store, "# h\nline1\nline2\nline3\nline4", (size_t) -1);
  keep_push (1);
  keep_push (0);
  keep_push (1);
  keep_push (0);
  keep_push (1);                /* Truncated, so dropped anyway.  */
  compact_store (fx.store);
  ck_assert_str_eq (read_file (fx.store), "# h\nline2\n");
  if (asprintf (&tmp, "%s.tmp", fx.store) < 0)
    abort ();
  ck_assert_int_ne (stat (tmp, &st), 0);
}
END_TEST

/* The whole program.  */

START_TEST (test_index_basic)
{
  char *store;

  put ("Recipes/bread.txt", "sourdough loaf");
  put ("Music/notes.txt", "sonata in C");
  put (".hidden/secret.txt", "secret");
  put (".bashrc", "secret");
  put ("empty.txt", "");
  ck_assert_int_eq (symlink ("Recipes/bread.txt",
                             home_file ("link.txt")), 0);

  ck_assert_int_eq (run_index (NULL), 0);
  store = read_file (fx.store);
  ck_assert_ptr_nonnull (store);
  ck_assert (starts_with (store, "# embed-index model=" EMBED_MODEL " "
                          "dim=8\n"));
  ck_assert_uint_eq (record_count (), 2);
  ck_assert_uint_eq (records_for ("Recipes/bread.txt"), 1);
  ck_assert_uint_eq (records_for ("Music/notes.txt"), 1);
  ck_assert_ptr_nonnull (strstr (err_text (), "embedded 2, unchanged 0, "
                                 "failed 0"));
  ck_assert_str_eq (read_file (fx.out), "");

  /* One warm-up request plus one per file, as documents.  */
  ck_assert_uint_eq (requests (), 3);
  ck_assert_uint_eq (count_substr (read_file (fx.log),
                                   "search_document: File name: "), 2);
}
END_TEST

START_TEST (test_index_vectors_are_unit_length)
{
  char *store, *line;

  put ("a.txt", "sourdough sourdough omega");
  ck_assert_int_eq (run_index (NULL), 0);
  store = read_file (fx.store);
  line = strchr (store, '\n') + 1;
  line = strchr (strchr (line, '\t') + 1, '\t') + 1;
  float v[FAKE_DIM];
  double norm = 0;
  ck_assert (parse_hex_vector (line, v, FAKE_DIM));
  for (int i = 0; i < FAKE_DIM; i++)
    norm += (double) v[i] * v[i];
  ck_assert_double_eq_tol (norm, 1.0, 1e-6);
}
END_TEST

START_TEST (test_index_hidden_with_all)
{
  put ("visible.txt", "alpha");
  put (".hidden/inner.txt", "secret");
  put (".profile", "secret");

  ck_assert_int_eq (run_index ("-a", NULL), 0);
  ck_assert_uint_eq (records_for (".hidden/inner.txt"), 1);
  ck_assert_uint_eq (records_for (".profile"), 1);
  /* The store itself is never indexed, even on a second run.  */
  ck_assert_int_eq (run_index ("-a", NULL), 0);
  ck_assert_ptr_null (strstr (read_file (fx.store), "/.local/share/embed/"));
  ck_assert_uint_eq (record_count (), 3);
}
END_TEST

START_TEST (test_index_rerun_skips_unchanged)
{
  put ("a.txt", "alpha");
  put ("b.txt", "omega");
  ck_assert_int_eq (run_index (NULL), 0);
  size_t before = requests ();
  char *store_before = read_file (fx.store);

  ck_assert_int_eq (run_index (NULL), 0);
  ck_assert_uint_eq (requests (), before + 1);  /* Only the warm-up.  */
  ck_assert_ptr_nonnull (strstr (err_text (), "embedded 0, unchanged 2"));
  ck_assert_str_eq (read_file (fx.store), store_before);
}
END_TEST

START_TEST (test_index_modified_file)
{
  char *store;

  put ("a.txt", "alpha");
  put ("b.txt", "omega");
  ck_assert_int_eq (run_index (NULL), 0);
  put ("a.txt", "sourdough");
  touch_by ("a.txt", 10);
  ck_assert_int_eq (run_index (NULL), 0);
  ck_assert_ptr_nonnull (strstr (err_text (), "embedded 1, unchanged 1"));
  ck_assert_uint_eq (records_for ("a.txt"), 1);
  ck_assert_uint_eq (record_count (), 2);
  store = read_file (fx.store);
  ck_assert_ptr_nonnull (strstr (read_file (fx.log), "sourdough"));
  (void) store;
}
END_TEST

START_TEST (test_index_same_mtime_new_size)
{
  put ("a.txt", "alpha");
  touch_by ("a.txt", -100);
  ck_assert_int_eq (run_index (NULL), 0);
  put ("a.txt", "alpha and more");
  touch_by ("a.txt", -100);
  ck_assert_int_eq (run_index (NULL), 0);
  ck_assert_ptr_nonnull (strstr (err_text (), "embedded 1, unchanged 0"));
  ck_assert_uint_eq (records_for ("a.txt"), 1);
}
END_TEST

START_TEST (test_index_deleted_file)
{
  char *p = home_file ("gone.txt");

  put ("keep.txt", "alpha");
  put ("gone.txt", "omega");
  ck_assert_int_eq (run_index (NULL), 0);
  unlink (p);
  ck_assert_int_eq (run_index (NULL), 0);
  ck_assert_uint_eq (records_for ("gone.txt"), 0);
  ck_assert_uint_eq (records_for ("keep.txt"), 1);
}
END_TEST

START_TEST (test_index_subdirectory_keeps_others)
{
  char *sub = home_file ("sub");
  char *gone = home_file ("outside-gone.txt");

  put ("outside.txt", "alpha");
  put ("outside-gone.txt", "alpha");
  put ("sub/inside.txt", "omega");
  put ("sub/inside-gone.txt", "omega");
  ck_assert_int_eq (run_index (NULL), 0);
  ck_assert_uint_eq (record_count (), 4);

  unlink (gone);
  unlink (home_file ("sub/inside-gone.txt"));
  put ("sub/new.txt", "sunset");
  ck_assert_int_eq (run_index (sub, NULL), 0);
  ck_assert_uint_eq (records_for ("outside.txt"), 1);
  ck_assert_uint_eq (records_for ("outside-gone.txt"), 0);
  ck_assert_uint_eq (records_for ("sub/inside.txt"), 1);
  ck_assert_uint_eq (records_for ("sub/inside-gone.txt"), 0);
  ck_assert_uint_eq (records_for ("sub/new.txt"), 1);
}
END_TEST

START_TEST (test_index_relative_directory)
{
  put ("docs/a.txt", "alpha");
  ck_assert_int_eq (chdir (fx.home), 0);
  ck_assert_int_eq (run_index ("docs/", NULL), 0);
  /* Stored by absolute path, however the directory was named.  */
  ck_assert_uint_eq (records_for ("docs/a.txt"), 1);
}
END_TEST

START_TEST (test_index_several_directories)
{
  put ("one/a.txt", "alpha");
  put ("two/b.txt", "omega");
  put ("three/c.txt", "sunset");
  ck_assert_int_eq (run_index (home_file ("one"), home_file ("two"), NULL),
                    0);
  ck_assert_uint_eq (record_count (), 2);
  ck_assert_uint_eq (records_for ("three/c.txt"), 0);
}
END_TEST

START_TEST (test_index_missing_directory)
{
  ck_assert_int_eq (run_index ("/nonexistent/dir", NULL), EXIT_FAILURE);
  ck_assert_ptr_nonnull (strstr (err_text (), "/nonexistent/dir"));
}
END_TEST

START_TEST (test_index_odd_file_names)
{
  put ("tab\there.txt", "alpha");
  put ("new\nline.txt", "alpha");
  put ("spaces and 'quotes' \"too\".txt", "omega");
  put ("ünïcödé ✓.txt", "sunset");
  ck_assert_int_eq (run_index (NULL), 0);
  /* Tabs and newlines would break the store's format.  */
  ck_assert_uint_eq (record_count (), 2);
  ck_assert_uint_eq (records_for ("spaces and 'quotes' \"too\".txt"), 1);
  ck_assert_uint_eq (records_for ("ünïcödé ✓.txt"), 1);
}
END_TEST

START_TEST (test_index_falls_back_to_name)
{
  char *log;

  put ("tricky.txt", "this content says FAKE_500");
  ck_assert_int_eq (run_index (NULL), 0);
  ck_assert_uint_eq (records_for ("tricky.txt"), 1);
  ck_assert_ptr_nonnull (strstr (err_text (), "embedded 1, unchanged 0, "
                                 "failed 0"));
  log = read_file (fx.log);
  /* Warm-up, the full attempt, then the name alone.  */
  ck_assert_uint_eq (requests (), 3);
  ck_assert_uint_eq (count_substr (log, "File name: tricky.txt"), 2);
  ck_assert_uint_eq (count_substr (log, "FAKE_500"), 1);
}
END_TEST

START_TEST (test_index_falls_back_on_wrong_dimension)
{
  put ("odd.txt", "content makes FAKE_DIM3");
  ck_assert_int_eq (run_index (NULL), 0);
  ck_assert_uint_eq (records_for ("odd.txt"), 1);
}
END_TEST

START_TEST (test_index_unembeddable_file)
{
  put ("FAKE_500.txt", "alpha");
  put ("fine.txt", "omega");
  ck_assert_int_eq (run_index (NULL), 0);
  ck_assert_uint_eq (records_for ("FAKE_500.txt"), 0);
  ck_assert_uint_eq (records_for ("fine.txt"), 1);
  ck_assert_ptr_nonnull (strstr (err_text (), "FAKE_500.txt: could not "
                                 "embed"));
  ck_assert_ptr_nonnull (strstr (err_text (), "embedded 1, unchanged 0, "
                                 "failed 1"));
  /* It is tried again next time.  */
  ck_assert_int_eq (run_index (NULL), 0);
  ck_assert_ptr_nonnull (strstr (err_text (), "failed 1"));
}
END_TEST

START_TEST (test_index_quiet_unless_verbose)
{
  put ("tricky.txt", "alpha");
  ck_assert_int_eq (run_index (NULL), 0);
  ck_assert_ptr_null (strstr (err_text (), "tricky.txt"));
  ck_assert_ptr_null (strstr (err_text (), "embedding on the"));

  touch_by ("tricky.txt", 10);
  ck_assert_int_eq (run_index ("-v", NULL), 0);
  ck_assert_ptr_nonnull (strstr (err_text (), "embedding on the GPU\n"));
  ck_assert_ptr_nonnull (strstr (err_text (), "/tricky.txt\n"));
}
END_TEST

START_TEST (test_index_long_file_is_cut)
{
  char *big = malloc (100001);
  char *log;

  memset (big, 'w', 100000);
  big[100000] = '\0';
  put ("big.txt", big);
  ck_assert_int_eq (run_index (NULL), 0);
  log = read_file (fx.log);
  ck_assert_uint_lt (strlen (log), MAX_CONTENT + 1000);
  ck_assert_uint_eq (records_for ("big.txt"), 1);
}
END_TEST

START_TEST (test_index_binary_content_is_safe)
{
  char bytes[256];

  for (int i = 0; i < 256; i++)
    bytes[i] = (char) i;
  char *p = home_file ("all-bytes.txt");
  write_file (p, bytes, sizeof bytes);
  /* Make libmagic call it text, so that its bytes reach the request.  */
  put ("latin1.txt", "caf\xe9 cr\xe8me br\xfbl\xe9\xe9 \x00 sourdough");
  ck_assert_int_eq (run_index (NULL), 0);
  ck_assert_uint_eq (record_count (), 2);
}
END_TEST

START_TEST (test_index_model_change)
{
  put ("a.txt", "alpha");
  ck_assert_int_eq (run_index (NULL), 0);
  /* A store left by an older version that used another model.  */
  char *store = read_file (fx.store);
  char *rest = strchr (store, '\n');
  char *old;
  ck_assert_int_ge (asprintf (&old, "# embed-index model=other-model "
                              "dim=8%s", rest), 0);
  write_file (fx.store, old, (size_t) -1);

  ck_assert_int_eq (run_index (NULL), 0);
  ck_assert_ptr_nonnull (strstr (err_text (), "model changed from "
                                 "other-model to " EMBED_MODEL));
  ck_assert (starts_with (read_file (fx.store),
                          "# embed-index model=" EMBED_MODEL " dim=8\n"));
  ck_assert_uint_eq (record_count (), 1);
  ck_assert_ptr_nonnull (strstr (err_text (), "embedded 1, unchanged 0"));
  ck_assert_int_eq (run_index (NULL), 0);
  ck_assert_ptr_nonnull (strstr (err_text (), "unchanged 1"));
  ck_assert_ptr_null (strstr (read_file (fx.log), "other-model"));
}
END_TEST

START_TEST (test_index_model_option_removed)
{
  ck_assert_int_eq (run_index ("-m", "other-model", NULL), EXIT_FAILURE);
  ck_assert_ptr_null (read_file (fx.store));
}
END_TEST

START_TEST (test_index_model_unavailable)
{
  setenv ("FAKE_MODEL", "fail", 1);
  put ("a.txt", "alpha");
  ck_assert_int_eq (run_index (NULL), EXIT_FAILURE);
  ck_assert_ptr_nonnull (strstr (err_text (), "cannot open the "
                                 EMBED_MODEL " model"));
  ck_assert_ptr_null (read_file (fx.store));
  ck_assert_ptr_null (read_file (fx.log));
}
END_TEST

START_TEST (test_index_gpu_by_default)
{
  put ("a.txt", "alpha");
  ck_assert_int_eq (run_index (NULL), 0);
  ck_assert (starts_with (read_file (fx.log), "open gpu=1\n----\n"));
  unlink (fx.log);
  touch_by ("a.txt", 10);
  ck_assert_int_eq (run_index ("--cpu", "-v", NULL), 0);
  ck_assert (starts_with (read_file (fx.log), "open gpu=0\n----\n"));
  ck_assert_ptr_nonnull (strstr (err_text (), "embedding on the CPU\n"));
  ck_assert_uint_eq (records_for ("a.txt"), 1);
}
END_TEST

START_TEST (test_index_removed_options)
{
  ck_assert_int_eq (run_index ("-t", "5", NULL), EXIT_FAILURE);
  ck_assert_int_eq (run_index ("-H", "http://localhost:1", NULL),
                    EXIT_FAILURE);
  ck_assert_int_eq (run_index ("--host=http://localhost:1", NULL),
                    EXIT_FAILURE);
  ck_assert_ptr_null (read_file (fx.store));
}
END_TEST

START_TEST (test_index_recovers_from_crash)
{
  char *store;

  put ("a.txt", "alpha");
  put ("b.txt", "omega");
  ck_assert_int_eq (run_index (NULL), 0);
  /* Simulate a crash in the middle of appending a record for b.  */
  store = read_file (fx.store);
  char *b = strstr (store, "/b.txt\t");
  ck_assert_ptr_nonnull (b);
  size_t eol = strchr (b, '\n') - store;
  store = realloc (store, strlen (store) + strlen (fx.home) + 64);
  strcpy (store + eol + 1, "123 4\t");
  strcat (store, fx.home);
  strcat (store, "/b.txt\t0.1 0.2");
  write_file (fx.store, store, (size_t) -1);
  touch_by ("b.txt", 10);

  ck_assert_int_eq (run_index (NULL), 0);
  ck_assert_uint_eq (records_for ("b.txt"), 1);
  ck_assert_uint_eq (record_count (), 2);
  store = read_file (fx.store);
  ck_assert_int_eq (store[strlen (store) - 1], '\n');
  ck_assert_ptr_null (strstr (store, "0.1 0.2"));
}
END_TEST

START_TEST (test_index_header_without_newline)
{
  put ("a.txt", "alpha");
  write_file (fx.store, "# embed-index model=" EMBED_MODEL " dim=8",
              (size_t) -1);
  ck_assert_msg (run_index (NULL) == 0, "stderr: %s", err_text ());
  ck_assert (starts_with (read_file (fx.store), "# embed-index "
                          "model=" EMBED_MODEL " dim=8\n"));
  ck_assert_uint_eq (records_for ("a.txt"), 1);
}
END_TEST

START_TEST (test_index_wrong_dimension_reported)
{
  put ("FAKE_DIM3.txt", "alpha");
  ck_assert_int_eq (run_index ("-v", NULL), 0);
  ck_assert_ptr_nonnull (strstr (err_text (), "model returned 3 dimensions "
                                 "instead of 8"));
  ck_assert_uint_eq (record_count (), 0);
}
END_TEST

START_TEST (test_index_unreadable_store)
{
  if (geteuid () == 0)
    return;
  put ("a.txt", "alpha");
  ck_assert_int_eq (run_index (NULL), 0);
  chmod (fx.store, 0);
  ck_assert_int_eq (run_index (NULL), EXIT_FAILURE);
  ck_assert_ptr_nonnull (strstr (err_text (), "Permission denied"));
}
END_TEST

START_TEST (test_index_read_only_store)
{
  if (geteuid () == 0)
    return;
  put ("a.txt", "alpha");
  ck_assert_int_eq (run_index (NULL), 0);
  chmod (fx.store, 0444);
  put ("b.txt", "omega");
  ck_assert_int_eq (run_index (NULL), EXIT_FAILURE);
  ck_assert_ptr_nonnull (strstr (err_text (), "Permission denied"));
  ck_assert_uint_eq (records_for ("a.txt"), 1);
}
END_TEST

START_TEST (test_index_compaction_fails_safely)
{
  char *dir;
  char *store;

  if (geteuid () == 0)
    return;
  put ("a.txt", "alpha");
  ck_assert_int_eq (run_index (NULL), 0);
  /* Records can still be appended, but no new file can be made beside
     the store to compact it into.  */
  dir = home_file (".local/share/embed");
  chmod (dir, 0555);
  put ("b.txt", "omega");
  ck_assert_int_eq (run_index (NULL), EXIT_FAILURE);
  chmod (dir, 0755);
  /* Nothing is lost: the new record was appended before.  */
  store = read_file (fx.store);
  ck_assert (starts_with (store, "# embed-index "));
  ck_assert_uint_eq (records_for ("a.txt"), 1);
  ck_assert_uint_eq (records_for ("b.txt"), 1);
  ck_assert_int_eq (run_index (NULL), 0);
  ck_assert_ptr_nonnull (strstr (err_text (), "embedded 0, unchanged 2"));
}
END_TEST

START_TEST (test_index_progress_on_terminal)
{
  int master;
  pid_t pid;
  char out[65536];
  size_t len = 0;

  for (int i = 0; i < 25; i++)
    {
      char name[32];
      snprintf (name, sizeof name, "f%02d.txt", i);
      put (name, "alpha");
    }
  pid = forkpty (&master, NULL, NULL, NULL);
  ck_assert_int_ge (pid, 0);
  if (pid == 0)
    {
      char *argv[] = { "embed-index", NULL };
      /* Only standard error is a terminal.  */
      freopen ("/dev/null", "w", stdout);
      exit (index_main (1, argv));
    }
  for (;;)
    {
      struct pollfd p = { .fd = master, .events = POLLIN };
      if (poll (&p, 1, 10000) <= 0)
        break;
      ssize_t n = read (master, out + len, sizeof out - 1 - len);
      if (n <= 0)
        break;
      len += n;
    }
  out[len] = '\0';
  waitpid (pid, NULL, 0);
  ck_assert_msg (strstr (out, "\rembedded 20, unchanged 0"), "got: %s", out);
  ck_assert_ptr_nonnull (strstr (out, "embedded 25, unchanged 0, failed 0"));
}
END_TEST

START_TEST (test_index_corrupt_store)
{
  put ("a.txt", "alpha");
  write_file (fx.store, "this is not a store\n", (size_t) -1);
  ck_assert_int_eq (run_index (NULL), EXIT_FAILURE);
  ck_assert_ptr_nonnull (strstr (err_text (), "invalid header"));
  /* It is left alone for the user to inspect.  */
  ck_assert_str_eq (read_file (fx.store), "this is not a store\n");
}
END_TEST

START_TEST (test_index_interrupted)
{
  put ("a.txt", "alpha");
  ck_assert_int_eq (run_index (NULL), 0);

  /* Make files that each take three seconds, and interrupt the run
     during the first.  */
  put ("slow1.txt", "FAKE_SLOW");
  put ("slow2.txt", "FAKE_SLOW");
  put ("slow3.txt", "FAKE_SLOW");
  unlink (home_file ("a.txt"));
  char *argv[] = { "embed-index", NULL };
  double start = now ();
  ck_assert_int_eq (run_main_signal (index_main, argv, fx.out, fx.err, 1500,
                                     SIGINT), 130);
  ck_assert_double_lt (now () - start, 5);
  ck_assert_ptr_nonnull (strstr (err_text (), "(interrupted)"));
  /* The file in progress was finished and saved.  */
  ck_assert_uint_eq (record_count (), 1);
  ck_assert_uint_eq (records_for ("a.txt"), 0);
}
END_TEST

START_TEST (test_index_interrupted_keeps_unvisited)
{
  /* Entries the walk did not reach are kept, even though it did not
     see them.  */
  put ("a/1.txt", "alpha");
  put ("z/2.txt", "omega");
  ck_assert_int_eq (run_index (NULL), 0);
  put ("a/slow.txt", "FAKE_SLOW");
  put ("a/slow2.txt", "FAKE_SLOW");
  char *argv[] = { "embed-index", NULL };
  ck_assert_int_eq (run_main_signal (index_main, argv, fx.out, fx.err, 1000,
                                     SIGTERM), 130);
  ck_assert_uint_eq (records_for ("a/1.txt"), 1);
  ck_assert_uint_eq (records_for ("z/2.txt"), 1);
}
END_TEST

START_TEST (test_index_options)
{
  char *help[] = { "embed-index", "--help", NULL };
  char *version[] = { "embed-index", "--version", NULL };
  char *bad[] = { "embed-index", "--bogus", NULL };

  ck_assert_int_eq (run_main (index_main, help, fx.out, fx.err), 0);
  ck_assert (starts_with (read_file (fx.out), "Usage: embed-index"));
  ck_assert_ptr_nonnull (strstr (read_file (fx.out), "model " EMBED_MODEL));
  ck_assert_ptr_null (strstr (read_file (fx.out), "--model"));
  ck_assert_int_eq (run_main (index_main, version, fx.out, fx.err), 0);
  ck_assert_str_eq (read_file (fx.out), "embed-index " PACKAGE_VERSION "\n");
  ck_assert_int_eq (run_main (index_main, bad, fx.out, fx.err),
                    EXIT_FAILURE);
  ck_assert_ptr_nonnull (strstr (err_text (), "--help"));
}
END_TEST

static Suite *
index_suite (void)
{
  Suite *s = suite_create ("index");
  TCase *tc;

  tc = tcase_create ("helpers");
  tcase_add_checked_fixture (tc, setup, teardown);
  tcase_set_timeout (tc, 30);
  tcase_add_test (tc, test_starts_ends_with);
  tcase_add_test (tc, test_textual_mime);
  tcase_add_test (tc, test_under_root);
  tcase_add_test (tc, test_strip_markup);
  tcase_add_test (tc, test_read_head);
  tcase_add_test (tc, test_read_head_missing_and_unreadable);
  tcase_add_test (tc, test_run_capture_output);
  tcase_add_test (tc, test_run_capture_ignores_stderr);
  tcase_add_test (tc, test_run_capture_limits_output);
  tcase_add_test (tc, test_run_capture_time_limit);
  tcase_add_test (tc, test_run_capture_stdin_is_empty);
  tcase_add_test (tc, test_run_capture_missing_program);
  tcase_add_test (tc, test_run_capture_many_calls);
  suite_add_tcase (s, tc);

  tc = tcase_create ("extract");
  tcase_add_checked_fixture (tc, setup, teardown);
  tcase_set_timeout (tc, 30);
  tcase_add_test (tc, test_extract_text);
  tcase_add_test (tc, test_extract_html_and_xml);
  tcase_add_test (tc, test_extract_json_kept_verbatim);
  tcase_add_test (tc, test_extract_unknown_type);
  tcase_add_test (tc, test_extract_pdf);
  tcase_add_test (tc, test_extract_broken_pdf);
  tcase_add_test (tc, test_extract_image_ocr);
  tcase_add_test (tc, test_extract_audio_tags);
  tcase_add_test (tc, test_extract_video_streams);
  tcase_add_test (tc, test_extract_docx);
  tcase_add_test (tc, test_extract_odt);
  tcase_add_test (tc, test_extract_pptx_all_slides);
  tcase_add_test (tc, test_extract_xlsx_strings);
  tcase_add_test (tc, test_describe_text_file);
  tcase_add_test (tc, test_describe_without_content);
  tcase_add_test (tc, test_describe_outside_home);
  tcase_add_test (tc, test_describe_home_prefix_sibling);
  tcase_add_test (tc, test_describe_unreadable_file);
  tcase_add_test (tc, test_describe_pdf_type);
  suite_add_tcase (s, tc);

  tc = tcase_create ("store");
  tcase_add_checked_fixture (tc, setup, teardown);
  tcase_add_test (tc, test_keep_push_grows);
  tcase_add_test (tc, test_write_record_format);
  tcase_add_test (tc, test_write_record_full_disk);
  tcase_add_test (tc, test_load_store_missing);
  tcase_add_exit_test (tc, test_load_store_bad_header, EXIT_FAILURE);
  tcase_add_exit_test (tc, test_load_store_empty_file, EXIT_FAILURE);
  tcase_add_test (tc, test_load_store_header_only);
  tcase_add_test (tc, test_load_store_dedupes_and_skips_junk);
  tcase_add_test (tc, test_load_store_many);
  tcase_add_test (tc, test_compact_store);
  suite_add_tcase (s, tc);

  tc = tcase_create ("program");
  tcase_add_checked_fixture (tc, setup, teardown);
  tcase_set_timeout (tc, 60);
  tcase_add_test (tc, test_index_basic);
  tcase_add_test (tc, test_index_vectors_are_unit_length);
  tcase_add_test (tc, test_index_hidden_with_all);
  tcase_add_test (tc, test_index_rerun_skips_unchanged);
  tcase_add_test (tc, test_index_modified_file);
  tcase_add_test (tc, test_index_same_mtime_new_size);
  tcase_add_test (tc, test_index_deleted_file);
  tcase_add_test (tc, test_index_subdirectory_keeps_others);
  tcase_add_test (tc, test_index_relative_directory);
  tcase_add_test (tc, test_index_several_directories);
  tcase_add_test (tc, test_index_missing_directory);
  tcase_add_test (tc, test_index_odd_file_names);
  tcase_add_test (tc, test_index_falls_back_to_name);
  tcase_add_test (tc, test_index_falls_back_on_wrong_dimension);
  tcase_add_test (tc, test_index_unembeddable_file);
  tcase_add_test (tc, test_index_quiet_unless_verbose);
  tcase_add_test (tc, test_index_long_file_is_cut);
  tcase_add_test (tc, test_index_binary_content_is_safe);
  tcase_add_test (tc, test_index_model_change);
  tcase_add_test (tc, test_index_model_option_removed);
  tcase_add_test (tc, test_index_model_unavailable);
  tcase_add_test (tc, test_index_gpu_by_default);
  tcase_add_test (tc, test_index_removed_options);
  tcase_add_test (tc, test_index_recovers_from_crash);
  tcase_add_test (tc, test_index_header_without_newline);
  tcase_add_test (tc, test_index_wrong_dimension_reported);
  tcase_add_test (tc, test_index_unreadable_store);
  tcase_add_test (tc, test_index_read_only_store);
  tcase_add_test (tc, test_index_compaction_fails_safely);
  tcase_add_test (tc, test_index_progress_on_terminal);
  tcase_add_test (tc, test_index_corrupt_store);
  tcase_add_test (tc, test_index_interrupted);
  tcase_add_test (tc, test_index_interrupted_keeps_unvisited);
  tcase_add_test (tc, test_index_options);
  suite_add_tcase (s, tc);

  return s;
}

int
main (void)
{
  SRunner *sr = srunner_create (index_suite ());
  int failed;

  srunner_run_all (sr, CK_NORMAL);
  failed = srunner_ntests_failed (sr);
  srunner_free (sr);
  return failed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
