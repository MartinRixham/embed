/* embed-index -- embed every file under the home directory.
   Copyright (C) 2026 Martin Rixham

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   Each file is turned into a short text description (its name,
   location, type and, where possible, its content: plain text, PDF
   text, office document text, OCR of images, or audio/video metadata)
   and that text is embedded by a model running on the GPU in a private
   Ollama server that this program starts and stops itself.  The results are
   appended to ~/.local/share/embed/embeddings.txt, one file per line.
   Files whose size and modification time are unchanged since the last
   run are skipped, so the program can be rerun cheaply.  */

#include "common.h"
#include "server.h"

#include <errno.h>
#include <error.h>
#include <fcntl.h>
#include <fts.h>
#include <getopt.h>
#include <magic.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

/* How much file content to feed the model, in bytes.  */
#define MAX_CONTENT 1500

/* Time limit for embedding just a file's name and type, in seconds.  */
#define FALLBACK_TIMEOUT 2

/* An entry loaded from an existing store.  */
struct entry
{
  char *path;
  long long mtime;
  long long size;
  size_t line;                  /* Line number of the newest record.  */
  bool seen;                    /* Found during this run's walk.  */
};

static struct entry *entries;
static size_t n_entries;

/* KEEP[N] is nonzero if line N of the store should survive
   compaction.  */
static char *keep;
static size_t n_lines, keep_cap;

/* Whether the store ends in a partial line, left by a crash.  */
static bool store_unterminated;

static volatile sig_atomic_t interrupted;
static bool verbose;
static bool include_hidden;
static long helper_timeout = 2;
static long embed_timeout = 4;

static struct option const long_options[] = {
  {"all", no_argument, NULL, 'a'},
  {"host", required_argument, NULL, 'H'},
  {"model", required_argument, NULL, 'm'},
  {"timeout", required_argument, NULL, 't'},
  {"verbose", no_argument, NULL, 'v'},
  {"help", no_argument, NULL, 'h'},
  {"version", no_argument, NULL, 'V'},
  {NULL, 0, NULL, 0}
};

static _Noreturn void
usage (int status)
{
  if (status != EXIT_SUCCESS)
    {
      fprintf (stderr, "Try 'embed-index --help' for more information.\n");
      exit (status);
    }
  printf ("Usage: embed-index [OPTION]... [DIRECTORY]...\n"
          "Embed every file under each DIRECTORY (default: $HOME) with a "
          "local model.\n\n"
          "  -a, --all            include hidden files and directories\n"
          "  -H, --host=URL       use the Ollama server at URL instead of "
          "starting\n"
          "                       a private one on the GPU\n"
          "  -m, --model=NAME     embedding model (default: the model of "
          "the\n"
          "                       existing store, or %s)\n"
          "  -t, --timeout=SECS   time limit for each embedding (default "
          "%ld)\n"
          "  -v, --verbose        print each file as it is embedded\n"
          "      --help           display this help and exit\n"
          "      --version        output version information and exit\n\n"
          "Embeddings are stored in ~/.local/share/embed/embeddings.txt.\n",
          DEFAULT_MODEL, embed_timeout);
  exit (status);
}

static void
handle_signal (int sig)
{
  (void) sig;
  interrupted = 1;
}

static void
keep_push (char value)
{
  if (n_lines == keep_cap)
    {
      keep_cap = keep_cap ? 2 * keep_cap : 4096;
      keep = xrealloc (keep, keep_cap);
    }
  keep[n_lines++] = value;
}

static int
compare_entries (const void *a, const void *b)
{
  const struct entry *x = a, *y = b;
  int c = strcmp (x->path, y->path);

  if (c != 0)
    return c;
  /* Older records first, so that deduplication keeps the newest.  */
  return (x->line > y->line) - (x->line < y->line);
}

static struct entry *
find_entry (const char *path)
{
  size_t lo = 0, hi = n_entries;

  while (lo < hi)
    {
      size_t mid = lo + (hi - lo) / 2;
      int c = strcmp (path, entries[mid].path);

      if (c == 0)
        return &entries[mid];
      if (c < 0)
        hi = mid;
      else
        lo = mid + 1;
    }
  return NULL;
}

/* Load the store at PATH.  Return true if it exists and was written by
   a model; store that model's name in MODEL.  */
static bool
load_store (const char *path, char *model, size_t model_size)
{
  FILE *f = fopen (path, "r");
  char *line = NULL;
  size_t line_size = 0, cap = 0, dim;
  ssize_t len;

  if (!f)
    {
      if (errno != ENOENT)
        error (EXIT_FAILURE, errno, "%s", path);
      return false;
    }

  len = getline (&line, &line_size, f);
  if (len <= 0 || !parse_header (line, model, model_size, &dim))
    error (EXIT_FAILURE, 0, "%s: missing or invalid header", path);
  keep_push (1);
  store_unterminated = line[len - 1] != '\n';

  while ((len = getline (&line, &line_size, f)) > 0)
    {
      long long mtime, size;
      char *p = line, *tab1, *tab2;

      keep_push (0);
      store_unterminated = line[len - 1] != '\n';
      if (store_unterminated)
        continue;                       /* Truncated by a crash.  */
      if (sscanf (p, "%lld %lld", &mtime, &size) != 2
          || !(tab1 = strchr (p, '\t'))
          || !(tab2 = strchr (tab1 + 1, '\t')))
        continue;

      if (n_entries == cap)
        {
          cap = cap ? 2 * cap : 4096;
          entries = xrealloc (entries, cap * sizeof *entries);
        }
      *tab2 = '\0';
      entries[n_entries].path = xstrdup (tab1 + 1);
      entries[n_entries].mtime = mtime;
      entries[n_entries].size = size;
      entries[n_entries].line = n_lines - 1;
      entries[n_entries].seen = false;
      n_entries++;
    }
  free (line);
  fclose (f);

  /* Sort by path and keep only the newest record for each.  */
  if (n_entries > 0)
    qsort (entries, n_entries, sizeof *entries, compare_entries);
  size_t out = 0;
  for (size_t i = 0; i < n_entries; i++)
    {
      if (out > 0 && strcmp (entries[out - 1].path, entries[i].path) == 0)
        {
          free (entries[out - 1].path);
          entries[out - 1] = entries[i];
        }
      else
        entries[out++] = entries[i];
    }
  n_entries = out;
  for (size_t i = 0; i < n_entries; i++)
    keep[entries[i].line] = 1;
  return true;
}

/* Run ARGV under timeout(1) and append up to MAX bytes of its standard
   output to OUT.  */
static void
run_capture (const char *const *argv, struct buf *out, size_t max)
{
  const char *args[16];
  char limit[32];
  int fds[2];
  pid_t pid;
  size_t i, got = 0;

  snprintf (limit, sizeof limit, "%ld", helper_timeout);
  args[0] = "timeout";
  args[1] = "-k";
  args[2] = "1";
  args[3] = limit;
  for (i = 0; argv[i] && i < 11; i++)
    args[i + 4] = argv[i];
  args[i + 4] = NULL;

  if (pipe (fds) < 0)
    error (EXIT_FAILURE, errno, "pipe");
  pid = fork ();
  if (pid < 0)
    error (EXIT_FAILURE, errno, "fork");
  if (pid == 0)
    {
      int null = open ("/dev/null", O_RDWR);
      dup2 (null, STDIN_FILENO);
      dup2 (null, STDERR_FILENO);
      dup2 (fds[1], STDOUT_FILENO);
      close (fds[0]);
      close (fds[1]);
      execvp (args[0], (char *const *) args);
      _exit (127);
    }

  close (fds[1]);
  while (got < max)
    {
      char chunk[4096];
      size_t want = max - got < sizeof chunk ? max - got : sizeof chunk;
      ssize_t n = read (fds[0], chunk, want);

      if (n < 0 && errno == EINTR && !interrupted)
        continue;
      if (n <= 0)
        break;
      buf_append (out, chunk, n);
      got += n;
    }
  /* Closing the pipe early makes a chatty helper die of SIGPIPE.  */
  close (fds[0]);
  while (waitpid (pid, NULL, 0) < 0 && errno == EINTR)
    ;
}

/* Append up to MAX bytes from the start of FILE to OUT.  */
static void
read_head (const char *file, struct buf *out, size_t max)
{
  char *chunk = xmalloc (max);
  int fd = open (file, O_RDONLY | O_NOCTTY | O_NONBLOCK | O_NOATIME);
  ssize_t n;

  if (fd < 0)
    fd = open (file, O_RDONLY | O_NOCTTY | O_NONBLOCK);
  if (fd >= 0)
    {
      n = read (fd, chunk, max);
      if (n > 0)
        buf_append (out, chunk, n);
      close (fd);
    }
  free (chunk);
}

/* Replace XML markup in B with spaces and squeeze runs of white space.  */
static void
strip_markup (struct buf *b)
{
  size_t out = 0;
  bool in_tag = false, space = true;

  for (size_t i = 0; i < b->len; i++)
    {
      char c = b->data[i];

      if (c == '<')
        in_tag = true;
      if (in_tag || c == ' ' || c == '\n' || c == '\t' || c == '\r')
        {
          if (!space)
            b->data[out++] = ' ';
          space = true;
        }
      else
        {
          b->data[out++] = c;
          space = false;
        }
      if (c == '>')
        in_tag = false;
    }
  b->len = out;
  b->data[out] = '\0';
}

static bool
starts_with (const char *s, const char *prefix)
{
  return strncmp (s, prefix, strlen (prefix)) == 0;
}

static bool
ends_with (const char *s, const char *suffix)
{
  size_t n = strlen (s), m = strlen (suffix);
  return n >= m && strcmp (s + n - m, suffix) == 0;
}

static bool
textual_mime (const char *mime)
{
  static const char *const types[] = {
    "application/json", "application/javascript", "application/xml",
    "application/x-shellscript", "application/x-sh", "application/sql",
    "application/x-subrip", "application/toml", "application/x-ndjson",
    "application/x-wine-extension-ini", "message/rfc822", NULL
  };

  if (starts_with (mime, "text/") || ends_with (mime, "+xml")
      || ends_with (mime, "+json"))
    return true;
  for (size_t i = 0; types[i]; i++)
    if (strcmp (mime, types[i]) == 0)
      return true;
  return false;
}

/* Append a textual rendering of the content of FILE, of type MIME, to
   OUT.  */
static void
extract_content (const char *file, const char *mime, struct buf *out)
{
  struct buf tmp;

  if (textual_mime (mime))
    {
      read_head (file, out, MAX_CONTENT);
      if (strcmp (mime, "text/html") == 0 || ends_with (mime, "xml"))
        strip_markup (out);
    }
  else if (strcmp (mime, "application/pdf") == 0)
    {
      const char *argv[] = { "pdftotext", "-q", "-l", "2", "-enc", "UTF-8",
                             file, "-", NULL };
      run_capture (argv, out, MAX_CONTENT);
    }
  else if (starts_with (mime, "image/"))
    {
      const char *argv[] = { "tesseract", file, "stdout", "-l", "eng",
                             NULL };
      run_capture (argv, out, MAX_CONTENT);
    }
  else if (starts_with (mime, "audio/") || starts_with (mime, "video/"))
    {
      const char *argv[] = { "ffprobe", "-v", "quiet", "-show_entries",
                             ("format=duration:format_tags"
                              ":stream=codec_name,width,height"),
                             "-of", "default=nw=1", file, NULL };
      run_capture (argv, out, MAX_CONTENT);
    }
  else if (starts_with (mime, "application/vnd.openxmlformats")
           || starts_with (mime, "application/vnd.oasis.opendocument"))
    {
      const char *member = "content.xml";

      if (strstr (mime, "wordprocessingml"))
        member = "word/document.xml";
      else if (strstr (mime, "presentationml"))
        member = "ppt/slides/slide*.xml";
      else if (strstr (mime, "spreadsheetml"))
        member = "xl/sharedStrings.xml";

      const char *argv[] = { "unzip", "-p", file, member, NULL };
      buf_init (&tmp);
      run_capture (argv, &tmp, 4 * MAX_CONTENT);
      strip_markup (&tmp);
      buf_append (out, tmp.data, tmp.len < MAX_CONTENT ? tmp.len
                  : MAX_CONTENT);
      buf_free (&tmp);
    }
}

/* Build the text that represents FILE for the model, in OUT.  HOME is
   used to shorten the displayed location.  */
static void
describe_file (magic_t magic_mime, magic_t magic_desc, const char *file,
               const char *home, const char *model, bool with_content,
               struct buf *out)
{
  const char *name = strrchr (file, '/');
  const char *mime = magic_file (magic_mime, file);
  const char *desc = magic_file (magic_desc, file);
  size_t home_len = strlen (home);

  name = name ? name + 1 : file;
  buf_puts (out, document_prefix (model));
  buf_puts (out, "File name: ");
  buf_puts (out, name);
  buf_puts (out, "\nLocation: ");
  if (strncmp (file, home, home_len) == 0 && file[home_len] == '/')
    {
      buf_puts (out, "~");
      buf_append (out, file + home_len, name - file - home_len - 1);
    }
  else
    buf_append (out, file, name - file - 1);
  buf_puts (out, "\nType: ");
  buf_puts (out, desc ? desc : "unknown");
  buf_puts (out, "\n\n");

  if (with_content && mime)
    {
      mime = strcpy (xmalloc (strlen (mime) + 1), mime);
      extract_content (file, mime, out);
      free ((char *) mime);
    }
}

/* Embed TEXT like embed_text, but fail unless the vector has DIM
   dimensions, so that the store stays consistent.  */
static bool
embed_checked (struct embedder *e, const char *text, long timeout,
               size_t dim, float **vec)
{
  size_t d;

  if (!embed_text (e, text, timeout, vec, &d))
    return false;
  if (d == dim)
    return true;
  if (embed_report_errors)
    error (0, 0, "model returned %zu dimensions instead of %zu", d, dim);
  free (*vec);
  return false;
}

/* Write a store record for FILE to F and return true on success.  */
static bool
write_record (FILE *f, const char *file, const struct stat *st,
              const float *vec, size_t dim)
{
  fprintf (f, "%lld %lld\t%s\t", (long long) st->st_mtime,
           (long long) st->st_size, file);
  for (size_t i = 0; i < dim; i++)
    fprintf (f, i ? " %.4g" : "%.4g", vec[i]);
  putc ('\n', f);
  return fflush (f) == 0 && !ferror (f);
}

static bool
under_root (const char *path, char *const *roots)
{
  for (size_t i = 0; roots[i]; i++)
    {
      size_t n = strlen (roots[i]);
      if (strncmp (path, roots[i], n) == 0
          && (path[n] == '/' || path[n] == '\0'
              || (n > 0 && roots[i][n - 1] == '/')))
        return true;
    }
  return false;
}

/* Rewrite the store at PATH keeping only lines marked in KEEP.  */
static void
compact_store (const char *path)
{
  struct buf tmp_path;
  FILE *in, *out;
  char *line = NULL;
  size_t line_size = 0, n = 0;
  ssize_t len;

  buf_init (&tmp_path);
  buf_puts (&tmp_path, path);
  buf_puts (&tmp_path, ".tmp");

  in = fopen (path, "r");
  if (!in)
    error (EXIT_FAILURE, errno, "%s", path);
  out = fopen (tmp_path.data, "w");
  if (!out)
    error (EXIT_FAILURE, errno, "%s", tmp_path.data);

  while ((len = getline (&line, &line_size, in)) > 0)
    {
      if (n < n_lines && keep[n] && line[len - 1] == '\n')
        fwrite (line, 1, len, out);
      n++;
    }
  free (line);
  fclose (in);
  if (fclose (out) != 0)
    error (EXIT_FAILURE, errno, "%s", tmp_path.data);
  if (rename (tmp_path.data, path) != 0)
    error (EXIT_FAILURE, errno, "cannot replace %s", path);
  buf_free (&tmp_path);
}

int
main (int argc, char **argv)
{
  const char *host = NULL;
  struct server srv = { 0 };
  const char *model_opt = NULL;
  char store_model[256];
  char *store = store_path ();
  char *store_dir;
  const char *home = getenv ("HOME");
  char *default_roots[2];
  char **roots;
  struct embedder emb;
  magic_t magic_mime, magic_desc;
  FILE *out;
  FTS *fts;
  FTSENT *ent;
  size_t dim = 0, n_new = 0, n_same = 0, n_failed = 0;
  float *vec;
  bool have_store;
  int c;

  while ((c = getopt_long (argc, argv, "aH:m:t:v", long_options, NULL))
         != -1)
    switch (c)
      {
      case 'a':
        include_hidden = true;
        break;
      case 'H':
        host = optarg;
        break;
      case 'm':
        model_opt = optarg;
        break;
      case 't':
        if (!parse_positive (optarg, 3600, &embed_timeout))
          error (EXIT_FAILURE, 0, "invalid timeout: %s", optarg);
        break;
      case 'v':
        verbose = true;
        break;
      case 'h':
        usage (EXIT_SUCCESS);
      case 'V':
        printf ("embed-index %s\n", PACKAGE_VERSION);
        return EXIT_SUCCESS;
      default:
        usage (EXIT_FAILURE);
      }

  if (optind < argc)
    {
      roots = xmalloc ((argc - optind + 1) * sizeof *roots);
      for (int i = optind; i < argc; i++)
        if (!(roots[i - optind] = realpath (argv[i], NULL)))
          error (EXIT_FAILURE, errno, "%s", argv[i]);
      roots[argc - optind] = NULL;
    }
  else
    {
      default_roots[0] = (char *) home;
      default_roots[1] = NULL;
      roots = default_roots;
    }

  store_dir = xstrdup (store);
  *strrchr (store_dir, '/') = '\0';

  have_store = load_store (store, store_model, sizeof store_model);
  if (have_store && model_opt && strcmp (model_opt, store_model) != 0)
    {
      error (0, 0, "model changed from %s to %s; reindexing everything",
             store_model, model_opt);
      have_store = false;
      for (size_t i = 0; i < n_entries; i++)
        free (entries[i].path);
      n_entries = 0;
      n_lines = 0;
    }
  if (!have_store)
    snprintf (store_model, sizeof store_model, "%s",
              model_opt ? model_opt : DEFAULT_MODEL);

  if (!host)
    {
      server_start (&srv, true);
      host = srv.url;
    }
  embedder_init (&emb, host, store_model);

  /* Warm up the model, which may take a while to load, and learn its
     dimension.  */
  if (!embed_text (&emb, "warm up", 120, &vec, &dim))
    error (EXIT_FAILURE, 0, "cannot load model %s at %s; has it been "
           "pulled with 'ollama pull %s'?", store_model, host, store_model);
  free (vec);

  if (have_store)
    {
      out = fopen (store, "a");
      /* End a partial line, so that new records do not run into it.  */
      if (out && store_unterminated)
        putc ('\n', out);
    }
  else
    {
      make_parent_dirs (store);
      out = fopen (store, "w");
      if (out)
        {
          fprintf (out, "# embed-index model=%s dim=%zu\n", store_model,
                   dim);
          keep_push (1);
        }
    }
  if (!out)
    error (EXIT_FAILURE, errno, "%s", store);

  magic_mime = magic_open (MAGIC_MIME_TYPE | MAGIC_ERROR);
  magic_desc = magic_open (MAGIC_ERROR);
  if (!magic_mime || !magic_desc || magic_load (magic_mime, NULL) != 0
      || magic_load (magic_desc, NULL) != 0)
    error (EXIT_FAILURE, 0, "cannot load libmagic database");

  struct sigaction sa = { .sa_handler = handle_signal };
  sigaction (SIGINT, &sa, NULL);
  sigaction (SIGTERM, &sa, NULL);
  signal (SIGPIPE, SIG_IGN);

  embed_report_errors = verbose;
  fts = fts_open (roots, FTS_PHYSICAL | FTS_NOCHDIR, NULL);
  if (!fts)
    error (EXIT_FAILURE, errno, "cannot walk directories");

  while (!interrupted && (ent = fts_read (fts)))
    {
      struct entry *e;
      struct buf text;

      if (ent->fts_level > 0 && ent->fts_name[0] == '.' && !include_hidden)
        {
          if (ent->fts_info == FTS_D)
            fts_set (fts, ent, FTS_SKIP);
          continue;
        }
      if (ent->fts_info == FTS_D && strcmp (ent->fts_path, store_dir) == 0)
        {
          fts_set (fts, ent, FTS_SKIP);
          continue;
        }
      if (ent->fts_info != FTS_F || ent->fts_statp->st_size == 0
          || strpbrk (ent->fts_path, "\t\n"))
        continue;

      e = find_entry (ent->fts_path);
      if (e)
        {
          e->seen = true;
          if (e->mtime == ent->fts_statp->st_mtime
              && e->size == ent->fts_statp->st_size && dim > 0)
            {
              n_same++;
              continue;
            }
        }

      if (verbose)
        fprintf (stderr, "%s\n", ent->fts_path);

      buf_init (&text);
      describe_file (magic_mime, magic_desc, ent->fts_path, home,
                     store_model, true, &text);
      bool ok = embed_checked (&emb, text.data, embed_timeout, dim, &vec);
      if (!ok && !interrupted)
        {
          /* Fall back to embedding just the name and type.  */
          text.len = 0;
          describe_file (magic_mime, magic_desc, ent->fts_path, home,
                         store_model, false, &text);
          ok = embed_checked (&emb, text.data, FALLBACK_TIMEOUT, dim, &vec);
        }
      buf_free (&text);

      if (!ok)
        {
          n_failed++;
          error (0, 0, "%s: could not embed", ent->fts_path);
          continue;
        }
      if (!write_record (out, ent->fts_path, ent->fts_statp, vec, dim))
        error (EXIT_FAILURE, errno, "%s", store);
      free (vec);

      if (e)
        {
          keep[e->line] = 0;
          e->line = n_lines;
        }
      keep_push (1);
      n_new++;

      if (!verbose && isatty (STDERR_FILENO) && n_new % 10 == 0)
        fprintf (stderr, "\rembedded %zu, unchanged %zu", n_new, n_same);
    }
  if (!verbose && isatty (STDERR_FILENO))
    fputc ('\n', stderr);
  fts_close (fts);
  embedder_free (&emb);
  server_stop (&srv);
  if (fclose (out) != 0)
    error (EXIT_FAILURE, errno, "%s", store);

  /* Forget files that have vanished, unless the walk was cut short and
     so may simply not have reached them.  */
  for (size_t i = 0; i < n_entries; i++)
    {
      struct stat st;
      if (!entries[i].seen
          && ((!interrupted && under_root (entries[i].path, roots))
              || lstat (entries[i].path, &st) != 0))
        keep[entries[i].line] = 0;
    }
  compact_store (store);

  fprintf (stderr, "%s: embedded %zu, unchanged %zu, failed %zu%s\n",
           store, n_new, n_same, n_failed,
           interrupted ? " (interrupted)" : "");

  magic_close (magic_mime);
  magic_close (magic_desc);
  return interrupted ? 130 : EXIT_SUCCESS;
}
