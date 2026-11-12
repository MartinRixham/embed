/* embed-index -- embed every file under the home directory.
   Copyright (C) 2026 Martin Rixham

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   Each file is turned into a short text description (its name,
   location, type and, where possible, its content: plain text, PDF
   text, office document text, OCR of images, or audio/video metadata;
   see describe.c) and that text is embedded by nomic-embed-text,
   running on the GPU if there is a usable one.  Images and videos are also embedded by what
   they show: nomic-embed-vision embeds an image, or a few frames of a
   video, into the same space as the text, from pixels that ffmpeg
   decodes.  Speech in audio and video files is transcribed by Whisper,
   from a few windows spread through them, and the transcript embedded
   with the rest of the text.  The results are appended to
   ~/.local/share/embed/embeddings.txt, one file per line:

     MTIME SIZE VERSION <tab> PATH <tab> TEXT-VECTOR <tab> PICTURE-VECTOR

   where the picture vector is "-" for a file that has none.  The
   version of the record format is always 1 for now; it is written
   but not yet read.

   A long file is also embedded in chunks: passages spread through its
   text, each embedded on its own, so that a search can find it by what
   one part of it says.  A recording's chunks are the transcript of
   what is said in it, which its description leaves out.  A search scores a file by the best of its
   vectors.  There are many chunks, so they are embedded a few at a
   time, after the files that are new or changed, for as long as
   --chunk-time allows, each run carrying on where the last left off
   and starting with the files that have fewest so far.  The first
   chunks of a file are spread through the whole of it, and later ones
   fall between them.  They are kept in
   ~/.local/share/embed/chunks.txt, one chunk per line:

     MTIME SIZE I K <tab> PATH <tab> VECTOR

   for the I'th chunk of the K to embed of the file as it was at MTIME
   and SIZE, with its vector stored compactly by write_q8_vector.  A
   file that needs no chunks has the single line I = K = 0 and VECTOR
   "-", so that it is not looked at again until it changes.
   Files whose size and modification time are unchanged since the last
   run are skipped, so the program can be rerun cheaply.  Only one run
   at a time may work on the store, since each rewrites it at the end
   without what another has added meanwhile, so a run takes a lock on
   ~/.local/share/embed/index.lock and gives up if another holds it.  */

#include "common.h"
#include "describe.h"
#include "model.h"

#include <errno.h>
#include <error.h>
#include <fcntl.h>
#include <fts.h>
#include <getopt.h>
#include <magic.h>
#include <signal.h>
#include <stdio.h>
#include <sys/file.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

/* The version of the record format; see above.  */
#define VERSION 1

/* An entry loaded from an existing store.  */
struct entry
{
  char *path;
  long long mtime;
  long long size;
  size_t line;                  /* Line number of the newest record.  */
  bool seen;                    /* Found during this run's walk.  */
  bool changed;                 /* Embedded afresh during this run.  */
  bool chunks_reset;            /* Its chunks are being redone.  */
  long chunk_total;             /* Chunks to embed, or -1 if unknown.  */
  size_t chunk_done;            /* Chunks embedded so far.  */
};

/* A file whose chunks may still need embedding.  */
struct candidate
{
  char *path;
  long long mtime;
  long long size;
  struct entry *e;              /* Its entry in the old store, if any.  */
  long total;                   /* Chunks to embed, or -1 if unknown.  */
  size_t done;                  /* Chunks embedded so far.  */
  bool skip;                    /* Changed or unreadable: leave it.  */
};

/* How many chunks of one file to embed before turning to the next.  */
#define CHUNK_BATCH 8

/* How long a run may spend on chunks by default, in seconds.  */
#define DEFAULT_CHUNK_TIME 600

static struct candidate *candidates;
static size_t n_candidates, candidates_cap;

/* An owner of a chunk line that is no entry: a line written during
   this run, or one to drop.  */
#define CHUNK_NEW ((size_t) -1)
#define CHUNK_DROP ((size_t) -2)

/* For each line of the chunk store, the entry whose chunk it is, or
   CHUNK_NEW or CHUNK_DROP.  The header is CHUNK_NEW.  */
static size_t *chunk_owner;
static size_t n_chunk_lines, chunk_owner_cap;
static bool chunks_unterminated;

static struct entry *entries;
static size_t n_entries;

/* KEEP[N] is nonzero if line N of the store should survive
   compaction.  */
static char *keep;
static size_t n_lines, keep_cap;

/* Whether the store ends in a partial line, left by a crash.  */
static bool store_unterminated;

static bool verbose;
static bool include_hidden;
static long chunk_time = DEFAULT_CHUNK_TIME;

static struct option const long_options[] = {
  {"all", no_argument, NULL, 'a'},
  {"cpu", no_argument, NULL, 'c'},
  {"chunk-time", required_argument, NULL, 't'},
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
          "Embed every file under each DIRECTORY (default: $HOME) with "
          "the local\nmodel %s, which is downloaded if it is missing.\n"
          "Images and videos are also embedded by what they show, with "
          "%s,\nand speech in audio and video is transcribed by %s.\n\n"
          "  -a, --all            include hidden files and directories\n"
          "      --chunk-time=SECONDS  spend at most SECONDS embedding "
          "passages of\n"
          "                       long files, after new and changed "
          "files (default %d;\n"
          "                       0 for none)\n"
          "      --cpu            run the model on the CPU even if there is "
          "a GPU\n"
          "  -v, --verbose        print each file as it is embedded\n"
          "      --help           display this help and exit\n"
          "      --version        output version information and exit\n\n"
          "Embeddings are stored in ~/.local/share/embed/embeddings.txt,\n"
          "and those of passages in ~/.local/share/embed/chunks.txt.\n",
          EMBED_MODEL, VISION_MODEL, SPEECH_MODEL, DEFAULT_CHUNK_TIME);
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
      entries[n_entries].changed = false;
      entries[n_entries].chunks_reset = false;
      entries[n_entries].chunk_total = -1;
      entries[n_entries].chunk_done = 0;
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

static void
chunk_owner_push (size_t owner)
{
  if (n_chunk_lines == chunk_owner_cap)
    {
      chunk_owner_cap = chunk_owner_cap ? 2 * chunk_owner_cap : 4096;
      chunk_owner = xrealloc (chunk_owner,
                              chunk_owner_cap * sizeof *chunk_owner);
    }
  chunk_owner[n_chunk_lines++] = owner;
}

/* Load the chunk store at PATH, for a store of vectors of DIM floats,
   and note against each entry how far its chunks have got.  Return
   false if there is no usable chunk store, so that it must be begun
   afresh.  */
static bool
load_chunks (const char *path, size_t dim)
{
  FILE *f = fopen (path, "r");
  char *line = NULL, model[256];
  size_t line_size = 0, d;
  ssize_t len;
  bool ok;

  n_chunk_lines = 0;
  if (!f)
    {
      if (errno != ENOENT)
        error (EXIT_FAILURE, errno, "%s", path);
      return false;
    }
  len = getline (&line, &line_size, f);
  ok = len > 0 && line[len - 1] == '\n'
    && parse_header (line, model, sizeof model, &d)
    && strcmp (model, EMBED_MODEL) == 0 && d == dim;
  if (!ok)
    {
      free (line);
      fclose (f);
      return false;
    }
  chunk_owner_push (CHUNK_NEW);
  chunks_unterminated = false;

  while ((len = getline (&line, &line_size, f)) > 0)
    {
      long long mtime, size;
      long i, k;
      char *tab1, *tab2;
      struct entry *e;

      chunk_owner_push (CHUNK_DROP);
      chunks_unterminated = line[len - 1] != '\n';
      if (chunks_unterminated
          || sscanf (line, "%lld %lld %ld %ld", &mtime, &size, &i, &k) != 4
          || !(tab1 = strchr (line, '\t'))
          || !(tab2 = strchr (tab1 + 1, '\t')))
        continue;
      *tab2 = '\0';
      e = find_entry (tab1 + 1);
      if (!e || e->mtime != mtime || e->size != size || k < 0
          || (e->chunk_total >= 0 && e->chunk_total != k))
        continue;
      chunk_owner[n_chunk_lines - 1] = e - entries;
      e->chunk_total = k;
      if (tab2[1] != '-')
        e->chunk_done++;
    }
  free (line);
  fclose (f);
  return true;
}

/* Write a store record for FILE to F and return
   true on success.  VEC is the embedding of its text, and PICTURE that
   of what it shows, if anything.  */
static bool
write_record (FILE *f, const char *file, const struct stat *st,
              const float *vec, const float *picture, size_t dim)
{
  fprintf (f, "%lld %lld %d\t%s\t", (long long) st->st_mtime,
           (long long) st->st_size, VERSION, file);
  write_hex_vector (f, vec, dim);
  putc ('\t', f);
  if (picture)
    write_hex_vector (f, picture, dim);
  else
    putc ('-', f);
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

/* Rewrite the file at PATH keeping only the complete lines among its
   first N for which KEEP_LINE is nonzero.  */
static void
compact_file (const char *path, const char *keep_line, size_t n_keep)
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
      if (n < n_keep && keep_line[n] && line[len - 1] == '\n')
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

/* Rewrite the store at PATH keeping only lines marked in KEEP.  */
static void
compact_store (const char *path)
{
  compact_file (path, keep, n_lines);
}

/* Rewrite the chunk store at PATH keeping only the chunks of files as
   they are now in the store.  */
static void
compact_chunks (const char *path)
{
  char *k = xmalloc (n_chunk_lines + 1);

  for (size_t i = 0; i < n_chunk_lines; i++)
    {
      size_t o = chunk_owner[i];

      k[i] = o == CHUNK_NEW
        || (o != CHUNK_DROP && keep[entries[o].line]
            && !entries[o].changed && !entries[o].chunks_reset);
    }
  compact_file (path, k, n_chunk_lines);
  free (k);
}

/* Note that the chunks of the file at PATH, of the given MTIME and
   SIZE, whose entry in the old store is E, if any, may still need
   embedding.  */
static void
add_candidate (const char *path, long long mtime, long long size,
               struct entry *e, long total, size_t done)
{
  struct candidate *c;

  if (n_candidates == candidates_cap)
    {
      candidates_cap = candidates_cap ? 2 * candidates_cap : 1024;
      candidates = xrealloc (candidates,
                             candidates_cap * sizeof *candidates);
    }
  c = &candidates[n_candidates++];
  c->path = xstrdup (path);
  c->mtime = mtime;
  c->size = size;
  c->e = e;
  c->total = total;
  c->done = done;
  c->skip = false;
}

/* Fewest chunks first, so that every file gets some before any gets
   many.  */
static int
compare_candidates (const void *a, const void *b)
{
  const struct candidate *x = a, *y = b;

  if (x->done != y->done)
    return x->done < y->done ? -1 : 1;
  return strcmp (x->path, y->path);
}

static double
now (void)
{
  struct timespec ts;

  clock_gettime (CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec / 1e9;
}

/* Write a chunk store line for chunk I of K of C, with the vector VEC,
   or "-" if VEC is NULL, to F.  */
static void
write_chunk (FILE *f, const struct candidate *c, size_t i, size_t k,
             const float *vec, size_t dim, const char *path)
{
  fprintf (f, "%lld %lld %zu %zu\t%s\t", c->mtime, c->size, i, k, c->path);
  if (vec)
    write_q8_vector (f, vec, dim);
  else
    putc ('-', f);
  putc ('\n', f);
  if (fflush (f) != 0 || ferror (f))
    error (EXIT_FAILURE, errno, "%s", path);
  chunk_owner_push (CHUNK_NEW);
}

/* Embed the next few chunks of C with M into F, the chunk store at
   PATH, until DEADLINE.  Return how many were embedded.  */
static size_t
chunk_candidate (struct candidate *c, struct model *m, magic_t magic_mime,
                 size_t dim, double deadline, FILE *f, const char *path)
{
  struct stat st;
  struct buf text;
  size_t k, skip, i, n = 0;

  if (lstat (c->path, &st) != 0 || !S_ISREG (st.st_mode)
      || st.st_mtime != c->mtime || st.st_size != c->size)
    {
      /* Changed since the walk: the next run embeds it afresh.  */
      c->skip = true;
      return 0;
    }

  buf_init (&text);
  k = chunk_text (magic_mime, c->path, c->size, &text, &skip);
  if (k == 0 && c->total > 0)
    {
      /* It had text before, so this is likely a helper timing out.  */
      c->skip = true;
      buf_free (&text);
      return 0;
    }
  if (c->total >= 0 && (size_t) c->total != k)
    {
      /* Rendered differently from before, say by a newer helper, so
         its chunks would fall elsewhere: begin them again.  */
      if (c->e)
        c->e->chunks_reset = true;
      c->done = 0;
    }
  c->total = k;
  if (k == 0)
    {
      write_chunk (f, c, 0, 0, NULL, dim, path);
      buf_free (&text);
      return 0;
    }

  if (verbose)
    fprintf (stderr, "%s: chunks from %zu of %zu\n", c->path, c->done + 1, k);
  for (i = c->done; i < k && i < c->done + CHUNK_BATCH && !interrupted
         && now () < deadline; i++)
    {
      float *vec;

      if (!embed_chunk (m, c->path, &text, k, skip, i, dim, &vec))
        {
          error (0, 0, "%s: could not embed chunk %zu", c->path, i + 1);
          c->skip = true;
          break;
        }
      write_chunk (f, c, i, k, vec, dim, path);
      free (vec);
      n++;
    }
  c->done = i;
  buf_free (&text);
  return n;
}

/* Embed chunks of the candidates with M, for at most chunk_time
   seconds, into the chunk store at PATH, which is begun afresh unless
   HAVE_CHUNKS.  Return how many were embedded.  */
static size_t
embed_chunks (struct model *m, magic_t magic_mime, size_t dim,
              const char *path, bool have_chunks)
{
  double deadline = now () + chunk_time;
  size_t n = 0;
  bool progress = true;
  FILE *f;

  if (have_chunks)
    {
      f = fopen (path, "a");
      if (f && chunks_unterminated)
        putc ('\n', f);
    }
  else
    {
      make_parent_dirs (path);
      f = fopen (path, "w");
      n_chunk_lines = 0;
      if (f)
        {
          fprintf (f, "# embed-index chunks model=%s dim=%zu\n",
                   EMBED_MODEL, dim);
          chunk_owner_push (CHUNK_NEW);
        }
    }
  if (!f)
    error (EXIT_FAILURE, errno, "%s", path);

  if (n_candidates > 0)
    qsort (candidates, n_candidates, sizeof *candidates,
           compare_candidates);
  while (progress && !interrupted && now () < deadline)
    {
      progress = false;
      for (size_t i = 0; i < n_candidates && !interrupted
             && now () < deadline; i++)
        {
          struct candidate *c = &candidates[i];

          if (c->skip || (c->total >= 0 && c->done >= (size_t) c->total))
            continue;
          n += chunk_candidate (c, m, magic_mime, dim, deadline, f, path);
          progress = true;
          if (!verbose && isatty (STDERR_FILENO) && n > 0)
            fprintf (stderr, "\rchunks %zu", n);
        }
    }
  if (!verbose && isatty (STDERR_FILENO) && n > 0)
    fputc ('\n', stderr);
  if (fclose (f) != 0)
    error (EXIT_FAILURE, errno, "%s", path);
  return n;
}

/* The exit status when another run holds the lock, so that a timer can
   tell it from a failure: EX_TEMPFAIL.  */
#define EXIT_LOCKED 75

/* Lock the store against other runs, for as long as the program
   runs.  Exit if another run holds the lock.  */
static void
lock_store (void)
{
  char *path = data_path ("index.lock");
  int fd;

  make_parent_dirs (path);
  fd = open (path, O_RDWR | O_CREAT | O_CLOEXEC, 0644);
  if (fd < 0)
    error (EXIT_FAILURE, errno, "%s", path);
  if (flock (fd, LOCK_EX | LOCK_NB) != 0)
    {
      if (errno == EWOULDBLOCK)
        error (EXIT_LOCKED, 0, "another embed-index is already running");
      error (EXIT_FAILURE, errno, "%s", path);
    }
  free (path);
}

int
main (int argc, char **argv)
{
  char store_model[256];
  char *store = store_path ();
  char *chunks = chunk_store_path ();
  char *store_dir;
  const char *home = getenv ("HOME");
  char *default_roots[2];
  char **roots;
  struct model *model;
  magic_t magic_mime, magic_desc;
  FILE *out;
  FTS *fts;
  FTSENT *ent;
  size_t dim = 0, n_new = 0, n_same = 0, n_failed = 0, n_chunks = 0;
  float *vec;
  bool have_store, have_chunks = false;
  int c;

  while ((c = getopt_long (argc, argv, "av", long_options, NULL))
         != -1)
    switch (c)
      {
      case 'a':
        include_hidden = true;
        break;
      case 'c':
        use_cpu = true;
        break;
      case 't':
        {
          char *end;
          errno = 0;
          chunk_time = strtol (optarg, &end, 10);
          if (errno || end == optarg || *end || chunk_time < 0)
            error (EXIT_FAILURE, 0, "invalid chunk time: %s", optarg);
        }
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

  lock_store ();
  store_dir = xstrdup (store);
  *strrchr (store_dir, '/') = '\0';

  have_store = load_store (store, store_model, sizeof store_model);
  if (have_store && strcmp (store_model, EMBED_MODEL) != 0)
    {
      error (0, 0, "model changed from %s to %s; reindexing everything",
             store_model, EMBED_MODEL);
      have_store = false;
      for (size_t i = 0; i < n_entries; i++)
        free (entries[i].path);
      n_entries = 0;
      n_lines = 0;
    }

  model = model_open (!use_cpu);
  if (!model)
    error (EXIT_FAILURE, 0, "cannot open the %s model", EMBED_MODEL);
  if (verbose)
    fprintf (stderr, "embedding on the %s\n", model_device (model));

  /* Learn the model's dimension.  */
  if (!model_embed (model, "warm up", &vec, &dim))
    error (EXIT_FAILURE, 0, "cannot run the %s model", EMBED_MODEL);
  free (vec);

  if (have_store)
    have_chunks = load_chunks (chunks, dim);

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
          fprintf (out, "# embed-index model=%s dim=%zu\n", EMBED_MODEL,
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

  fts = fts_open (roots, FTS_PHYSICAL | FTS_NOCHDIR, NULL);
  if (!fts)
    error (EXIT_FAILURE, errno, "cannot walk directories");

  while (!interrupted && (ent = fts_read (fts)))
    {
      struct entry *e;
      float *picture;

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
              if (e->chunk_total < 0
                  || e->chunk_done < (size_t) e->chunk_total)
                add_candidate (e->path, e->mtime, e->size, e,
                               e->chunk_total, e->chunk_done);
              continue;
            }
        }

      if (verbose)
        fprintf (stderr, "%s\n", ent->fts_path);

      if (!embed_file (model, magic_mime, magic_desc, ent->fts_path, home,
                       dim, &vec, &picture))
        {
          n_failed++;
          error (0, 0, "%s: could not embed", ent->fts_path);
          continue;
        }

      if (!write_record (out, ent->fts_path, ent->fts_statp, vec, picture,
                         dim))
        error (EXIT_FAILURE, errno, "%s", store);
      free (vec);
      free (picture);

      if (e)
        {
          keep[e->line] = 0;
          e->line = n_lines;
          e->changed = true;
        }
      keep_push (1);
      n_new++;
      add_candidate (ent->fts_path, ent->fts_statp->st_mtime,
                     ent->fts_statp->st_size, e, -1, 0);

      if (!verbose && isatty (STDERR_FILENO) && n_new % 10 == 0)
        fprintf (stderr, "\rembedded %zu, unchanged %zu", n_new, n_same);
    }
  if (!verbose && isatty (STDERR_FILENO))
    fputc ('\n', stderr);
  fts_close (fts);
  if (!interrupted && chunk_time > 0)
    {
      n_chunks = embed_chunks (model, magic_mime, dim, chunks, have_chunks);
      have_chunks = true;
    }
  model_close (model);
  close_media_models ();
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
  if (have_chunks)
    compact_chunks (chunks);

  fprintf (stderr, "%s: embedded %zu, unchanged %zu, failed %zu, "
           "chunks %zu%s\n", store, n_new, n_same, n_failed, n_chunks,
           interrupted ? " (interrupted)" : "");

  magic_close (magic_mime);
  magic_close (magic_desc);
  return interrupted ? 130 : EXIT_SUCCESS;
}
