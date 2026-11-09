/* describe.c -- turning a file into the vectors that stand for it.
   Copyright (C) 2026 Martin Rixham

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   A file is turned into a short text description (its name, location,
   type and, where possible, its content: plain text, PDF text, office
   document text, OCR of images, or audio/video metadata and a
   transcript of what is said) for the text model, and an image or
   video into a picture vector, from pixels that ffmpeg decodes.  */

#include "describe.h"
#include "model.h"

#include <errno.h>
#include <error.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

/* How much file content to feed the model, in bytes.  */
#define MAX_CONTENT 1500

/* How many frames of a video to embed; their average stands for the
   video.  */
#define VIDEO_FRAMES 8

/* Bytes in a picture for the vision model.  */
#define PICTURE_BYTES (3 * VISION_SIDE * VISION_SIDE)

/* How many windows of speech to transcribe at most: the whole of a
   short recording, and windows spread through a long one.  */
#define SPEECH_WINDOWS 4

volatile sig_atomic_t interrupted;
long helper_timeout = 2;
bool use_cpu;

/* The vision and speech models, each opened when the first file needs
   it.  */
static struct vision *vision;
static bool vision_tried;
static struct speech *speech;
static bool speech_tried;

/* Run ARGV under timeout(1) and append up to MAX bytes of its standard
   output to OUT.  */
static void
run_capture (const char *const *argv, struct buf *out, size_t max)
{
  const char *args[32];
  char limit[32];
  int fds[2];
  pid_t pid;
  size_t i, got = 0;

  snprintf (limit, sizeof limit, "%ld", helper_timeout);
  args[0] = "timeout";
  args[1] = "-k";
  args[2] = "1";
  args[3] = limit;
  for (i = 0; argv[i] && i < 27; i++)
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

bool
picture_mime (const char *mime)
{
  return starts_with (mime, "image/") || starts_with (mime, "video/");
}

/* Return the MIME type of FILE, freshly allocated, or NULL.  */
char *
file_mime (magic_t magic_mime, const char *file)
{
  const char *mime = magic_file (magic_mime, file);
  return mime ? xstrdup (mime) : NULL;
}

/* Return the length of the audio or video FILE in seconds, or a
   negative number if it is unknown.  */
static double
media_duration (const char *file)
{
  const char *argv[] = { "ffprobe", "-v", "quiet", "-show_entries",
                         "format=duration", "-of", "default=nw=1:nk=1",
                         file, NULL };
  struct buf out;
  double d;
  char *end;

  buf_init (&out);
  run_capture (argv, &out, 64);
  d = strtod (out.data, &end);
  if (end == out.data || !(d > 0) || d > 1e9)
    d = -1;
  buf_free (&out);
  return d;
}

/* Return the speech model, opening it if this is the first time it is
   needed, or NULL if it cannot be opened.  */
static struct speech *
get_speech (void)
{
  if (!speech_tried)
    {
      speech_tried = true;
      speech = speech_open (!use_cpu);
      if (!speech)
        error (0, 0, "cannot open the %s model; audio and videos will be "
               "embedded without their speech", SPEECH_MODEL);
    }
  return speech;
}

/* Decode up to SPEECH_SAMPLES of the sound of FILE, from AT seconds in,
   as mono at SPEECH_RATE, into PCM.  Return the number of samples.  */
static size_t
grab_audio (const char *file, double at, float *pcm)
{
  char seek[32], length[32];
  struct buf url, out;
  const char *argv[] = { "ffmpeg", "-nostdin", "-v", "quiet", "-ss", seek,
                         "-t", length, "-i", NULL, "-vn", "-ac", "1",
                         "-ar", NULL, "-f", "f32le", "-", NULL };
  char rate[16];
  size_t n;

  snprintf (seek, sizeof seek, "%.3f", at);
  snprintf (length, sizeof length, "%d", SPEECH_SAMPLES / SPEECH_RATE);
  snprintf (rate, sizeof rate, "%d", SPEECH_RATE);
  buf_init (&url);
  buf_puts (&url, "file:");
  buf_puts (&url, file);
  argv[9] = url.data;
  argv[14] = rate;

  buf_init (&out);
  run_capture (argv, &out, SPEECH_SAMPLES * sizeof *pcm);
  n = out.len / sizeof *pcm;
  memcpy (pcm, out.data, n * sizeof *pcm);
  buf_free (&out);
  buf_free (&url);
  return n;
}

/* Append to OUT what is said in FILE: the transcripts of up to
   SPEECH_WINDOWS windows, together covering the whole of a short
   recording and spread evenly through a long one.  */
static void
transcribe (struct speech *sp, const char *file, struct buf *out)
{
  const double window = (double) SPEECH_SAMPLES / SPEECH_RATE;
  double duration = media_duration (file);
  size_t n_windows = 1, start = out->len;
  float *pcm = xmalloc (SPEECH_SAMPLES * sizeof *pcm);

  if (duration > 0)
    n_windows = duration / window + 1 < SPEECH_WINDOWS
      ? (size_t) (duration / window) + 1 : SPEECH_WINDOWS;
  for (size_t i = 0; i < n_windows && !interrupted; i++)
    {
      double at = 0;
      size_t n;

      if (duration > SPEECH_WINDOWS * window)
        at = (duration - window) * i / (n_windows - 1);
      else
        at = i * window;
      n = grab_audio (file, at, pcm);
      if (n < SPEECH_RATE / 10)
        continue;
      if (out->len > start)
        buf_puts (out, " ...");
      if (!speech_transcribe (sp, pcm, n, out)
          || out->len - start > MAX_CONTENT)
        break;
    }
  free (pcm);
  if (out->len - start > MAX_CONTENT)
    {
      out->len = start + MAX_CONTENT;
      out->data[out->len] = '\0';
    }
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
      struct speech *sp = get_speech ();
      struct buf said;

      run_capture (argv, out, MAX_CONTENT);
      buf_init (&said);
      if (sp)
        transcribe (sp, file, &said);
      if (said.len > 0)
        {
          buf_puts (out, "Speech:");
          buf_puts (out, said.data);
          buf_puts (out, "\n");
        }
      buf_free (&said);
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

/* Decode one frame of FILE, from AT seconds in if AT is not negative,
   scaled and cropped to the middle square for the vision model, into
   the PICTURE_BYTES of RGB.  The scaling is done as for CLIP: the
   shorter side is scaled to VISION_SIDE, with bicubic filtering.  */
static bool
grab_frame (const char *file, double at, unsigned char *rgb)
{
  char filter[160], seek[32];
  struct buf url, out;
  const char *argv[24];
  size_t n = 0;
  bool ok;

  snprintf (filter, sizeof filter,
            "scale=%d:%d:force_original_aspect_ratio=increase"
            ":flags=bicubic,crop=%d:%d", VISION_SIDE, VISION_SIDE,
            VISION_SIDE, VISION_SIDE);
  snprintf (seek, sizeof seek, "%.3f", at);
  /* The file: protocol, so that ffmpeg reads FILE as a file whatever
     its name looks like.  */
  buf_init (&url);
  buf_puts (&url, "file:");
  buf_puts (&url, file);

  argv[n++] = "ffmpeg";
  argv[n++] = "-nostdin";
  argv[n++] = "-v";
  argv[n++] = "quiet";
  if (at >= 0)
    {
      argv[n++] = "-ss";
      argv[n++] = seek;
    }
  argv[n++] = "-i";
  argv[n++] = url.data;
  argv[n++] = "-an";
  argv[n++] = "-frames:v";
  argv[n++] = "1";
  argv[n++] = "-vf";
  argv[n++] = filter;
  argv[n++] = "-f";
  argv[n++] = "rawvideo";
  argv[n++] = "-pix_fmt";
  argv[n++] = "rgb24";
  argv[n++] = "-";
  argv[n] = NULL;

  buf_init (&out);
  run_capture (argv, &out, PICTURE_BYTES + 1);
  ok = out.len == PICTURE_BYTES;
  if (ok)
    memcpy (rgb, out.data, PICTURE_BYTES);
  buf_free (&out);
  buf_free (&url);
  return ok;
}

/* Return the vision model, opening it if this is the first time it is
   needed, or NULL if it cannot be opened.  */
struct vision *
get_vision (void)
{
  if (!vision_tried)
    {
      vision_tried = true;
      vision = vision_open (!use_cpu);
      if (!vision)
        error (0, 0, "cannot open the %s model; images and videos will "
               "be embedded by their text alone", VISION_MODEL);
    }
  return vision;
}

/* Embed what FILE, of type MIME, shows: the picture, if it is an
   image, or the average of a few frames spread through it, if it is a
   video.  Return a freshly allocated unit vector of DIM floats, or
   NULL if FILE cannot be decoded.  */
float *
embed_picture (struct vision *vis, const char *file, const char *mime,
               size_t dim)
{
  bool video = starts_with (mime, "video/");
  double duration = video ? media_duration (file) : -1;
  size_t n_frames = duration > 0 ? VIDEO_FRAMES : 1, got = 0;
  unsigned char *rgb = xmalloc (PICTURE_BYTES);
  float *sum = xmalloc (dim * sizeof *sum);
  double norm = 0;

  memset (sum, 0, dim * sizeof *sum);
  for (size_t i = 0; i < n_frames && !interrupted; i++)
    {
      /* Frames from the middle of equal slices, avoiding the very
         start and end, which are often black.  */
      double at = duration > 0 ? duration * (i + 0.5) / n_frames : -1;
      float *vec;
      size_t d;

      if (!grab_frame (file, at, rgb) || !vision_embed (vis, rgb, &vec, &d))
        continue;
      if (d != dim)
        {
          error (0, 0, "%s model returned %zu dimensions instead of %zu",
                 VISION_MODEL, d, dim);
          free (vec);
          break;
        }
      for (size_t j = 0; j < dim; j++)
        sum[j] += vec[j];
      free (vec);
      got++;
    }
  free (rgb);

  for (size_t j = 0; j < dim; j++)
    norm += (double) sum[j] * sum[j];
  if (got == 0 || norm == 0)
    {
      free (sum);
      return NULL;
    }
  norm = sqrt (norm);
  for (size_t j = 0; j < dim; j++)
    sum[j] /= norm;
  return sum;
}

/* Build the text that represents FILE for the model, in OUT.  HOME is
   used to shorten the displayed location.  */
void
describe_file (magic_t magic_mime, magic_t magic_desc, const char *file,
               const char *home, bool with_content, struct buf *out)
{
  const char *name = strrchr (file, '/');
  const char *mime = magic_file (magic_mime, file);
  const char *desc = magic_file (magic_desc, file);
  size_t home_len = strlen (home);

  name = name ? name + 1 : file;
  buf_puts (out, DOCUMENT_PREFIX);
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

/* Embed TEXT like model_embed, but fail unless the vector has DIM
   dimensions, so that the store stays consistent.  */
bool
embed_checked (struct model *m, const char *text, size_t dim, float **vec)
{
  size_t d;

  if (!model_embed (m, text, vec, &d))
    return false;
  if (d == dim)
    return true;
  error (0, 0, "model returned %zu dimensions instead of %zu", d, dim);
  free (*vec);
  return false;
}


bool
embed_file (struct model *m, magic_t magic_mime, magic_t magic_desc,
            const char *file, const char *home, size_t dim,
            float **vec, float **picture)
{
  struct buf text;
  char *mime;
  bool ok;

  buf_init (&text);
  describe_file (magic_mime, magic_desc, file, home, true, &text);
  ok = embed_checked (m, text.data, dim, vec);
  if (!ok && !interrupted)
    {
      /* Fall back to embedding just the name and type.  */
      text.len = 0;
      describe_file (magic_mime, magic_desc, file, home, false, &text);
      ok = embed_checked (m, text.data, dim, vec);
    }
  buf_free (&text);

  *picture = NULL;
  if (!ok)
    return false;
  mime = file_mime (magic_mime, file);
  if (mime && picture_mime (mime) && get_vision ())
    *picture = embed_picture (vision, file, mime, dim);
  free (mime);
  return true;
}

void
close_media_models (void)
{
  vision_close (vision);
  speech_close (speech);
  vision = NULL;
  speech = NULL;
  vision_tried = speech_tried = false;
}
