/* describe.c -- turning a file into the vectors that stand for it.
   Copyright (C) 2026 Martin Rixham

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   A file is turned into a short text description (its name, location,
   type and, where possible, its content: plain text, PDF text, office
   document text, EPUB book metadata and text, OCR of images, or a
   summary of a recording: its kind, length, format and tags) for the
   text model, and an image or video into a picture vector, from pixels
   that ffmpeg decodes.

   What is said in a recording goes into its chunks, not its
   description.  A transcript is mostly dialogue, which pulls the
   description's vector towards what is talked about and away from
   what the file is, so that a search for "movies" or "video files"
   would pass the videos by.  */

#include "describe.h"
#include "model.h"

#include <ctype.h>
#include <errno.h>
#include <error.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/wait.h>
#include <unistd.h>

/* How much file content to feed the model, in bytes.  */
#define MAX_CONTENT 1500

/* The size of a chunk: a passage of a long file, embedded on its own,
   as large as the content in a file's description.  */
#define CHUNK_BYTES MAX_CONTENT

/* How much of a file's text to split into chunks at most, in bytes.  */
#define CHUNK_TEXT_MAX (4 << 20)

/* The most chunks to embed of a long document, and of any other long
   text, such as code or data, which is less often searched for by
   what one passage of it says.  */
#define CHUNKS_PROSE 64
#define CHUNKS_OTHER 4

/* The most chunks of a recording's transcript.  */
#define CHUNKS_SPEECH 4

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

/* Replace the XML character references and predefined entities in B
   with the characters they stand for, and any other entity with a
   space.  */
static void
decode_entities (struct buf *b)
{
  static const struct { const char *name; char c; } named[] = {
    { "lt", '<' }, { "gt", '>' }, { "amp", '&' }, { "quot", '"' },
    { "apos", '\'' }
  };
  size_t out = 0;

  for (size_t i = 0; i < b->len; i++)
    {
      char *semi;
      size_t n;

      if (b->data[i] != '&'
          || !(semi = memchr (b->data + i, ';', b->len - i < 12
                              ? b->len - i : 12)))
        {
          b->data[out++] = b->data[i];
          continue;
        }
      n = semi - (b->data + i + 1);
      if (b->data[i + 1] == '#')
        {
          bool hex = b->data[i + 2] == 'x' || b->data[i + 2] == 'X';
          unsigned long cp = strtoul (b->data + i + 2 + hex, NULL,
                                      hex ? 16 : 10);

          /* The UTF-8 encoding of CP is never longer than the
             reference, so it fits in place.  */
          if (cp == 0 || cp > 0x10FFFF)
            b->data[out++] = ' ';
          else if (cp < 0x80)
            b->data[out++] = cp;
          else if (cp < 0x800)
            {
              b->data[out++] = 0xC0 | cp >> 6;
              b->data[out++] = 0x80 | (cp & 0x3F);
            }
          else if (cp < 0x10000)
            {
              b->data[out++] = 0xE0 | cp >> 12;
              b->data[out++] = 0x80 | (cp >> 6 & 0x3F);
              b->data[out++] = 0x80 | (cp & 0x3F);
            }
          else
            {
              b->data[out++] = 0xF0 | cp >> 18;
              b->data[out++] = 0x80 | (cp >> 12 & 0x3F);
              b->data[out++] = 0x80 | (cp >> 6 & 0x3F);
              b->data[out++] = 0x80 | (cp & 0x3F);
            }
        }
      else
        {
          char c = ' ';

          for (size_t k = 0; k < sizeof named / sizeof *named; k++)
            if (strlen (named[k].name) == n
                && strncmp (b->data + i + 1, named[k].name, n) == 0)
              c = named[k].c;
          b->data[out++] = c;
        }
      i += n + 1;
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

/* Return the first start tag named NAME in S, or NULL.  */
static const char *
find_tag (const char *s, const char *name)
{
  size_t n = strlen (name);

  for (; (s = strchr (s, '<')); s++)
    if (strncmp (s + 1, name, n) == 0 && strchr (" \t\r\n/>", s[n + 1])
        && s[n + 1] != '\0')
      return s;
  return NULL;
}

/* Return the value of the attribute NAME in the XML start tag that
   begins at TAG, freshly allocated, or NULL.  */
static char *
xml_attr (const char *tag, const char *name)
{
  const char *end = strchr (tag, '>');
  size_t n = strlen (name);

  for (const char *p = tag; end && (p = strstr (p, name)) && p < end; p++)
    {
      const char *q = p + n;

      if (p[-1] != ' ' && p[-1] != '\t' && p[-1] != '\n' && p[-1] != '\r')
        continue;
      while (*q == ' ')
        q++;
      if (*q++ != '=')
        continue;
      while (*q == ' ')
        q++;
      if (*q != '"' && *q != '\'')
        continue;
      const char *close = strchr (q + 1, *q);
      if (!close || close > end)
        return NULL;
      char *v = xmalloc (close - q);
      memcpy (v, q + 1, close - q - 1);
      v[close - q - 1] = '\0';
      return v;
    }
  return NULL;
}

/* Replace the %XX escapes in the URL path S, in place.  */
static void
url_unescape (char *s)
{
  char *out = s;

  for (; *s; s++)
    {
      unsigned int c;

      if (*s == '%' && sscanf (s + 1, "%2x", &c) == 1 && c != 0)
        {
          *out++ = c;
          s += 2;
        }
      else
        *out++ = *s;
    }
  *out = '\0';
}

/* Append up to MAX bytes of the member MEMBER of the zip archive FILE
   to OUT.  */
static void
read_zip_member (const char *file, const char *member, struct buf *out,
                 size_t max)
{
  struct buf name;

  /* Escape the wildcards, so that unzip matches MEMBER exactly.  */
  buf_init (&name);
  for (; *member; member++)
    {
      if (strchr ("*?[]\\", *member))
        buf_puts (&name, "\\");
      buf_append (&name, member, 1);
    }
  const char *argv[] = { "unzip", "-p", file, name.data, NULL };
  run_capture (argv, out, max);
  buf_free (&name);
}

/* Append the text of the member MEMBER of the zip archive FILE, with
   its markup stripped, to OUT, until OUT holds END bytes.  */
static void
append_zip_text (const char *file, const char *member, struct buf *out,
                 size_t end)
{
  struct buf tmp;

  if (out->len >= end)
    return;
  buf_init (&tmp);
  /* Markup takes room too: allow for a page of it or so.  */
  read_zip_member (file, member, &tmp, 4 * (end - out->len) + 16384);
  strip_markup (&tmp);
  decode_entities (&tmp);
  strip_markup (&tmp);
  if (tmp.len > 0 && strcmp (tmp.data, " ") != 0)
    buf_append (out, tmp.data, tmp.len < end - out->len ? tmp.len
                : end - out->len);
  buf_free (&tmp);
}

/* Append a textual rendering of the EPUB book FILE to OUT: its
   metadata (title, author, description, subjects) and then the start
   of its text, in reading order.  */
static void
extract_epub (const char *file, struct buf *out, size_t limit)
{
  struct buf tmp, opf;
  const char *p, *spine;
  char *opf_path = NULL;
  size_t end = out->len + limit, dir_len;

  buf_init (&tmp);
  read_zip_member (file, "META-INF/container.xml", &tmp, 16 * 1024);
  if ((p = find_tag (tmp.data, "rootfile")))
    opf_path = xml_attr (p, "full-path");
  buf_free (&tmp);
  if (!opf_path)
    return;

  /* The package document: its metadata, then its manifest and spine.  */
  buf_init (&opf);
  read_zip_member (file, opf_path, &opf, 1024 * 1024);
  if ((p = find_tag (opf.data, "metadata"))
      || (p = find_tag (opf.data, "opf:metadata")))
    {
      const char *stop = strstr (p + 9, "metadata>");

      buf_init (&tmp);
      buf_append (&tmp, p, stop ? (size_t) (stop - p) : strlen (p));
      strip_markup (&tmp);
      decode_entities (&tmp);
      strip_markup (&tmp);
      buf_append (out, tmp.data, tmp.len < limit ? tmp.len : limit);
      buf_free (&tmp);
    }

  p = strrchr (opf_path, '/');
  dir_len = p ? (size_t) (p - opf_path + 1) : 0;
  /* The package elements may carry an opf: prefix.  */
  const char *itemref = "itemref", *item_tag = "item";
  spine = find_tag (opf.data, "spine");
  if (!spine && (spine = find_tag (opf.data, "opf:spine")))
    {
      itemref = "opf:itemref";
      item_tag = "opf:item";
    }
  for (p = spine; p && out->len < end && !interrupted
         && (p = find_tag (p, itemref)); p++)
    {
      char *idref = xml_attr (p, "idref");
      const char *item;

      if (!idref)
        continue;
      /* Find the manifest item with that id.  */
      for (item = opf.data; (item = find_tag (item, item_tag)); item++)
        {
          char *id;
          bool match;

          id = xml_attr (item, "id");
          match = id && strcmp (id, idref) == 0;
          free (id);
          if (match)
            break;
        }
      free (idref);
      if (!item || item > spine)
        continue;

      char *href = xml_attr (item, "href");
      if (!href)
        continue;
      url_unescape (href);
      buf_init (&tmp);
      buf_append (&tmp, opf_path, dir_len);
      buf_puts (&tmp, href);
      free (href);
      if (out->len > 0 && out->data[out->len - 1] != ' ')
        buf_puts (out, " ");
      append_zip_text (file, tmp.data, out, end);
      buf_free (&tmp);
    }
  buf_free (&opf);
  free (opf_path);
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
   recording and spread evenly through a long one, and up to LIMIT
   bytes in all.  */
static void
transcribe (struct speech *sp, const char *file, struct buf *out,
            size_t limit)
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
          || out->len - start > limit)
        break;
    }
  free (pcm);
  if (out->len - start > limit)
    {
      out->len = start + limit;
      out->data[out->len] = '\0';
    }
}

/* Codecs by the names people know them by, where those are not
   ffprobe's.  */
static const char *const codec_names[][2] = {
  {"h264", "H.264"}, {"hevc", "H.265"}, {"av1", "AV1"}, {"vp8", "VP8"},
  {"vp9", "VP9"}, {"mpeg4", "MPEG-4"}, {"mpeg2video", "MPEG-2"},
  {"mpeg1video", "MPEG-1"}, {"theora", "Theora"}, {"prores", "ProRes"},
  {"aac", "AAC"}, {"mp3", "MP3"}, {"mp2", "MP2"}, {"ac3", "Dolby Digital"},
  {"eac3", "Dolby Digital Plus"}, {"truehd", "Dolby TrueHD"},
  {"dts", "DTS"}, {"opus", "Opus"}, {"vorbis", "Vorbis"}, {"flac", "FLAC"},
  {"alac", "Apple Lossless"}, {"wmav2", "WMA"}, {"amr_nb", "AMR"},
};

/* Tags that say how a file was made rather than what is in it.  */
static const char *const technical_tags[] = {
  "major_brand", "minor_version", "compatible_brands", "encoder",
  "encoded_by", "creation_time", "handler_name", "vendor_id", "duration",
  "itunsmpb", "itunnorm", "itunes_cddb_ids", "media_type", NULL
};

/* Append the name of CODEC, as ffprobe names it, to OUT.  */
static void
put_codec (struct buf *out, const char *codec)
{
  for (size_t i = 0; i < sizeof codec_names / sizeof *codec_names; i++)
    if (strcmp (codec, codec_names[i][0]) == 0)
      {
        buf_puts (out, codec_names[i][1]);
        return;
      }
  if (starts_with (codec, "pcm_"))
    buf_puts (out, "PCM");
  else
    buf_puts (out, codec);
}

/* Append SECONDS, rounded, to OUT in words.  */
static void
put_duration (struct buf *out, double seconds)
{
  long s = (long) (seconds + 0.5), m = (seconds + 30) / 60;
  char t[64];

  if (s < 60)
    snprintf (t, sizeof t, "%ld second%s", s, s == 1 ? "" : "s");
  else if (m < 60)
    snprintf (t, sizeof t, "%ld minute%s", m, m == 1 ? "" : "s");
  else if (m % 60 == 0)
    snprintf (t, sizeof t, "%ld hour%s", m / 60, m / 60 == 1 ? "" : "s");
  else
    snprintf (t, sizeof t, "%ld hour%s %ld minute%s", m / 60,
              m / 60 == 1 ? "" : "s", m % 60, m % 60 == 1 ? "" : "s");
  buf_puts (out, t);
}

/* Append a summary of the recording FILE, of type MIME, to OUT: what
   kind of recording it is, how long, its picture size and codecs, then
   its tags, one to a line, such as "Title: Moonlight Sonata".  */
static void
describe_media (const char *file, const char *mime, struct buf *out)
{
  const char *argv[] = { "ffprobe", "-v", "quiet", "-show_entries",
                         ("format=duration:format_tags"
                          ":stream=codec_type,codec_name,width,height"
                          ":stream_disposition=attached_pic"),
                         file, NULL };
  bool video = starts_with (mime, "video/"), in_stream = false;
  bool subtitles = false, cover = false;
  char type[32] = "", codec[64] = "", video_codec[64] = "";
  long width = 0, height = 0, w = 0, h = 0;
  double duration = -1;
  struct buf info, sound, tags;
  char *line, *save;

  buf_init (&info);
  buf_init (&sound);
  buf_init (&tags);
  run_capture (argv, &info, MAX_CONTENT);
  for (line = strtok_r (info.data, "\n", &save); line;
       line = strtok_r (NULL, "\n", &save))
    {
      char *eq = strchr (line, '=');

      if (strcmp (line, "[STREAM]") == 0)
        {
          in_stream = true;
          type[0] = codec[0] = '\0';
          w = h = 0;
          cover = false;
        }
      else if (strcmp (line, "[/STREAM]") == 0)
        {
          in_stream = false;
          if (strcmp (type, "video") == 0 && !cover && !video_codec[0])
            {
              strcpy (video_codec, codec);
              width = w;
              height = h;
            }
          else if (strcmp (type, "audio") == 0 && codec[0])
            {
              /* Each codec once, however many tracks use it.  */
              struct buf name;

              buf_init (&name);
              put_codec (&name, codec);
              if (!sound.data || !strstr (sound.data, name.data))
                {
                  if (sound.len > 0)
                    buf_puts (&sound, ", ");
                  buf_append (&sound, name.data, name.len);
                }
              buf_free (&name);
            }
          else if (strcmp (type, "subtitle") == 0)
            subtitles = true;
        }
      else if (!eq)
        continue;
      else if (in_stream)
        {
          *eq++ = '\0';
          if (strcmp (line, "codec_type") == 0)
            snprintf (type, sizeof type, "%s", eq);
          else if (strcmp (line, "codec_name") == 0)
            snprintf (codec, sizeof codec, "%s", eq);
          else if (strcmp (line, "width") == 0)
            w = atol (eq);
          else if (strcmp (line, "height") == 0)
            h = atol (eq);
          else if (strcmp (line, "DISPOSITION:attached_pic") == 0)
            cover = atoi (eq) != 0;
        }
      else if (starts_with (line, "duration="))
        {
          char *end;

          duration = strtod (eq + 1, &end);
          if (end == eq + 1 || !(duration > 0) || duration > 1e9)
            duration = -1;
        }
      else if (starts_with (line, "TAG:") && eq[1])
        {
          char *key = line + 4;
          bool skip = false;

          *eq++ = '\0';
          for (size_t i = 0; technical_tags[i] && !skip; i++)
            skip = strcasecmp (key, technical_tags[i]) == 0;
          if (skip || !*key)
            continue;
          for (char *k = key; *k; k++)
            {
              *k = k == key ? toupper ((unsigned char) *k)
                : tolower ((unsigned char) *k);
              if (*k == '_')
                *k = ' ';
            }
          buf_puts (&tags, key);
          buf_puts (&tags, ": ");
          buf_puts (&tags, eq);
          buf_puts (&tags, "\n");
        }
    }

  buf_puts (out, video ? "Video" : "Audio recording");
  if (duration > 0)
    {
      buf_puts (out, ", ");
      put_duration (out, duration);
      buf_puts (out, " long");
    }
  if (video && width > 0 && height > 0)
    {
      char size[64];

      snprintf (size, sizeof size, ", %ldx%ld", width, height);
      buf_puts (out, size);
    }
  if (video && video_codec[0])
    {
      buf_puts (out, ", ");
      put_codec (out, video_codec);
    }
  if (video)
    {
      if (sound.len > 0)
        {
          buf_puts (out, ", with ");
          buf_append (out, sound.data, sound.len);
          buf_puts (out, " sound");
        }
      else
        buf_puts (out, ", silent");
      if (subtitles)
        buf_puts (out, " and subtitles");
    }
  else if (sound.len > 0)
    {
      buf_puts (out, ", ");
      buf_append (out, sound.data, sound.len);
    }
  buf_puts (out, "\n");
  if (tags.len > MAX_CONTENT)
    tags.len = MAX_CONTENT;
  buf_append (out, tags.data, tags.len);
  buf_free (&info);
  buf_free (&sound);
  buf_free (&tags);
}

/* Append up to LIMIT bytes of the text of the office document FILE,
   of type MIME, to OUT.  */
static void
extract_office (const char *file, const char *mime, struct buf *out,
                size_t limit)
{
  const char *member = "content.xml";
  struct buf tmp;

  if (strstr (mime, "wordprocessingml"))
    member = "word/document.xml";
  else if (strstr (mime, "presentationml"))
    member = "ppt/slides/slide*.xml";
  else if (strstr (mime, "spreadsheetml"))
    member = "xl/sharedStrings.xml";

  const char *argv[] = { "unzip", "-p", file, member, NULL };
  buf_init (&tmp);
  run_capture (argv, &tmp, 4 * limit);
  strip_markup (&tmp);
  buf_append (out, tmp.data, tmp.len < limit ? tmp.len : limit);
  buf_free (&tmp);
}

/* Append a textual rendering of the content of FILE, of type MIME, to
   OUT: up to LIMIT bytes of it.  If WHOLE, it is for chunks, so render
   all of a PDF, what is said in a recording, and nothing of images,
   whose content is not text that runs on.  */
static void
extract_upto (const char *file, const char *mime, struct buf *out,
              size_t limit, bool whole)
{
  if (textual_mime (mime))
    {
      read_head (file, out, limit);
      if (strcmp (mime, "text/html") == 0 || ends_with (mime, "xml"))
        strip_markup (out);
    }
  else if (strcmp (mime, "application/pdf") == 0)
    {
      const char *argv[] = { "pdftotext", "-q", "-l", "2", "-enc", "UTF-8",
                             file, "-", NULL };
      const char *all[] = { "pdftotext", "-q", "-enc", "UTF-8", file, "-",
                            NULL };
      run_capture (whole ? all : argv, out, limit);
    }
  else if (whole)
    {
      if (starts_with (mime, "audio/") || starts_with (mime, "video/"))
        {
          struct speech *sp = get_speech ();

          if (sp)
            transcribe (sp, file, out, limit);
        }
      else if (strcmp (mime, "application/epub+zip") == 0)
        extract_epub (file, out, limit);
      else if (starts_with (mime, "application/vnd.openxmlformats")
               || starts_with (mime, "application/vnd.oasis.opendocument"))
        extract_office (file, mime, out, limit);
    }
  else if (starts_with (mime, "image/"))
    {
      const char *argv[] = { "tesseract", file, "stdout", "-l", "eng",
                             NULL };
      run_capture (argv, out, MAX_CONTENT);
    }
  else if (starts_with (mime, "audio/") || starts_with (mime, "video/"))
    describe_media (file, mime, out);
  else if (strcmp (mime, "application/epub+zip") == 0)
    extract_epub (file, out, limit);
  else if (starts_with (mime, "application/vnd.openxmlformats")
           || starts_with (mime, "application/vnd.oasis.opendocument"))
    extract_office (file, mime, out, limit);
}

/* Append a textual rendering of the content of FILE, of type MIME, to
   OUT: as much as goes into its description.  */
static void
extract_content (const char *file, const char *mime, struct buf *out)
{
  extract_upto (file, mime, out, MAX_CONTENT, false);
}

/* Return the most chunks to embed of a file of type MIME: 0 if its
   content is not running text.  */
static size_t
chunk_cap (const char *mime)
{
  if (starts_with (mime, "audio/") || starts_with (mime, "video/"))
    return CHUNKS_SPEECH;
  static const char *const prose[] = {
    "application/pdf", "application/epub+zip", "text/plain",
    "text/markdown", "text/x-markdown", "text/html", "text/rtf",
    "text/x-tex", "text/troff", "message/rfc822", NULL
  };

  for (size_t i = 0; prose[i]; i++)
    if (strcmp (mime, prose[i]) == 0)
      return CHUNKS_PROSE;
  if (starts_with (mime, "application/vnd.openxmlformats")
      || starts_with (mime, "application/vnd.oasis.opendocument"))
    return CHUNKS_PROSE;
  if (textual_mime (mime))
    return CHUNKS_OTHER;
  return 0;
}

/* Return the number of windows of CHUNK_BYTES in LEN bytes of text,
   counting a last partial one.  */
static size_t
n_windows (size_t len)
{
  return (len + CHUNK_BYTES - 1) / CHUNK_BYTES;
}

size_t
chunk_plan (size_t text_len, size_t skip, size_t cap)
{
  size_t n = n_windows (text_len);

  if (n <= skip)
    return 0;
  return n - skip < cap ? n - skip : cap;
}

size_t
chunk_text (magic_t magic_mime, const char *file, long long size,
            struct buf *out, size_t *skip)
{
  const char *m = magic_file (magic_mime, file);
  char *mime;
  size_t cap, k;
  long saved = helper_timeout;

  /* The first window of text is in the file's own description already,
     but none of what is said in a recording.  */
  *skip = m && (starts_with (m, "audio/") || starts_with (m, "video/"))
    ? 0 : 1;
  if (!m || !(cap = chunk_cap (m)))
    return 0;
  /* Plain text is no longer than the file.  */
  if (textual_mime (m) && size <= CHUNK_BYTES)
    return 0;
  mime = xstrdup (m);
  /* Rendering the whole of a book takes longer than its start, and
     decoding sound from far into a long video longer than its type.  */
  helper_timeout = saved * 15;
  extract_upto (file, mime, out, CHUNK_TEXT_MAX, true);
  helper_timeout = saved;
  free (mime);
  k = chunk_plan (out->len, *skip, cap);
  return k;
}

/* Return the I'th of the slots 0 to K - 1 in the order chunks are
   embedded: the bit-reversed order, so that the first few chunks are
   spread through the whole file, and later ones fall between them.  */
static size_t
chunk_slot (size_t k, size_t i)
{
  unsigned bits = 0;

  while (((size_t) 1 << bits) < k)
    bits++;
  for (size_t t = 0, seen = 0; t < (size_t) 1 << bits; t++)
    {
      size_t j = 0;

      for (unsigned b = 0; b < bits; b++)
        if (t & (size_t) 1 << b)
          j |= (size_t) 1 << (bits - 1 - b);
      if (j < k && seen++ == i)
        return j;
    }
  return 0;
}

/* Whether C is white space.  */
static bool
is_space (char c)
{
  return c == ' ' || c == '\n' || c == '\t' || c == '\r';
}

/* Return AT, an offset into TEXT of LEN bytes, moved forward to the
   start of a word, looking at most a little way ahead, or else to the
   start of a UTF-8 character.  */
static size_t
word_start (const char *text, size_t len, size_t at)
{
  for (size_t p = at; p < len && p < at + 200; p++)
    if (p == 0 || is_space (text[p - 1]))
      return p;
  while (at < len && ((unsigned char) text[at] & 0xC0) == 0x80)
    at++;
  return at;
}

void
chunk_window (const struct buf *text, size_t k, size_t skip, size_t i,
              size_t *start, size_t *end)
{
  size_t n = n_windows (text->len);
  size_t w = skip + chunk_slot (k, i) * (n - skip) / k;

  *start = word_start (text->data, text->len, w * CHUNK_BYTES);
  *end = w * CHUNK_BYTES + CHUNK_BYTES;
  if (*end >= text->len)
    *end = text->len;
  else
    *end = word_start (text->data, text->len, *end);
  if (*end < *start)
    *end = *start;
}

bool
embed_chunk (struct model *m, const char *file, const struct buf *text,
             size_t k, size_t skip, size_t i, size_t dim, float **vec)
{
  const char *name = strrchr (file, '/');
  struct buf b;
  size_t start, end;
  bool ok;

  chunk_window (text, k, skip, i, &start, &end);
  buf_init (&b);
  buf_puts (&b, DOCUMENT_PREFIX);
  buf_puts (&b, "From the file ");
  buf_puts (&b, name ? name + 1 : file);
  buf_puts (&b, ":\n\n");
  buf_append (&b, text->data + start, end - start);
  ok = embed_checked (m, b.data, dim, vec);
  buf_free (&b);
  return ok;
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
