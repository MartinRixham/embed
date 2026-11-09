/* model.c -- the models: nomic-embed-text and -vision v1.5, and Whisper.
   Copyright (C) 2026 Martin Rixham

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   Each model is Nomic's ONNX export in float16, pinned to one revision
   on Hugging Face and checked by its SHA-256 hash.  The same file runs
   on the GPU through OpenVINO and on the CPU through ONNX Runtime.
   The vision model was trained to embed images into the text model's
   space, so that text queries can find them.  Whisper is an export
   by Hugging Face's optimum, in float32, which runs faster than float16
   on the CPU; whisper.c runs it.  */

#include "model.h"
#include "common.h"
#include "fetch.h"
#include "onnx-embed.h"
#include "ov-embed.h"
#include "tokenizer.h"
#include "whisper.h"

#include <errno.h>
#include <error.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

struct model_file
{
  const char *remote, *local, *sha256;
};

/* Where a model's files come from, and where they go.  */
struct model_source
{
  const char *name;
  const char *url;              /* Ends in a slash.  */
  const char *dir;              /* Under data_path, ending in a slash.  */
  const char *size;             /* For the user, before downloading.  */
  size_t n_files;
  const struct model_file *files;
};

static const struct model_file text_files[] = {
  { "onnx/model_fp16.onnx", "model_fp16.onnx",
    "cf5b5a86edb00f895561803cfc04729090a958340b8ca2ad76c143f565f6bb04" },
  { "vocab.txt", "vocab.txt",
    "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3" },
};

static const struct model_source text_source = {
  EMBED_MODEL,
  "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/"
  "e9b6763023c676ca8431644204f50c2b100d9aab/",
  "models/nomic-embed-text-v1.5/", "274 MB",
  sizeof text_files / sizeof *text_files, text_files
};

static const struct model_file vision_files[] = {
  { "onnx/model_fp16.onnx", "model_fp16.onnx",
    "6adad2d74c1da0452d1f90a1427679735bae0a1ee7e17b57a0bf70d8a795ca3d" },
};

static const struct model_source vision_source = {
  VISION_MODEL,
  "https://huggingface.co/nomic-ai/nomic-embed-vision-v1.5/resolve/"
  "e3a725bce72db07ca4adb1d83da08903f3ee02f8/",
  "models/nomic-embed-vision-v1.5/", "187 MB",
  sizeof vision_files / sizeof *vision_files, vision_files
};

static const struct model_file speech_files[] = {
  { "onnx/encoder_model.onnx", "encoder_model.onnx",
    "a9f3b752833b49e880dec91ee5b6d936112be7c3ea07c221024ba493439f46fe" },
  { "onnx/decoder_model_merged.onnx", "decoder_model_merged.onnx",
    "514903744bb1b45803ec571af99b31110491c6f77b0a154825866995fb124b73" },
  { "vocab.json", "vocab.json",
    "50d6a919f0a0601d56a04eb583c780d18553aa388254ba3158eb6a00f13e2c1a" },
};

static const struct model_source speech_source = {
  SPEECH_MODEL,
  "https://huggingface.co/onnx-community/whisper-base/resolve/"
  "27a07b4b52b3a4c5886134bb4320e93e1a4bbc7c/",
  "models/whisper-base/", "290 MB",
  sizeof speech_files / sizeof *speech_files, speech_files
};

/* How the vision model expects pixels to be scaled, per channel (those
   of OpenAI's CLIP).  */
static const float pixel_mean[3] = { 0.48145466, 0.4578275, 0.40821073 };
static const float pixel_std[3] = { 0.26862954, 0.26130258, 0.27577711 };

/* The model's input limit, in tokens; it can take more, but slowly,
   and the content of a file is cut short well before this anyway.  */
#define MAX_TOKENS OV_EMBED_MAX

/* Threads for ONNX Runtime on the CPU, leaving the rest of the machine
   usable.  */
#define CPU_THREADS 4

struct model
{
  struct tokenizer tokenizer;
  struct ov_embed *gpu;
  struct onnx_model *cpu;
};

/* Return the path of the local copy of the Ith file of SRC, freshly
   allocated.  */
static char *
local_path (const struct model_source *src, size_t i)
{
  struct buf b;
  char *dir = data_path (src->dir);

  buf_init (&b);
  buf_puts (&b, dir);
  buf_puts (&b, src->files[i].local);
  free (dir);
  return b.data;
}

/* Download the files of SRC if they are missing.  */
static bool
fetch_model (const struct model_source *src)
{
  bool announced = false;

  for (size_t i = 0; i < src->n_files; i++)
    {
      const struct model_file *file = &src->files[i];
      char *path = local_path (src, i);
      char *url;
      bool ok;

      if (!announced && access (path, F_OK) != 0)
        {
          error (0, 0, "downloading the %s model (%s); this is needed "
                 "only once", src->name, src->size);
          announced = true;
        }
      url = xmalloc (strlen (src->url) + strlen (file->remote) + 1);
      strcpy (url, src->url);
      strcat (url, file->remote);
      ok = fetch_file (url, path, file->sha256);
      free (url);
      free (path);
      if (!ok)
        return false;
    }
  return true;
}

/* Return the directory for OpenVINO's compiled kernels, freshly
   allocated.  */
static char *
cache_dir (void)
{
  const char *xdg = getenv ("XDG_CACHE_HOME");
  const char *home = getenv ("HOME");
  struct buf b;

  buf_init (&b);
  if (xdg && *xdg == '/')
    buf_puts (&b, xdg);
  else
    {
      buf_puts (&b, home ? home : ".");
      buf_puts (&b, "/.cache");
    }
  buf_puts (&b, "/embed/openvino/");
  make_parent_dirs (b.data);
  return b.data;
}

struct model *
model_open (bool use_gpu)
{
  struct model *m;
  char *onnx, *vocab;

  if (!fetch_model (&text_source))
    return NULL;
  m = xmalloc (sizeof *m);
  memset (m, 0, sizeof *m);
  onnx = local_path (&text_source, 0);
  vocab = local_path (&text_source, 1);
  if (!tokenizer_load (&m->tokenizer, vocab))
    {
      error (0, errno, "cannot load %s", vocab);
      free (m);
      m = NULL;
      goto out;
    }
  if (use_gpu)
    {
      char *cache = cache_dir ();
      const char *why;

      m->gpu = ov_embed_load (onnx, cache, &why);
      if (!m->gpu)
        error (0, 0, "%s; embedding on the CPU instead", why);
      free (cache);
    }
  if (!m->gpu && !(m->cpu = onnx_model_load (onnx, CPU_THREADS)))
    {
      tokenizer_free (&m->tokenizer);
      free (m);
      m = NULL;
    }

 out:
  free (onnx);
  free (vocab);
  return m;
}

void
model_close (struct model *m)
{
  if (!m)
    return;
  ov_embed_free (m->gpu);
  onnx_model_free (m->cpu);
  tokenizer_free (&m->tokenizer);
  free (m);
}

const char *
model_device (const struct model *m)
{
  return m->gpu ? "GPU" : "CPU";
}

bool
model_embed (struct model *m, const char *text, float **vec, size_t *dim)
{
  int64_t ids[MAX_TOKENS];
  size_t n = tokenize (&m->tokenizer, text, ids, MAX_TOKENS);

  if (m->gpu)
    return ov_embed (m->gpu, ids, n, vec, dim);
  return onnx_embed (m->cpu, ids, n, vec, dim);
}

struct vision
{
  struct ov_embed *gpu;
  struct onnx_model *cpu;
  float pixels[3 * VISION_SIDE * VISION_SIDE];
};

struct vision *
vision_open (bool use_gpu)
{
  struct vision *v;
  char *onnx;

  if (!fetch_model (&vision_source))
    return NULL;
  v = xmalloc (sizeof *v);
  memset (v, 0, sizeof *v);
  onnx = local_path (&vision_source, 0);
  if (use_gpu)
    {
      char *cache = cache_dir ();
      const char *why;

      v->gpu = ov_embed_load_image (onnx, cache, VISION_SIDE, &why);
      if (!v->gpu)
        error (0, 0, "%s; embedding images on the CPU instead", why);
      free (cache);
    }
  if (!v->gpu && !(v->cpu = onnx_model_load (onnx, CPU_THREADS)))
    {
      free (v);
      v = NULL;
    }
  free (onnx);
  return v;
}

void
vision_close (struct vision *v)
{
  if (!v)
    return;
  ov_embed_free (v->gpu);
  onnx_model_free (v->cpu);
  free (v);
}

bool
vision_embed (struct vision *v, const unsigned char *rgb, float **vec,
              size_t *dim)
{
  const size_t area = VISION_SIDE * VISION_SIDE;

  /* Interleaved bytes to planes of scaled floats.  */
  for (size_t c = 0; c < 3; c++)
    for (size_t i = 0; i < area; i++)
      v->pixels[c * area + i]
        = (rgb[3 * i + c] / 255.0f - pixel_mean[c]) / pixel_std[c];
  if (v->gpu)
    return ov_embed_image (v->gpu, v->pixels, vec, dim);
  return onnx_embed_image (v->cpu, v->pixels, VISION_SIDE, vec, dim);
}

struct speech
{
  struct whisper *whisper;
};

struct speech *
speech_open (bool use_gpu)
{
  struct speech *s;
  char *paths[3], *cache;

  if (!fetch_model (&speech_source))
    return NULL;
  cache = use_gpu ? cache_dir () : NULL;
  s = xmalloc (sizeof *s);
  for (size_t i = 0; i < 3; i++)
    paths[i] = local_path (&speech_source, i);
  s->whisper = whisper_load (paths[0], paths[1], paths[2], CPU_THREADS,
                             cache);
  for (size_t i = 0; i < 3; i++)
    free (paths[i]);
  free (cache);
  if (!s->whisper)
    {
      free (s);
      return NULL;
    }
  return s;
}

void
speech_close (struct speech *s)
{
  if (!s)
    return;
  whisper_free (s->whisper);
  free (s);
}

bool
speech_transcribe (struct speech *s, const float *pcm, size_t n,
                   struct buf *out)
{
  return whisper_transcribe (s->whisper, pcm, n, out);
}
