/* model.c -- the embedding model, nomic-embed-text v1.5.
   Copyright (C) 2026 Martin Rixham

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   The model is Nomic's ONNX export in float16, pinned to one revision
   on Hugging Face and checked by its SHA-256 hash.  The same file runs
   on the GPU through OpenVINO and on the CPU through ONNX Runtime.  */

#include "model.h"
#include "common.h"
#include "fetch.h"
#include "onnx-embed.h"
#include "ov-embed.h"
#include "tokenizer.h"

#include <errno.h>
#include <error.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MODEL_REVISION "e9b6763023c676ca8431644204f50c2b100d9aab"
#define MODEL_URL "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/" \
  "resolve/" MODEL_REVISION "/"
#define MODEL_DIR "models/nomic-embed-text-v1.5/"

static const struct
{
  const char *remote, *local, *sha256;
} files[] = {
  { "onnx/model_fp16.onnx", "model_fp16.onnx",
    "cf5b5a86edb00f895561803cfc04729090a958340b8ca2ad76c143f565f6bb04" },
  { "vocab.txt", "vocab.txt",
    "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3" },
};

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

/* Return the path of the local copy of files[I], freshly allocated.  */
static char *
local_path (size_t i)
{
  struct buf b;
  char *dir = data_path (MODEL_DIR);

  buf_init (&b);
  buf_puts (&b, dir);
  buf_puts (&b, files[i].local);
  free (dir);
  return b.data;
}

/* Download the model's files if they are missing.  */
static bool
fetch_model (void)
{
  bool announced = false;

  for (size_t i = 0; i < sizeof files / sizeof *files; i++)
    {
      char *path = local_path (i);
      char *url;
      bool ok;

      if (!announced && access (path, F_OK) != 0)
        {
          error (0, 0, "downloading the %s model (274 MB); this is needed "
                 "only once", EMBED_MODEL);
          announced = true;
        }
      url = xmalloc (strlen (MODEL_URL) + strlen (files[i].remote) + 1);
      strcpy (url, MODEL_URL);
      strcat (url, files[i].remote);
      ok = fetch_file (url, path, files[i].sha256);
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

  if (!fetch_model ())
    return NULL;
  m = xmalloc (sizeof *m);
  memset (m, 0, sizeof *m);
  onnx = local_path (0);
  vocab = local_path (1);
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
