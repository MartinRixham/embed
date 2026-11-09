/* ov-embed.h -- run embedding models on a GPU with OpenVINO.
   Copyright (C) 2026 Martin Rixham

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.  */

#ifndef EMBED_OV_EMBED_H
#define EMBED_OV_EMBED_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

struct ov_embed;

/* Load the ONNX model in PATH for OpenVINO's GPU device, caching
   compiled kernels in CACHE_DIR.  Return NULL, with a message in
   *WHY (static storage), if there is no GPU or the model cannot be
   compiled for it.  */
struct ov_embed *ov_embed_load (const char *path, const char *cache_dir,
                                const char **why);
void ov_embed_free (struct ov_embed *m);

/* Like onnx_embed: embed the N token ids IDS, at most OV_EMBED_MAX
   of them, into a freshly allocated unit vector.  */
bool ov_embed (struct ov_embed *m, const int64_t *ids, size_t n,
               float **vec, size_t *dim);

/* Like ov_embed_load, but for an image model like the one
   onnx_embed_image takes, with images of SIDE pixels square.  */
struct ov_embed *ov_embed_load_image (const char *path,
                                      const char *cache_dir, size_t side,
                                      const char **why);

/* Like onnx_embed_image, for a model loaded by ov_embed_load_image.  */
bool ov_embed_image (struct ov_embed *m, const float *pixels, float **vec,
                     size_t *dim);

/* The most dimensions of a fixed-shape model's input or output.  */
#define OV_MAX_RANK 4

/* Load the model in PATH, which takes one input of floats of RANK
   dimensions DIMS, like ov_embed_load.  */
struct ov_embed *ov_embed_load_fixed (const char *path,
                                      const char *cache_dir, size_t rank,
                                      const int64_t *dims, const char **why);

/* Run a model loaded by ov_embed_load_fixed or ov_embed_load_image on
   the floats IN, which fill its input shape.  On success store a
   freshly allocated copy of its first output in *OUT, and that
   output's shape in OUT_DIMS (OV_MAX_RANK of them at most) and
   *OUT_RANK, and return true.  */
bool ov_run_fixed (struct ov_embed *m, const float *in, float **out,
                   int64_t *out_dims, size_t *out_rank);

/* The GPU needs fixed input lengths, so texts are padded to the
   smallest of these that fits.  */
#define OV_EMBED_MAX 512

#endif /* EMBED_OV_EMBED_H */
