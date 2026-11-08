/* onnx-embed.h -- run a BERT-style embedding model with ONNX Runtime.
   Copyright (C) 2026 Martin Rixham

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.  */

#ifndef EMBED_ONNX_EMBED_H
#define EMBED_ONNX_EMBED_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

struct onnx_model;

/* Load the ONNX model in PATH, to run on THREADS threads (0 for the
   ONNX Runtime default).  Report a problem and return NULL if it
   cannot be loaded.  */
struct onnx_model *onnx_model_load (const char *path, int threads);
void onnx_model_free (struct onnx_model *m);

/* Embed the N token ids IDS: run the model, average its outputs over
   the tokens and normalize the result to unit length.  On success
   store a freshly allocated vector in *VEC and its length in *DIM and
   return true.  */
bool onnx_embed (struct onnx_model *m, const int64_t *ids, size_t n,
                 float **vec, size_t *dim);

#endif /* EMBED_ONNX_EMBED_H */
