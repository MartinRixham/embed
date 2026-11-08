/* model.h -- the embedding model, nomic-embed-text v1.5.
   Copyright (C) 2026 Martin Rixham

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   The programs use the model only through these functions, so that
   the tests can link a fake model in its place.  */

#ifndef EMBED_MODEL_H
#define EMBED_MODEL_H

#include <stdbool.h>
#include <stddef.h>

struct model;

/* Open the model, downloading its files first if they are missing.
   If USE_GPU, run it on the GPU through OpenVINO when possible, and
   otherwise (or if not USE_GPU) on the CPU through ONNX Runtime.
   Report problems and return NULL if it cannot be opened.  */
struct model *model_open (bool use_gpu);
void model_close (struct model *m);

/* Return "GPU" or "CPU", wherever M runs.  */
const char *model_device (const struct model *m);

/* Embed TEXT, cutting it off at the model's input limit.  On success
   store a freshly allocated unit vector in *VEC and its length in *DIM
   and return true.  */
bool model_embed (struct model *m, const char *text, float **vec,
                  size_t *dim);

#endif /* EMBED_MODEL_H */
