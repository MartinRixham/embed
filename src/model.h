/* model.h -- the models: nomic-embed-text and -vision v1.5, and Whisper.
   Copyright (C) 2026 Martin Rixham

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   The programs use the models only through these functions, so that
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

/* The image model, which embeds pictures into the same space as the
   text model, so that a text query can find them.  */
struct vision;

/* Images are given to vision_embed this many pixels square.  */
#define VISION_SIDE 224

/* Open the image model like model_open.  */
struct vision *vision_open (bool use_gpu);
void vision_close (struct vision *v);

/* Embed the picture RGB: VISION_SIDE rows of VISION_SIDE pixels, each
   three bytes of red, green and blue.  Return the vector like
   model_embed.  */
bool vision_embed (struct vision *v, const unsigned char *rgb, float **vec,
                   size_t *dim);

/* The speech model, which transcribes what is said, for the text
   model to embed.  */
struct speech;
struct buf;

/* It takes up to this many samples at a time, of mono sound at this
   rate, from -1 to 1.  */
#define SPEECH_RATE 16000
#define SPEECH_SAMPLES (30 * SPEECH_RATE)

/* Open the speech model like model_open.  */
struct speech *speech_open (bool use_gpu);
void speech_close (struct speech *s);

/* Transcribe the N samples of PCM, at most SPEECH_SAMPLES, appending
   the text to OUT; append nothing if there is no speech.  Return false
   if the model fails.  */
bool speech_transcribe (struct speech *s, const float *pcm, size_t n,
                        struct buf *out);

#endif /* EMBED_MODEL_H */
