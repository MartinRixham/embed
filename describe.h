/* describe.h -- turning a file into the vectors that stand for it.
   Copyright (C) 2026 Martin Rixham

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   embed-index uses these to build the store, and embed-query to embed
   a file given as the query when the store has no fresh record of
   it, so that both come to the same vectors.  */

#ifndef EMBED_DESCRIBE_H
#define EMBED_DESCRIBE_H

#include "common.h"

#include <magic.h>
#include <signal.h>
#include <stdbool.h>
#include <stddef.h>

struct model;
struct vision;

/* Set by a signal handler to cut work short.  */
extern volatile sig_atomic_t interrupted;

/* Seconds a helper program (ffmpeg, pdftotext, ...) may run.  */
extern long helper_timeout;

/* Whether to keep the vision and speech models off the GPU.  */
extern bool use_cpu;

/* Whether a file of type MIME can be embedded by what it shows.  */
bool picture_mime (const char *mime);

/* Return the MIME type of FILE, freshly allocated, or NULL.  */
char *file_mime (magic_t magic_mime, const char *file);

/* Build the text that represents FILE for the model, in OUT: its name,
   location (with HOME shortened to ~), type and, if WITH_CONTENT, a
   rendering of its content.  */
void describe_file (magic_t magic_mime, magic_t magic_desc, const char *file,
                    const char *home, bool with_content, struct buf *out);

/* Embed TEXT like model_embed, but fail unless the vector has DIM
   dimensions.  */
bool embed_checked (struct model *m, const char *text, size_t dim,
                    float **vec);

/* Return the vision model, opening it the first time it is needed, or
   NULL if it cannot be opened.  */
struct vision *get_vision (void);

/* Embed what FILE, of type MIME, shows: the picture, if it is an
   image, or the average of a few frames spread through it, if it is a
   video.  Return a freshly allocated unit vector of DIM floats, or
   NULL if FILE cannot be decoded.  */
float *embed_picture (struct vision *vis, const char *file, const char *mime,
                      size_t dim);

/* Embed FILE as embed-index does: its description by M, falling back
   to its name and type alone if that fails, into *VEC, and what it
   shows, if it is an image or video, into *PICTURE, or NULL.  Return
   false if even the fallback cannot be embedded.  */
bool embed_file (struct model *m, magic_t magic_mime, magic_t magic_desc,
                 const char *file, const char *home, size_t dim,
                 float **vec, float **picture);

/* Close the vision and speech models, if they were opened.  */
void close_media_models (void);

#endif /* EMBED_DESCRIBE_H */
