/* fake-model.h -- a stand-in for the embedding model, for tests.
   Copyright (C) 2026 Martin Rixham

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   fake-model.c implements model.h.  model_open fails if the
   environment variable FAKE_MODEL is "fail", and otherwise appends
   "open gpu=1" or "open gpu=0" and a line "----" to the file named by
   FAKE_MODEL_LOG, if set.  model_embed appends each text and a line
   "----" to the same file and returns a vector with 0.05 plus the
   number of occurrences (ignoring case) of fake_keywords[I] in
   dimension I, normalized, unless the text contains one of these
   markers:

     FAKE_500    fail, with a message
     FAKE_DIM3   return a 3-dimensional vector
     FAKE_ZERO   return a vector of zeros
     FAKE_SLOW   wait 3 seconds, or until a signal arrives, first

   vision_open fails if the environment variable FAKE_VISION is "fail".
   vision_embed returns, normalized, 0.05 plus the mean of the red,
   green and blue channels (from 0 to 1) in dimensions 0, 1 and 2, and
   0.05 in the others.  It logs nothing.  */

#ifndef FAKE_MODEL_H
#define FAKE_MODEL_H

/* Number of dimensions of the vectors the fake model returns.  */
#define FAKE_DIM 8

extern const char *const fake_keywords[FAKE_DIM];

#endif /* FAKE_MODEL_H */
