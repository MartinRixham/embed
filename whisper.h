/* whisper.h -- transcribe speech with OpenAI's Whisper, in ONNX Runtime.
   Copyright (C) 2026 Martin Rixham

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.  */

#ifndef EMBED_WHISPER_H
#define EMBED_WHISPER_H

#include <stdbool.h>
#include <stddef.h>

struct buf;
struct whisper;

/* Whisper takes 30 seconds of 16 kHz mono audio at a time.  */
#define WHISPER_RATE 16000
#define WHISPER_SAMPLES (30 * WHISPER_RATE)

/* Its input: this many mel bands for each of this many frames.  */
#define WHISPER_MELS 80
#define WHISPER_FRAMES 3000

/* Load the encoder and merged decoder (as exported by Hugging Face's
   optimum) from ENCODER and DECODER and the tokens from VOCAB, a
   vocab.json, to run on THREADS threads of the CPU.  If GPU_CACHE is
   not NULL, run the encoder on the GPU through OpenVINO instead, if it
   can, caching compiled kernels there.  Report a problem and return
   NULL if they cannot be loaded.  */
struct whisper *whisper_load (const char *encoder, const char *decoder,
                              const char *vocab, int threads,
                              const char *gpu_cache);

/* Return "GPU" or "CPU", wherever W runs its encoder.  */
const char *whisper_device (const struct whisper *w);
void whisper_free (struct whisper *w);

/* Transcribe the N samples of PCM, at most WHISPER_SAMPLES, from -1 to
   1, in whatever language they are spoken, and append the text to OUT.
   Append nothing if there seems to be no speech.  Return false if the
   model fails.  */
bool whisper_transcribe (struct whisper *w, const float *pcm, size_t n,
                         struct buf *out);

/* Compute the log mel spectrogram of PCM as Whisper's feature
   extractor does, padding it with silence to WHISPER_SAMPLES, into the
   WHISPER_MELS rows of WHISPER_FRAMES floats of MEL.  */
void whisper_log_mel (const float *pcm, size_t n, float *mel);

#endif /* EMBED_WHISPER_H */
