/* whisper.c -- transcribe speech with OpenAI's Whisper, in ONNX Runtime.
   Copyright (C) 2026 Martin Rixham

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   The audio is turned into a log mel spectrogram exactly as Hugging
   Face's WhisperFeatureExtractor does it, and the encoder's output is
   decoded greedily, as transcription in the language the model
   detects, without timestamps.  The merged decoder takes either no
   past (on the first step) or the keys and values it returned on the
   step before, so that each step needs only the newest token.

   The encoder, which does most of the work for little speech, takes
   input of one shape, so it can run on a GPU through OpenVINO; the
   decoder, whose input grows with every token, stays on the CPU.

   Whisper makes things up when there is no speech, so a window that is
   all but silent is not even given to it, and a window is dropped, as
   OpenAI's own decoder drops it, when the model both thinks
   it likely there is no speech and is unsure of what it decoded; and
   decoding stops when it falls into a loop, repeating itself.  */

#include "whisper.h"
#include "common.h"
#include "ov-embed.h"

#include <complex.h>
#include <error.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <onnxruntime/onnxruntime_c_api.h>

/* Analysis parameters of the spectrogram.  */
#define N_FFT 400
#define HOP 160
#define N_BINS (N_FFT / 2 + 1)

/* Special tokens of the multilingual models.  */
#define TOKEN_EOT 50257
#define TOKEN_SOT 50258
#define TOKEN_FIRST_LANGUAGE 50259
#define TOKEN_LAST_LANGUAGE 50357
#define TOKEN_TRANSCRIBE 50359
#define TOKEN_NO_SPEECH 50362
#define TOKEN_NO_TIMESTAMPS 50363

/* The most tokens to decode from one window, as OpenAI's decoder
   allows.  */
#define MAX_TOKENS 224

/* Sound quieter than this, as a root mean square (-60 dB of full
   scale), is taken to be silence.  */
#define SILENCE 0.001

/* When to drop a window as having no speech.  */
#define NO_SPEECH_THRESHOLD 0.6
#define LOGPROB_THRESHOLD -1.0

/* The model's generation_config.json: tokens that are never to be
   produced, mostly symbols that are not speech.  */
static const int suppress[] = {
  1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63,
  90, 91, 92, 93, 359, 503, 522, 542, 873, 893, 902, 918, 922, 931, 1350,
  1853, 1982, 2460, 2627, 3246, 3253, 3268, 3536, 3846, 3961, 4183, 4667,
  6585, 6647, 7273, 9061, 9383, 10428, 10929, 11938, 12033, 12331, 12562,
  13793, 14157, 14635, 15265, 15618, 16553, 16604, 18362, 18956, 20075,
  21675, 22520, 26130, 26161, 26435, 28279, 29464, 31650, 32302, 32470,
  36865, 42863, 47425, 49870, 50254
};

/* Nor these as the first token: a space, or the end.  */
static const int begin_suppress[] = { 220, TOKEN_EOT };

struct whisper
{
  const OrtApi *api;
  OrtEnv *env;
  OrtSession *encoder, *decoder;
  struct ov_embed *gpu_encoder;  /* In place of ENCODER, if not NULL.  */
  OrtMemoryInfo *memory;
  OrtAllocator *alloc;
  char *encoder_input, *encoder_output;
  size_t n_inputs, n_outputs;
  char **input_names, **output_names;
  /* For each decoder input that takes a past key or value, the index
     of the output that gives its next value, or -1 for others.  */
  int *present_of;
  size_t logits_index;
  /* The bytes each ordinary token stands for, and their lengths.  */
  char **tokens;
  size_t *token_len;
  size_t n_tokens;
  float *mel;
};

/* Report STATUS, if it is an error, and return true if it was.  */
static bool
failed (const OrtApi *api, OrtStatus *status, const char *what)
{
  if (!status)
    return false;
  error (0, 0, "%s: %s", what, api->GetErrorMessage (status));
  api->ReleaseStatus (status);
  return true;
}

/* The spectrogram.  */

/* Convert between hertz and the mel scale of Slaney's Auditory Toolbox,
   as librosa and Whisper do: linear below 1 kHz, logarithmic above.  */
static double
hz_to_mel (double hz)
{
  if (hz < 1000)
    return 3 * hz / 200;
  return 15 + 27 * log (hz / 1000) / log (6.4);
}

static double
mel_to_hz (double mel)
{
  if (mel < 15)
    return 200 * mel / 3;
  return 1000 * exp (log (6.4) * (mel - 15) / 27);
}

/* The mel filter bank: WHISPER_MELS triangles over N_BINS frequency
   bins, from 0 to 8 kHz, each normalized to unit area.  */
static const float *
mel_filters (void)
{
  static float filters[WHISPER_MELS][N_BINS];
  static bool done;
  double edges[WHISPER_MELS + 2];
  double top = hz_to_mel (WHISPER_RATE / 2);

  if (done)
    return &filters[0][0];
  for (int i = 0; i < WHISPER_MELS + 2; i++)
    edges[i] = mel_to_hz (top * i / (WHISPER_MELS + 1));
  for (int m = 0; m < WHISPER_MELS; m++)
    {
      double area = 2 / (edges[m + 2] - edges[m]);

      for (int b = 0; b < N_BINS; b++)
        {
          double f = (double) b * (WHISPER_RATE / 2) / (N_BINS - 1);
          double down = (f - edges[m]) / (edges[m + 1] - edges[m]);
          double up = (edges[m + 2] - f) / (edges[m + 2] - edges[m + 1]);
          double w = down < up ? down : up;

          filters[m][b] = w > 0 ? w * area : 0;
        }
    }
  done = true;
  return &filters[0][0];
}

/* Compute the discrete Fourier transform of the N values of IN, read
   STRIDE apart, into OUT, with TW holding exp (-2 pi i k / N_FFT) for
   each K below N_FFT, which N must divide.  This is the plain mixed
   radix algorithm; N_FFT is 2^4 * 5^2.  */
static void
fft (const double complex *in, size_t stride, size_t n, double complex *out,
     const double complex *tw)
{
  double complex tmp[N_FFT];
  size_t p, m;

  if (n == 1)
    {
      out[0] = in[0];
      return;
    }
  for (p = 2; n % p != 0; p++)
    ;
  m = n / p;
  for (size_t r = 0; r < p; r++)
    fft (in + r * stride, stride * p, m, out + r * m, tw);
  for (size_t k = 0; k < n; k++)
    {
      double complex sum = 0;

      for (size_t r = 0; r < p; r++)
        sum += out[r * m + k % m] * tw[(r * k % n) * (N_FFT / n)];
      tmp[k] = sum;
    }
  memcpy (out, tmp, n * sizeof *out);
}

void
whisper_log_mel (const float *pcm, size_t n, float *mel)
{
  static double complex tw[N_FFT];
  static double window[N_FFT];
  const float *filters = mel_filters ();
  int first[WHISPER_MELS], last[WHISPER_MELS];
  float *x = xmalloc ((WHISPER_SAMPLES + N_FFT) * sizeof *x);
  double complex frame[N_FFT], spectrum[N_FFT];
  double power[N_BINS];
  float top = -INFINITY;

  for (int k = 0; k < N_FFT; k++)
    {
      tw[k] = cexp (-2 * M_PI * I * k / N_FFT);
      /* The periodic Hann window.  */
      window[k] = 0.5 - 0.5 * cos (2 * M_PI * k / N_FFT);
    }

  /* Each filter is zero but for a few bins.  */
  for (int m = 0; m < WHISPER_MELS; m++)
    {
      first[m] = N_BINS;
      last[m] = 0;
      for (int b = 0; b < N_BINS; b++)
        if (filters[m * N_BINS + b] != 0)
          {
            if (b < first[m])
              first[m] = b;
            last[m] = b + 1;
          }
    }

  /* Pad to 30 seconds with silence, and then by half a frame at each
     end, reflecting the signal, so that frames are centred on their
     times.  */
  if (n > WHISPER_SAMPLES)
    n = WHISPER_SAMPLES;
  for (long i = 0; i < WHISPER_SAMPLES + N_FFT; i++)
    {
      long j = i - N_FFT / 2;

      if (j < 0)
        j = -j;
      else if (j >= WHISPER_SAMPLES)
        j = 2 * (WHISPER_SAMPLES - 1) - j;
      x[i] = (size_t) j < n ? pcm[j] : 0;
    }

  /* The last frame, centred on the very end, is dropped.  */
  for (int f = 0; f < WHISPER_FRAMES; f++)
    {
      bool silent = true;

      for (int k = 0; k < N_FFT; k++)
        {
          frame[k] = x[f * HOP + k] * window[k];
          silent = silent && x[f * HOP + k] == 0;
        }
      /* The padding of a short recording needs no transform.  */
      if (silent)
        memset (power, 0, sizeof power);
      else
        {
          fft (frame, 1, N_FFT, spectrum, tw);
          for (int b = 0; b < N_BINS; b++)
            power[b] = creal (spectrum[b]) * creal (spectrum[b])
              + cimag (spectrum[b]) * cimag (spectrum[b]);
        }
      for (int m = 0; m < WHISPER_MELS; m++)
        {
          double sum = 0;
          float v;

          for (int b = first[m]; b < last[m]; b++)
            sum += filters[m * N_BINS + b] * power[b];
          v = log10 (sum > 1e-10 ? sum : 1e-10);
          mel[m * WHISPER_FRAMES + f] = v;
          if (v > top)
            top = v;
        }
    }
  free (x);

  /* At most 80 dB of range, scaled to about -1 to 1.  */
  for (size_t i = 0; i < WHISPER_MELS * WHISPER_FRAMES; i++)
    {
      float v = mel[i] > top - 8 ? mel[i] : top - 8;
      mel[i] = (v + 4) / 4;
    }
}

/* The tokens.  */

/* Append the UTF-8 encoding of C to B.  */
static void
put_utf8 (struct buf *b, unsigned c)
{
  char s[4];
  size_t n;

  if (c < 0x80)
    s[0] = c, n = 1;
  else if (c < 0x800)
    s[0] = 0xc0 | c >> 6, s[1] = 0x80 | (c & 0x3f), n = 2;
  else if (c < 0x10000)
    s[0] = 0xe0 | c >> 12, s[1] = 0x80 | (c >> 6 & 0x3f),
      s[2] = 0x80 | (c & 0x3f), n = 3;
  else
    s[0] = 0xf0 | c >> 18, s[1] = 0x80 | (c >> 12 & 0x3f),
      s[2] = 0x80 | (c >> 6 & 0x3f), s[3] = 0x80 | (c & 0x3f), n = 4;
  buf_append (b, s, n);
}

/* Read a JSON string starting at the quote at *P into B, decoding its
   escapes, and advance *P past it.  Return false if it is malformed.  */
static bool
json_string (const char **p, struct buf *b)
{
  const char *s = *p;

  if (*s++ != '"')
    return false;
  while (*s && *s != '"')
    {
      if (*s != '\\')
        {
          buf_append (b, s++, 1);
          continue;
        }
      s++;
      switch (*s)
        {
        case 'n': buf_puts (b, "\n"); break;
        case 't': buf_puts (b, "\t"); break;
        case 'r': buf_puts (b, "\r"); break;
        case 'b': buf_puts (b, "\b"); break;
        case 'f': buf_puts (b, "\f"); break;
        case 'u':
          {
            unsigned c = 0;

            for (int i = 1; i <= 4; i++)
              {
                char h = s[i];
                int d = h >= '0' && h <= '9' ? h - '0'
                  : h >= 'a' && h <= 'f' ? h - 'a' + 10
                  : h >= 'A' && h <= 'F' ? h - 'A' + 10 : -1;

                if (d < 0)
                  return false;
                c = c * 16 + d;
              }
            s += 4;
            /* A surrogate pair.  */
            if (c >= 0xd800 && c < 0xdc00 && s[1] == '\\' && s[2] == 'u')
              {
                unsigned lo = strtoul ((char[5]) { s[3], s[4], s[5], s[6],
                                                   0 }, NULL, 16);
                if (lo >= 0xdc00 && lo < 0xe000)
                  {
                    c = 0x10000 + ((c - 0xd800) << 10) + (lo - 0xdc00);
                    s += 6;
                  }
              }
            put_utf8 (b, c);
          }
          break;
        case '\0':
          return false;
        default:
          buf_append (b, s, 1);
        }
      s++;
    }
  if (*s != '"')
    return false;
  *p = s + 1;
  return true;
}

/* The byte that each code point of a GPT-2 token string stands for, or
   -1: printable bytes stand for themselves, and the others for code
   points from 256 up.  */
static int
token_byte (unsigned c)
{
  static int bytes[512];
  static bool done;

  if (!done)
    {
      int next = 256;

      for (int i = 0; i < 512; i++)
        bytes[i] = -1;
      for (int b = 0; b < 256; b++)
        {
          bool printable = (b >= '!' && b <= '~') || (b >= 0xa1 && b <= 0xac)
            || (b >= 0xae && b <= 0xff);
          bytes[printable ? b : next++] = b;
        }
      done = true;
    }
  return c < 512 ? bytes[c] : -1;
}

/* Decode the UTF-8 string S of N bytes, a token in vocab.json, to the
   bytes it stands for, in place.  Return their number.  */
static size_t
token_bytes (char *s, size_t n)
{
  size_t in = 0, out = 0;

  while (in < n)
    {
      unsigned char c = s[in];
      unsigned cp;
      int len, b;

      len = c < 0x80 ? 1 : c < 0xe0 ? 2 : c < 0xf0 ? 3 : 4;
      cp = len == 1 ? c : c & (0x3f >> (len - 1));
      for (int i = 1; i < len && in + i < n; i++)
        cp = cp << 6 | (s[in + i] & 0x3f);
      in += len;
      b = token_byte (cp);
      if (b >= 0)
        s[out++] = b;
    }
  return out;
}

/* Load the tokens from vocab.json at PATH into W.  */
static bool
load_vocab (struct whisper *w, const char *path)
{
  FILE *f = fopen (path, "r");
  struct buf text, key;
  const char *p;
  bool ok = false;

  if (!f)
    {
      error (0, errno, "%s", path);
      return false;
    }
  buf_init (&text);
  for (char chunk[65536];;)
    {
      size_t n = fread (chunk, 1, sizeof chunk, f);
      if (n == 0)
        break;
      buf_append (&text, chunk, n);
    }
  fclose (f);

  w->n_tokens = TOKEN_EOT;
  w->tokens = xmalloc (w->n_tokens * sizeof *w->tokens);
  w->token_len = xmalloc (w->n_tokens * sizeof *w->token_len);
  memset (w->tokens, 0, w->n_tokens * sizeof *w->tokens);
  p = text.data;
  while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')
    p++;
  if (*p++ != '{')
    goto out;
  for (;;)
    {
      char *end;
      long id;

      while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t'
             || *p == ',')
        p++;
      if (*p == '}')
        break;
      buf_init (&key);
      if (!json_string (&p, &key))
        {
          buf_free (&key);
          goto out;
        }
      while (*p == ' ' || *p == ':')
        p++;
      id = strtol (p, &end, 10);
      if (end == p)
        {
          buf_free (&key);
          goto out;
        }
      p = end;
      if (id >= 0 && (size_t) id < w->n_tokens && !w->tokens[id])
        {
          w->token_len[id] = token_bytes (key.data, key.len);
          w->tokens[id] = key.data;
        }
      else
        buf_free (&key);
    }
  ok = true;
  for (size_t i = 0; i < w->n_tokens; i++)
    if (!w->tokens[i])
      ok = false;

 out:
  if (!ok)
    error (0, 0, "%s: not a Whisper vocabulary", path);
  buf_free (&text);
  return ok;
}

/* The models.  */

static OrtSession *
open_session (struct whisper *w, const char *path, int threads)
{
  const OrtApi *api = w->api;
  OrtSessionOptions *opts = NULL;
  OrtSession *s = NULL;

  if (failed (api, api->CreateSessionOptions (&opts), "session options")
      || failed (api, api->SetIntraOpNumThreads (opts, threads),
                 "session options")
      || failed (api, api->SetSessionGraphOptimizationLevel
                 (opts, ORT_ENABLE_EXTENDED), "session options"))
    goto out;
  failed (api, api->CreateSession (w->env, path, opts, &s), path);
 out:
  if (opts)
    api->ReleaseSessionOptions (opts);
  return s;
}

/* Return the name of input (or, if OUTPUT, output) I of S, freshly
   allocated, or NULL.  */
static char *
io_name (struct whisper *w, OrtSession *s, size_t i, bool output)
{
  const OrtApi *api = w->api;
  char *name, *copy;

  if (failed (api, output ? api->SessionGetOutputName (s, i, w->alloc, &name)
              : api->SessionGetInputName (s, i, w->alloc, &name),
              "model names"))
    return NULL;
  copy = xstrdup (name);
  failed (api, api->AllocatorFree (w->alloc, name), "allocator");
  return copy;
}

struct whisper *
whisper_load (const char *encoder, const char *decoder, const char *vocab,
              int threads, const char *gpu_cache)
{
  struct whisper *w = xmalloc (sizeof *w);
  const OrtApi *api;
  bool have_logits = false;

  memset (w, 0, sizeof *w);
  api = w->api = OrtGetApiBase ()->GetApi (ORT_API_VERSION);
  if (!api)
    {
      error (0, 0, "ONNX Runtime does not provide API version %d",
             ORT_API_VERSION);
      free (w);
      return NULL;
    }
  if (!load_vocab (w, vocab)
      || failed (api, api->CreateEnv (ORT_LOGGING_LEVEL_WARNING, "embed",
                                      &w->env), "cannot start ONNX Runtime")
      || failed (api, api->CreateCpuMemoryInfo (OrtArenaAllocator,
                                                OrtMemTypeDefault,
                                                &w->memory), "memory info")
      || failed (api, api->GetAllocatorWithDefaultOptions (&w->alloc),
                 "allocator"))
    goto fail;
  if (gpu_cache)
    {
      int64_t dims[3] = { 1, WHISPER_MELS, WHISPER_FRAMES };
      const char *why;

      w->gpu_encoder = ov_embed_load_fixed (encoder, gpu_cache, 3, dims,
                                            &why);
      if (!w->gpu_encoder)
        error (0, 0, "%s; transcribing on the CPU instead", why);
    }
  if ((!w->gpu_encoder
       && (!(w->encoder = open_session (w, encoder, threads))
           || !(w->encoder_input = io_name (w, w->encoder, 0, false))
           || !(w->encoder_output = io_name (w, w->encoder, 0, true))))
      || !(w->decoder = open_session (w, decoder, threads))
      || failed (api, api->SessionGetInputCount (w->decoder, &w->n_inputs),
                 decoder)
      || failed (api, api->SessionGetOutputCount (w->decoder,
                                                  &w->n_outputs), decoder))
    goto fail;

  w->input_names = xmalloc (w->n_inputs * sizeof *w->input_names);
  w->output_names = xmalloc (w->n_outputs * sizeof *w->output_names);
  w->present_of = xmalloc (w->n_inputs * sizeof *w->present_of);
  memset (w->input_names, 0, w->n_inputs * sizeof *w->input_names);
  memset (w->output_names, 0, w->n_outputs * sizeof *w->output_names);
  for (size_t i = 0; i < w->n_outputs; i++)
    {
      if (!(w->output_names[i] = io_name (w, w->decoder, i, true)))
        goto fail;
      if (strcmp (w->output_names[i], "logits") == 0)
        {
          w->logits_index = i;
          have_logits = true;
        }
    }
  for (size_t i = 0; i < w->n_inputs; i++)
    {
      const char *prefix = "past_key_values.";
      char *name = w->input_names[i] = io_name (w, w->decoder, i, false);

      if (!name)
        goto fail;
      w->present_of[i] = -1;
      if (strncmp (name, prefix, strlen (prefix)) == 0)
        {
          for (size_t j = 0; j < w->n_outputs; j++)
            if (strncmp (w->output_names[j], "present.", 8) == 0
                && strcmp (w->output_names[j] + 8,
                           name + strlen (prefix)) == 0)
              w->present_of[i] = j;
          if (w->present_of[i] < 0)
            {
              error (0, 0, "%s: no output for %s", decoder, name);
              goto fail;
            }
        }
      else if (strcmp (name, "input_ids") != 0
               && strcmp (name, "encoder_hidden_states") != 0
               && strcmp (name, "use_cache_branch") != 0)
        {
          error (0, 0, "%s: unexpected input %s", decoder, name);
          goto fail;
        }
    }
  if (!have_logits)
    {
      error (0, 0, "%s: no logits", decoder);
      goto fail;
    }
  w->mel = xmalloc (WHISPER_MELS * WHISPER_FRAMES * sizeof *w->mel);
  return w;

 fail:
  whisper_free (w);
  return NULL;
}

void
whisper_free (struct whisper *w)
{
  if (!w)
    return;
  for (size_t i = 0; w->input_names && i < w->n_inputs; i++)
    free (w->input_names[i]);
  for (size_t i = 0; w->output_names && i < w->n_outputs; i++)
    free (w->output_names[i]);
  free (w->input_names);
  free (w->output_names);
  free (w->present_of);
  free (w->encoder_input);
  free (w->encoder_output);
  for (size_t i = 0; w->tokens && i < w->n_tokens; i++)
    free (w->tokens[i]);
  free (w->tokens);
  free (w->token_len);
  free (w->mel);
  ov_embed_free (w->gpu_encoder);
  if (w->encoder)
    w->api->ReleaseSession (w->encoder);
  if (w->decoder)
    w->api->ReleaseSession (w->decoder);
  if (w->memory)
    w->api->ReleaseMemoryInfo (w->memory);
  if (w->env)
    w->api->ReleaseEnv (w->env);
  free (w);
}

const char *
whisper_device (const struct whisper *w)
{
  return w->gpu_encoder ? "GPU" : "CPU";
}

/* Run the encoder on W's spectrogram, storing its output in *HIDDEN,
   and, if that output lives in a buffer of the caller's, the buffer in
   *DATA.  */
static bool
encode (struct whisper *w, OrtValue **hidden, float **data)
{
  const OrtApi *api = w->api;
  int64_t shape[3] = { 1, WHISPER_MELS, WHISPER_FRAMES };
  int64_t dims[OV_MAX_RANK];
  OrtValue *features = NULL;
  size_t rank;
  bool ok;

  if (w->gpu_encoder)
    {
      if (!ov_run_fixed (w->gpu_encoder, w->mel, data, dims, &rank))
        return false;
      if (rank != 3 || dims[0] != 1 || dims[1] <= 0 || dims[2] <= 0)
        {
          error (0, 0, "unexpected output from the Whisper encoder");
          return false;
        }
      return !failed (api, api->CreateTensorWithDataAsOrtValue
                      (w->memory, *data,
                       dims[1] * dims[2] * sizeof **data, dims, 3,
                       ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, hidden),
                      "tensor");
    }
  ok = !failed (api, api->CreateTensorWithDataAsOrtValue
                (w->memory, w->mel,
                 WHISPER_MELS * WHISPER_FRAMES * sizeof *w->mel, shape, 3,
                 ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &features), "tensor")
    && !failed (api, api->Run (w->encoder, NULL,
                               (const char *const *) &w->encoder_input,
                               (const OrtValue *const *) &features, 1,
                               (const char *const *) &w->encoder_output, 1,
                               hidden), "cannot run the encoder");
  if (features)
    api->ReleaseValue (features);
  return ok;
}

/* Decoding.  */

/* The state of one transcription.  */
struct decoding
{
  OrtValue *hidden;             /* The encoder's output.  */
  OrtValue **past;              /* For each decoder input, or NULL.  */
  bool first;                   /* No past yet.  */
  float *logits;                /* For the last token of the last step.  */
  size_t vocab;
};

/* Run the decoder on the N tokens IDS, after those before, and keep
   the logits for the last in D.  */
static bool
step (struct whisper *w, struct decoding *d, const int64_t *ids, size_t n)
{
  const OrtApi *api = w->api;
  OrtValue **in = xmalloc (w->n_inputs * sizeof *in);
  OrtValue **out = xmalloc (w->n_outputs * sizeof *out);
  OrtValue *ids_value = NULL, *branch = NULL;
  OrtTensorTypeAndShapeInfo *info = NULL;
  int64_t ids_shape[2] = { 1, (int64_t) n }, one = 1, dims[3];
  bool use_cache = !d->first, ok = false;
  size_t n_dims;
  float *logits;

  memset (out, 0, w->n_outputs * sizeof *out);
  if (failed (api, api->CreateTensorWithDataAsOrtValue
              (w->memory, (void *) ids, n * sizeof *ids, ids_shape, 2,
               ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &ids_value), "tensor")
      || failed (api, api->CreateTensorWithDataAsOrtValue
                 (w->memory, &use_cache, sizeof use_cache, &one, 1,
                  ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, &branch), "tensor"))
    goto out;
  for (size_t i = 0; i < w->n_inputs; i++)
    {
      const char *name = w->input_names[i];

      in[i] = strcmp (name, "input_ids") == 0 ? ids_value
        : strcmp (name, "use_cache_branch") == 0 ? branch
        : strcmp (name, "encoder_hidden_states") == 0 ? d->hidden
        : d->past[i];
    }
  if (failed (api, api->Run (w->decoder, NULL,
                             (const char *const *) w->input_names,
                             (const OrtValue *const *) in, w->n_inputs,
                             (const char *const *) w->output_names,
                             w->n_outputs, out), "cannot run the decoder"))
    goto out;

  if (failed (api, api->GetTensorTypeAndShape (out[w->logits_index], &info),
              "logits")
      || failed (api, api->GetDimensionsCount (info, &n_dims), "logits")
      || n_dims != 3
      || failed (api, api->GetDimensions (info, dims, 3), "logits")
      || dims[1] != (int64_t) n || dims[2] <= TOKEN_NO_TIMESTAMPS
      || failed (api, api->GetTensorMutableData (out[w->logits_index],
                                                 (void **) &logits),
                 "logits"))
    {
      error (0, 0, "unexpected logits from the Whisper decoder");
      goto out;
    }
  d->vocab = dims[2];
  d->logits = xrealloc (d->logits, d->vocab * sizeof *d->logits);
  memcpy (d->logits, logits + (n - 1) * d->vocab,
          d->vocab * sizeof *d->logits);

  /* The keys and values to take next time.  The cross attention's,
     over the encoder's output, come only from the first step: later
     steps give placeholders.  */
  for (size_t i = 0; i < w->n_inputs; i++)
    {
      int j = w->present_of[i];

      if (j < 0)
        continue;
      if (d->first || !strstr (w->output_names[j], ".encoder."))
        {
          api->ReleaseValue (d->past[i]);
          d->past[i] = out[j];
          out[j] = NULL;
        }
    }
  d->first = false;
  ok = true;

 out:
  if (info)
    api->ReleaseTensorTypeAndShapeInfo (info);
  for (size_t i = 0; i < w->n_outputs; i++)
    if (out[i])
      api->ReleaseValue (out[i]);
  if (ids_value)
    api->ReleaseValue (ids_value);
  if (branch)
    api->ReleaseValue (branch);
  free (in);
  free (out);
  return ok;
}

/* Return log (sum (exp (L))) over the N floats of L.  */
static double
log_sum_exp (const float *l, size_t n)
{
  double top = -INFINITY, sum = 0;

  for (size_t i = 0; i < n; i++)
    if (l[i] > top)
      top = l[i];
  for (size_t i = 0; i < n; i++)
    sum += exp (l[i] - top);
  return top + log (sum);
}

/* If the N tokens of T end in a loop, the same P tokens three times
   over, return the number of tokens to keep, with the loop once.
   Otherwise return N.  */
static size_t
cut_loop (const int64_t *t, size_t n)
{
  for (size_t p = 1; p <= 16 && 3 * p <= n; p++)
    {
      bool loop = true;

      for (size_t i = n - 2 * p; i < n && loop; i++)
        loop = t[i] == t[i - p];
      if (loop)
        return n - 2 * p;
    }
  return n;
}

bool
whisper_transcribe (struct whisper *w, const float *pcm, size_t n,
                    struct buf *out)
{
  const OrtApi *api = w->api;
  int64_t past_shape[4] = { 1, 8, 0, 64 };
  int64_t tokens[MAX_TOKENS + 1], prompt[3];
  struct decoding d = { 0 };
  float *hidden = NULL;
  size_t n_tokens = 0;
  double logprob = 0, no_speech;
  int language = TOKEN_FIRST_LANGUAGE;
  bool ok = false;
  double energy = 0;

  if (n > WHISPER_SAMPLES)
    n = WHISPER_SAMPLES;
  for (size_t i = 0; i < n; i++)
    energy += (double) pcm[i] * pcm[i];
  if (n == 0 || sqrt (energy / n) < SILENCE)
    return true;

  whisper_log_mel (pcm, n, w->mel);
  d.first = true;
  d.past = xmalloc (w->n_inputs * sizeof *d.past);
  memset (d.past, 0, w->n_inputs * sizeof *d.past);
  if (!encode (w, &d.hidden, &hidden))
    goto out;

  /* Empty pasts for the first step, which ignores them anyway.  */
  for (size_t i = 0; i < w->n_inputs; i++)
    if (w->present_of[i] >= 0
        && failed (api, api->CreateTensorAsOrtValue
                   (w->alloc, past_shape, 4,
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &d.past[i]),
                   "tensor"))
      goto out;

  /* The start: which language it is, and whether it is speech at
     all.  */
  prompt[0] = TOKEN_SOT;
  if (!step (w, &d, prompt, 1))
    goto out;
  for (int t = TOKEN_FIRST_LANGUAGE; t <= TOKEN_LAST_LANGUAGE; t++)
    if (d.logits[t] > d.logits[language])
      language = t;
  no_speech = exp (d.logits[TOKEN_NO_SPEECH]
                   - log_sum_exp (d.logits, d.vocab));

  prompt[0] = language;
  prompt[1] = TOKEN_TRANSCRIBE;
  prompt[2] = TOKEN_NO_TIMESTAMPS;
  if (!step (w, &d, prompt, 3))
    goto out;
  while (n_tokens < MAX_TOKENS)
    {
      size_t best = 0;

      /* Only ordinary tokens, or the end.  */
      for (size_t i = TOKEN_EOT + 1; i < d.vocab; i++)
        d.logits[i] = -INFINITY;
      for (size_t i = 0; i < sizeof suppress / sizeof *suppress; i++)
        d.logits[suppress[i]] = -INFINITY;
      if (n_tokens == 0)
        for (size_t i = 0; i < 2; i++)
          d.logits[begin_suppress[i]] = -INFINITY;
      for (size_t i = 1; i <= TOKEN_EOT; i++)
        if (d.logits[i] > d.logits[best])
          best = i;
      logprob += d.logits[best] - log_sum_exp (d.logits, d.vocab);
      if (best == TOKEN_EOT)
        break;
      tokens[n_tokens++] = best;
      size_t keep = cut_loop (tokens, n_tokens);
      if (keep < n_tokens)
        {
          n_tokens = keep;
          break;
        }
      if (!step (w, &d, &tokens[n_tokens - 1], 1))
        goto out;
    }
  ok = true;

  /* Like OpenAI's decoder, believe the model when it thinks there is no
     speech, unless it is confident of what it heard anyway.  */
  if (no_speech > NO_SPEECH_THRESHOLD
      && logprob / (n_tokens + 1) < LOGPROB_THRESHOLD)
    goto out;
  for (size_t i = 0; i < n_tokens; i++)
    buf_append (out, w->tokens[tokens[i]], w->token_len[tokens[i]]);

 out:
  for (size_t i = 0; i < w->n_inputs; i++)
    if (d.past[i])
      api->ReleaseValue (d.past[i]);
  free (d.past);
  free (d.logits);
  if (d.hidden)
    api->ReleaseValue (d.hidden);
  free (hidden);
  return ok;
}
