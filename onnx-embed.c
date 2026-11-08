/* onnx-embed.c -- run a BERT-style embedding model with ONNX Runtime.
   Copyright (C) 2026 Martin Rixham

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.  */

#include "onnx-embed.h"
#include "common.h"

#include <error.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <onnxruntime/onnxruntime_c_api.h>

struct onnx_model
{
  const OrtApi *api;
  OrtEnv *env;
  OrtSession *session;
  OrtMemoryInfo *memory;
  size_t n_inputs;
  char **input_names;
  char *output_name;
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

struct onnx_model *
onnx_model_load (const char *path, int threads)
{
  struct onnx_model *m = xmalloc (sizeof *m);
  OrtSessionOptions *opts = NULL;
  OrtAllocator *alloc;
  const OrtApi *api;
  char *name;

  memset (m, 0, sizeof *m);
  api = m->api = OrtGetApiBase ()->GetApi (ORT_API_VERSION);
  if (!api)
    {
      error (0, 0, "ONNX Runtime does not provide API version %d",
             ORT_API_VERSION);
      free (m);
      return NULL;
    }
  if (failed (api, api->CreateEnv (ORT_LOGGING_LEVEL_WARNING, "embed",
                                   &m->env), "cannot start ONNX Runtime")
      || failed (api, api->CreateSessionOptions (&opts), "session options")
      || failed (api, api->SetIntraOpNumThreads (opts, threads),
                 "session options")
      /* ORT_ENABLE_ALL breaks the float16 model: its layer norm fusion
         refers to a cast it has not inserted.  */
      || failed (api, api->SetSessionGraphOptimizationLevel
                 (opts, ORT_ENABLE_EXTENDED), "session options")
      || failed (api, api->CreateSession (m->env, path, opts, &m->session),
                 path)
      || failed (api, api->CreateCpuMemoryInfo (OrtArenaAllocator,
                                                OrtMemTypeDefault,
                                                &m->memory), "memory info")
      || failed (api, api->GetAllocatorWithDefaultOptions (&alloc),
                 "allocator")
      || failed (api, api->SessionGetInputCount (m->session, &m->n_inputs),
                 path))
    goto fail;

  m->input_names = xmalloc (m->n_inputs * sizeof *m->input_names);
  for (size_t i = 0; i < m->n_inputs; i++)
    {
      if (failed (api, api->SessionGetInputName (m->session, i, alloc,
                                                 &name), path))
        goto fail;
      m->input_names[i] = xstrdup (name);
      failed (api, api->AllocatorFree (alloc, name), "allocator");
    }
  /* The first output is the last hidden state, one vector per
     token.  */
  if (failed (api, api->SessionGetOutputName (m->session, 0, alloc, &name),
              path))
    goto fail;
  m->output_name = xstrdup (name);
  failed (api, api->AllocatorFree (alloc, name), "allocator");
  api->ReleaseSessionOptions (opts);
  return m;

 fail:
  if (opts)
    api->ReleaseSessionOptions (opts);
  onnx_model_free (m);
  return NULL;
}

void
onnx_model_free (struct onnx_model *m)
{
  if (!m)
    return;
  if (m->input_names)
    for (size_t i = 0; i < m->n_inputs; i++)
      free (m->input_names[i]);
  free (m->input_names);
  free (m->output_name);
  if (m->memory)
    m->api->ReleaseMemoryInfo (m->memory);
  if (m->session)
    m->api->ReleaseSession (m->session);
  if (m->env)
    m->api->ReleaseEnv (m->env);
  free (m);
}

bool
onnx_embed (struct onnx_model *m, const int64_t *ids, size_t n,
            float **vec, size_t *dim)
{
  const OrtApi *api = m->api;
  int64_t shape[2] = { 1, (int64_t) n };
  int64_t *ones = xmalloc (n * sizeof *ones);
  int64_t *zeros = xmalloc (n * sizeof *zeros);
  OrtValue **inputs = xmalloc (m->n_inputs * sizeof *inputs);
  OrtValue *output = NULL;
  OrtTensorTypeAndShapeInfo *info = NULL;
  int64_t out_shape[3];
  size_t n_dims;
  float *hidden, *v = NULL;
  bool ok = false;
  double norm = 0;

  for (size_t i = 0; i < n; i++)
    {
      ones[i] = 1;
      zeros[i] = 0;
    }
  memset (inputs, 0, m->n_inputs * sizeof *inputs);
  for (size_t i = 0; i < m->n_inputs; i++)
    {
      const char *name = m->input_names[i];
      const int64_t *data = strcmp (name, "input_ids") == 0 ? ids
        : strcmp (name, "attention_mask") == 0 ? ones : zeros;

      if (failed (api, api->CreateTensorWithDataAsOrtValue
                  (m->memory, (void *) data, n * sizeof *data, shape, 2,
                   ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &inputs[i]),
                  "cannot make input tensor"))
        goto out;
    }

  if (failed (api, api->Run (m->session, NULL,
                             (const char *const *) m->input_names,
                             (const OrtValue *const *) inputs, m->n_inputs,
                             (const char *const *) &m->output_name, 1,
                             &output), "cannot run model")
      || failed (api, api->GetTensorTypeAndShape (output, &info),
                 "output shape")
      || failed (api, api->GetDimensionsCount (info, &n_dims),
                 "output shape"))
    goto out;
  ONNXTensorElementDataType type;
  if (failed (api, api->GetTensorElementType (info, &type), "output type")
      || type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
      || n_dims != 3
      || failed (api, api->GetDimensions (info, out_shape, 3),
                 "output shape")
      || out_shape[0] != 1 || out_shape[1] != (int64_t) n
      || out_shape[2] <= 0)
    {
      error (0, 0, "unexpected output type or shape from the model");
      goto out;
    }
  if (failed (api, api->GetTensorMutableData (output, (void **) &hidden),
              "output data"))
    goto out;

  /* Mean pooling over the tokens, then unit length.  */
  *dim = out_shape[2];
  v = xmalloc (*dim * sizeof *v);
  for (size_t d = 0; d < *dim; d++)
    {
      double sum = 0;

      for (size_t t = 0; t < n; t++)
        sum += hidden[t * *dim + d];
      v[d] = sum / n;
      norm += (double) v[d] * v[d];
    }
  norm = sqrt (norm);
  if (norm > 0)
    for (size_t d = 0; d < *dim; d++)
      v[d] /= norm;
  *vec = v;
  ok = true;

 out:
  if (info)
    api->ReleaseTensorTypeAndShapeInfo (info);
  if (output)
    api->ReleaseValue (output);
  for (size_t i = 0; i < m->n_inputs; i++)
    if (inputs[i])
      api->ReleaseValue (inputs[i]);
  free (inputs);
  free (ones);
  free (zeros);
  return ok;
}
