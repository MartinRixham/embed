/* ov-embed.c -- run a BERT-style embedding model on a GPU with OpenVINO.
   Copyright (C) 2026 Martin Rixham

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   OpenVINO's GPU plugin cannot compile this model with inputs of
   varying length, so it is compiled for a few fixed lengths, each the
   first time a text needs it, and shorter texts are padded with
   [PAD] tokens that the attention mask hides.  */

#include "ov-embed.h"
#include "common.h"

#include <error.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <openvino/c/openvino.h>

#define MAX_INPUTS 8

static const size_t lengths[] = { 128, 256, OV_EMBED_MAX };
#define N_LENGTHS (sizeof lengths / sizeof *lengths)

struct ov_embed
{
  ov_core_t *core;
  ov_model_t *model;
  char *cache_dir;
  size_t n_inputs;
  char *input_names[MAX_INPUTS];
  /* One compiled model and request for each of LENGTHS, or NULL.  */
  ov_compiled_model_t *compiled[N_LENGTHS];
  ov_infer_request_t *request[N_LENGTHS];
};

/* Report a failure of WHAT with status STATUS, and return true, if
   STATUS is not OK.  */
static bool
failed (ov_status_e status, const char *what)
{
  if (status == OK)
    return false;
  error (0, 0, "OpenVINO: %s: %s", what, ov_get_last_err_msg ());
  return true;
}

static bool
have_gpu (ov_core_t *core)
{
  ov_available_devices_t devices;
  bool found = false;

  if (ov_core_get_available_devices (core, &devices) != OK)
    return false;
  for (size_t i = 0; i < devices.size; i++)
    if (strncmp (devices.devices[i], "GPU", 3) == 0)
      found = true;
  ov_available_devices_free (&devices);
  return found;
}

/* Compile the model for input length LENGTHS[I], if not yet done.  */
static bool
compile_length (struct ov_embed *m, size_t i)
{
  ov_partial_shape_t shapes[MAX_INPUTS];
  int64_t dims[2] = { 1, (int64_t) lengths[i] };
  bool ok;

  if (m->request[i])
    return true;
  for (size_t j = 0; j < m->n_inputs; j++)
    if (ov_partial_shape_create_static (2, dims, &shapes[j]) != OK)
      return false;
  ok = !failed (ov_model_reshape (m->model, (const char **) m->input_names,
                                  shapes, m->n_inputs), "reshape")
    && !failed (ov_core_compile_model (m->core, m->model, "GPU", 2,
                                       &m->compiled[i], "CACHE_DIR",
                                       m->cache_dir), "compile")
    && !failed (ov_compiled_model_create_infer_request (m->compiled[i],
                                                        &m->request[i]),
                "infer request");
  for (size_t j = 0; j < m->n_inputs; j++)
    ov_partial_shape_free (&shapes[j]);
  return ok;
}

struct ov_embed *
ov_embed_load (const char *path, const char *cache_dir, const char **why)
{
  struct ov_embed *m = xmalloc (sizeof *m);

  memset (m, 0, sizeof *m);
  m->cache_dir = xstrdup (cache_dir);
  *why = "OpenVINO cannot start";
  if (ov_core_create (&m->core) != OK)
    goto fail;
  *why = "OpenVINO finds no GPU";
  if (!have_gpu (m->core))
    goto fail;
  *why = "OpenVINO cannot load the model";
  if (failed (ov_core_read_model (m->core, path, NULL, &m->model), path)
      || failed (ov_model_inputs_size (m->model, &m->n_inputs), path)
      || m->n_inputs > MAX_INPUTS)
    goto fail;
  for (size_t i = 0; i < m->n_inputs; i++)
    {
      ov_output_const_port_t *port;
      char *name;

      if (failed (ov_model_const_input_by_index (m->model, i, &port), path))
        goto fail;
      if (failed (ov_port_get_any_name (port, &name), path))
        {
          ov_output_const_port_free (port);
          goto fail;
        }
      m->input_names[i] = xstrdup (name);
      ov_free (name);
      ov_output_const_port_free (port);
    }
  /* Find out now whether the GPU can run it at all.  */
  *why = "OpenVINO cannot compile the model for the GPU";
  if (!compile_length (m, 0))
    goto fail;
  return m;

 fail:
  ov_embed_free (m);
  return NULL;
}

void
ov_embed_free (struct ov_embed *m)
{
  if (!m)
    return;
  for (size_t i = 0; i < N_LENGTHS; i++)
    {
      if (m->request[i])
        ov_infer_request_free (m->request[i]);
      if (m->compiled[i])
        ov_compiled_model_free (m->compiled[i]);
    }
  for (size_t i = 0; i < m->n_inputs && i < MAX_INPUTS; i++)
    free (m->input_names[i]);
  if (m->model)
    ov_model_free (m->model);
  if (m->core)
    ov_core_free (m->core);
  free (m->cache_dir);
  free (m);
}

bool
ov_embed (struct ov_embed *m, const int64_t *ids, size_t n, float **vec,
          size_t *dim)
{
  size_t which = 0, len;
  int64_t data[3][OV_EMBED_MAX];        /* Ids, mask, token types.  */
  int64_t dims[2];
  ov_tensor_t *tensors[MAX_INPUTS] = { 0 }, *out = NULL;
  ov_shape_t shape = { 0 }, out_shape = { 0 };
  float *hidden, *v;
  bool ok = false;
  double norm = 0;

  if (n == 0 || n > OV_EMBED_MAX)
    return false;
  while (lengths[which] < n)
    which++;
  if (!compile_length (m, which))
    return false;
  len = lengths[which];
  for (size_t i = 0; i < len; i++)
    {
      data[0][i] = i < n ? ids[i] : 0;
      data[1][i] = i < n;
      data[2][i] = 0;
    }
  dims[0] = 1;
  dims[1] = len;
  if (failed (ov_shape_create (2, dims, &shape), "shape"))
    return false;
  for (size_t i = 0; i < m->n_inputs; i++)
    {
      const char *name = m->input_names[i];
      int64_t *d = strcmp (name, "input_ids") == 0 ? data[0]
        : strcmp (name, "attention_mask") == 0 ? data[1] : data[2];

      if (failed (ov_tensor_create_from_host_ptr (I64, shape, d,
                                                  &tensors[i]), "tensor")
          || failed (ov_infer_request_set_tensor (m->request[which], name,
                                                  tensors[i]), name))
        goto out;
    }
  if (failed (ov_infer_request_infer (m->request[which]), "infer")
      || failed (ov_infer_request_get_output_tensor_by_index
                 (m->request[which], 0, &out), "output")
      || failed (ov_tensor_get_shape (out, &out_shape), "output shape"))
    goto out;
  ov_element_type_e type;
  if (failed (ov_tensor_get_element_type (out, &type), "output type")
      || type != F32 || out_shape.rank != 3
      || out_shape.dims[1] != (int64_t) len || out_shape.dims[2] <= 0)
    {
      error (0, 0, "OpenVINO: unexpected output type or shape");
      goto out;
    }
  if (failed (ov_tensor_data (out, (void **) &hidden), "output data"))
    goto out;

  /* Mean pooling over the real tokens, then unit length.  */
  *dim = out_shape.dims[2];
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
  if (out_shape.dims)
    ov_shape_free (&out_shape);
  if (out)
    ov_tensor_free (out);
  for (size_t i = 0; i < m->n_inputs; i++)
    if (tensors[i])
      ov_tensor_free (tensors[i]);
  ov_shape_free (&shape);
  return ok;
}
