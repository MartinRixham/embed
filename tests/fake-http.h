/* fake-http.h -- a stand-in for the Ollama HTTP API, for tests.
   Copyright (C) 2026 Martin Rixham

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.  */

#ifndef FAKE_HTTP_H
#define FAKE_HTTP_H

/* Number of dimensions of the vectors the fake model returns.  */
#define FAKE_DIM 8

/* Keywords that the fake model maps to dimensions 0..FAKE_DIM-1.  An
   input's vector has 0.05 plus the number of occurrences of each
   keyword (ignoring case) in that dimension, so that texts sharing a
   keyword are close.  */
extern const char *const fake_keywords[FAKE_DIM];

/* Accept connections on LISTEN_FD forever.  Each POST body is appended
   to LOG (if not NULL) followed by a line containing only "----".

   Requests to /api/version answer 200.  Requests to /api/embed answer
   with an embedding computed from the keywords, unless the body
   contains one of these markers:

     FAKE_500        HTTP 500 with an error message
     FAKE_TOOLONG    HTTP 400, "input length exceeds the context length"
     FAKE_MALFORMED  HTTP 200 with a truncated vector
     FAKE_EMPTY      HTTP 200 with an empty vector
     FAKE_NOVEC      HTTP 200 without an "embeddings" member
     FAKE_BADSEP     HTTP 200 with numbers separated by semicolons
     FAKE_DIM3       a 3-dimensional vector
     FAKE_ZERO       a vector of zeros
     FAKE_SPACES     the normal answer, laid out with white space
     FAKE_SLOW       the normal answer, after a 3 second delay

   /api/show and /api/pull answer as for an available model, unless
   the environment variable FAKE_MODEL is

     missing    the model is missing until pulled; pulling succeeds
     pullfail   the model is missing, and pulling ends with an error
     pull500    the model is missing, and pulling answers HTTP 500
     showfail   /api/show answers HTTP 500

   While the model is missing, /api/embed answers 404.  */
void fake_serve (int listen_fd, const char *log) __attribute__ ((noreturn));

/* Bind a listening socket to a free loopback port.  Store the port in
   *PORT and return the socket.  */
int fake_listen (int *port);

#endif /* FAKE_HTTP_H */
