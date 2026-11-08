/* fake-http.c -- a stand-in for the Ollama HTTP API, for tests.
   Copyright (C) 2026 Martin Rixham

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.  */

#include "fake-http.h"

#include <arpa/inet.h>
#include <signal.h>
#include <stdbool.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <unistd.h>

const char *const fake_keywords[FAKE_DIM] = {
  "sourdough", "invoice", "sonata", "quicksort",
  "sunset", "secret", "alpha", "omega"
};

int
fake_listen (int *port)
{
  struct sockaddr_in addr = { .sin_family = AF_INET };
  socklen_t len = sizeof addr;
  int one = 1;
  int fd = socket (AF_INET, SOCK_STREAM, 0);

  addr.sin_addr.s_addr = htonl (INADDR_LOOPBACK);
  addr.sin_port = htons (*port);
  setsockopt (fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
  if (fd < 0 || bind (fd, (struct sockaddr *) &addr, sizeof addr) != 0
      || listen (fd, 16) != 0
      || getsockname (fd, (struct sockaddr *) &addr, &len) != 0)
    {
      perror ("fake_listen");
      exit (99);
    }
  *port = ntohs (addr.sin_port);
  return fd;
}

static void
write_all (int fd, const char *s, size_t n)
{
  while (n > 0)
    {
      ssize_t w = write (fd, s, n);
      if (w <= 0)
        return;
      s += w;
      n -= w;
    }
}

static void
respond (int fd, int status, const char *body)
{
  char head[256];

  snprintf (head, sizeof head,
            "HTTP/1.1 %d X\r\nContent-Type: application/json\r\n"
            "Content-Length: %zu\r\nConnection: close\r\n\r\n",
            status, strlen (body));
  write_all (fd, head, strlen (head));
  write_all (fd, body, strlen (body));
}

static size_t
count_ci (const char *hay, const char *needle)
{
  size_t n = 0, m = strlen (needle);

  for (const char *p = hay; *p; p++)
    if (strncasecmp (p, needle, m) == 0)
      n++;
  return n;
}

/* Whether the model has been pulled, for FAKE_MODEL=missing.  */
static bool pulled;

/* Return true if FAKE_MODEL is STATE.  */
static bool
model_state (const char *state)
{
  const char *s = getenv ("FAKE_MODEL");
  return s && strcmp (s, state) == 0;
}

static bool
model_missing (void)
{
  return !pulled && (model_state ("missing") || model_state ("pullfail")
                     || model_state ("pull500"));
}

/* Answer an /api/show request.  */
static void
show_response (int fd)
{
  if (model_state ("showfail"))
    respond (fd, 500, "{\"error\":\"show broke\"}");
  else if (model_missing ())
    respond (fd, 404, "{\"error\":\"model 'm' not found\"}");
  else
    respond (fd, 200, "{\"modelfile\":\"FROM m\"}");
}

/* Answer an /api/pull request, streaming progress like Ollama.  */
static void
pull_response (int fd)
{
  if (model_state ("pull500"))
    return respond (fd, 500, "{\"error\":\"registry unreachable\"}");
  if (model_state ("pullfail"))
    return respond (fd, 200, "{\"status\":\"pulling manifest\"}\n"
                    "{\"error\":\"pull model manifest: file does not "
                    "exist\"}\n");
  pulled = true;
  respond (fd, 200, "{\"status\":\"pulling manifest\"}\n"
           "{\"status\":\"pulling 970aa74c0a90\",\"total\":274290656,"
           "\"completed\":274290656}\n"
           "{\"status\":\"verifying sha256 digest\"}\n"
           "{\"status\":\"success\"}\n");
}

/* Answer an /api/embed request whose body is BODY.  */
static void
embed_response (int fd, const char *body)
{
  char out[1024];
  size_t dim = strstr (body, "FAKE_DIM3") ? 3 : FAKE_DIM;
  bool spaces = strstr (body, "FAKE_SPACES") != NULL;
  size_t len;

  if (model_missing ())
    return respond (fd, 404, "{\"error\":\"model \\\"m\\\" not found, "
                    "try pulling it first\"}");
  if (strstr (body, "FAKE_500"))
    return respond (fd, 500, "{\"error\":\"boom\"}");
  if (strstr (body, "FAKE_TOOLONG"))
    return respond (fd, 400, "{\"error\":\"the input length exceeds the "
                    "context length\"}");
  if (strstr (body, "FAKE_MALFORMED"))
    return respond (fd, 200, "{\"model\":\"m\",\"embeddings\":[[0.1,0.2,");
  if (strstr (body, "FAKE_EMPTY"))
    return respond (fd, 200, "{\"model\":\"m\",\"embeddings\":[[]]}");
  if (strstr (body, "FAKE_BADSEP"))
    return respond (fd, 200, "{\"embeddings\":[[0.1;0.2;0.3]]}");
  if (strstr (body, "FAKE_NOVEC"))
    return respond (fd, 200, "{\"model\":\"m\"}");
  if (strstr (body, "FAKE_SLOW"))
    sleep (3);

  len = snprintf (out, sizeof out, spaces
                  ? "{\n  \"model\": \"m\",\n  \"embeddings\": [ [ "
                  : "{\"model\":\"m\",\"embeddings\":[[");
  for (size_t i = 0; i < dim; i++)
    {
      double v = strstr (body, "FAKE_ZERO") ? 0
        : 0.05 + count_ci (body, fake_keywords[i]);
      len += snprintf (out + len, sizeof out - len, "%s%g",
                       i == 0 ? "" : spaces ? " , " : ",", v);
    }
  snprintf (out + len, sizeof out - len, spaces ? " ] ]\n}\n" : "]]}");
  respond (fd, 200, out);
}

static void
handle (int fd, const char *log)
{
  char *req = malloc (1);
  size_t len = 0, cap = 1, need = 0;
  char *body = NULL;
  bool continued = false;

  for (;;)
    {
      char chunk[65536];
      ssize_t n;

      if (body && len - (body - req) >= need)
        break;
      n = read (fd, chunk, sizeof chunk);
      if (n <= 0)
        break;
      if (len + n + 1 > cap)
        {
          size_t off = body ? (size_t) (body - req) : 0;
          cap = 2 * (len + n + 1);
          req = realloc (req, cap);
          if (body)
            body = req + off;
        }
      memcpy (req + len, chunk, n);
      len += n;
      req[len] = '\0';

      if (!body && (body = strstr (req, "\r\n\r\n")))
        {
          const char *cl = strcasestr (req, "Content-Length:");
          need = cl ? strtoul (cl + strlen ("Content-Length:"), NULL, 10)
            : 0;
          body += 4;
          /* libcurl waits for this before sending a large body.  */
          if (!continued && strcasestr (req, "Expect: 100-continue"))
            {
              const char *cont = "HTTP/1.1 100 Continue\r\n\r\n";
              write_all (fd, cont, strlen (cont));
              continued = true;
            }
        }
    }

  if (!body)
    ;
  else if (strncmp (req, "POST /api/embed", 15) == 0)
    {
      if (log)
        {
          FILE *f = fopen (log, "a");
          if (f)
            {
              fprintf (f, "%s\n----\n", body);
              fclose (f);
            }
        }
      embed_response (fd, body);
    }
  else if (strncmp (req, "POST /api/show ", 15) == 0)
    show_response (fd);
  else if (strncmp (req, "POST /api/pull ", 15) == 0)
    pull_response (fd);
  else if (strstr (req, " /api/version "))
    respond (fd, 200, "{\"version\":\"fake\"}");
  else
    respond (fd, 404, "{\"error\":\"not found\"}");
  free (req);
}

void
fake_serve (int listen_fd, const char *log)
{
  /* A client that gave up must not kill the server.  */
  signal (SIGPIPE, SIG_IGN);
  for (;;)
    {
      int fd = accept (listen_fd, NULL, NULL);
      if (fd < 0)
        continue;
      handle (fd, log);
      close (fd);
    }
}
