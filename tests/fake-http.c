/* fake-http.c -- a small file server, for testing downloads.
   Copyright (C) 2026 Martin Rixham

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.  */

#include "fake-http.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

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
respond (int fd, const char *status, const char *extra, const char *body,
         size_t len, size_t promised)
{
  char head[512];

  snprintf (head, sizeof head, "HTTP/1.1 %s\r\n%sContent-Length: %zu\r\n"
            "Connection: close\r\n\r\n", status, extra, promised);
  write_all (fd, head, strlen (head));
  write_all (fd, body, len);
}

/* Return the contents of PATH, and its length in *LEN, or NULL.  */
static char *
slurp (const char *path, size_t *len)
{
  FILE *f = fopen (path, "rb");
  char *data;
  long n;

  if (!f)
    return NULL;
  fseek (f, 0, SEEK_END);
  n = ftell (f);
  rewind (f);
  data = malloc (n + 1);
  *len = fread (data, 1, n, f);
  fclose (f);
  return data;
}

static void
handle (int fd, const char *dir)
{
  char req[4096], path[8192], *name, *end;
  size_t got = 0, len;
  char *data;
  FILE *log;

  /* Read the request head; the client sends no body.  */
  while (got < sizeof req - 1 && !strstr (req, "\r\n\r\n"))
    {
      ssize_t n = read (fd, req + got, sizeof req - 1 - got);
      if (n <= 0)
        return;
      got += n;
      req[got] = '\0';
    }
  if (strncmp (req, "GET /", 5) != 0 || !(end = strchr (req + 4, ' ')))
    return respond (fd, "400 Bad Request", "", "", 0, 0);
  *end = '\0';
  name = req + 5;

  snprintf (path, sizeof path, "%s/requests.log", dir);
  if ((log = fopen (path, "a")))
    {
      fprintf (log, "GET /%s\n", name);
      fclose (log);
    }

  if (strncmp (name, "redirect/", 9) == 0)
    {
      char location[4200];
      snprintf (location, sizeof location, "Location: /%s\r\n", name + 9);
      return respond (fd, "302 Found", location, "", 0, 0);
    }
  bool truncated = strncmp (name, "truncated/", 10) == 0;
  bool stall = strncmp (name, "stall/", 6) == 0;
  if (truncated)
    name += 10;
  if (stall)
    name += 6;
  snprintf (path, sizeof path, "%s/%s", dir, name);
  if (!(data = slurp (path, &len)))
    return respond (fd, "404 Not Found", "", "", 0, 0);
  if (truncated || stall)
    {
      respond (fd, "200 OK", "", data, len / 2, len);
      if (stall)
        sleep (600);
    }
  else
    respond (fd, "200 OK", "", data, len, len);
  free (data);
}

void
fake_serve (int listen_fd, const char *dir)
{
  /* A client that gave up must not kill the server.  */
  signal (SIGPIPE, SIG_IGN);
  for (;;)
    {
      int fd = accept (listen_fd, NULL, NULL);
      if (fd < 0)
        continue;
      /* Serve each client in its own process, so that a stalled one
         does not hold up the rest.  */
      if (fork () == 0)
        {
          prctl (PR_SET_PDEATHSIG, SIGKILL);
          close (listen_fd);
          handle (fd, dir);
          close (fd);
          _exit (0);
        }
      close (fd);
      while (waitpid (-1, NULL, WNOHANG) > 0)
        continue;
    }
}
