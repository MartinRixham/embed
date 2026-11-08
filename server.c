/* server.c -- run a private Ollama server on the GPU.
   Copyright (C) 2026 Martin Rixham

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   Rather than relying on a system-wide Ollama service, which may be
   absent or configured to run on the CPU, each program starts its own
   "ollama serve" on a free loopback port, using either the GPU
   through Vulkan or only the CPU, and stops it on exit.  The server runs in its own process
   group, so that Ctrl-C reaches only the parent, which can then finish
   cleanly; and it is told to exit if the parent dies.  */

#include "server.h"
#include "common.h"

#include <arpa/inet.h>
#include <errno.h>
#include <error.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

/* Where Fedora's ollama package keeps the models it pulls.  */
#define SYSTEM_MODELS "/var/lib/ollama/.ollama/models"

/* How long to wait for the server to accept requests, in seconds.  */
#ifndef START_TIMEOUT
# define START_TIMEOUT 30
#endif

/* Return a loopback TCP port that is currently free.  */
static int
free_port (void)
{
  struct sockaddr_in addr = { .sin_family = AF_INET };
  socklen_t len = sizeof addr;
  int fd = socket (AF_INET, SOCK_STREAM, 0);
  int port;

  addr.sin_addr.s_addr = htonl (INADDR_LOOPBACK);
  if (fd < 0
      || bind (fd, (struct sockaddr *) &addr, sizeof addr) != 0
      || getsockname (fd, (struct sockaddr *) &addr, &len) != 0)
    error (EXIT_FAILURE, errno, "cannot find a free port");
  port = ntohs (addr.sin_port);
  close (fd);
  return port;
}

static char *
log_path (void)
{
  const char *home = getenv ("HOME");
  struct buf b;

  buf_init (&b);
  buf_puts (&b, home ? home : "");
  buf_puts (&b, "/.local/state/embed/ollama.log");
  return b.data;
}

/* Set up the environment of the server process and run it.  Only
   called in the child.  */
static void
exec_server (const char *host, const char *log, pid_t parent, bool use_gpu)
{
  struct stat st;
  int fd;

  setpgid (0, 0);
  prctl (PR_SET_PDEATHSIG, SIGTERM);
  if (getppid () != parent)
    _exit (1);

  fd = open ("/dev/null", O_RDONLY);
  dup2 (fd, STDIN_FILENO);
  fd = open (log, O_WRONLY | O_CREAT | O_APPEND, 0644);
  if (fd >= 0)
    {
      dup2 (fd, STDOUT_FILENO);
      dup2 (fd, STDERR_FILENO);
    }

  setenv ("OLLAMA_HOST", host, 1);
  if (use_gpu)
    {
      /* Use the GPU, including integrated ones, through Vulkan.  */
      setenv ("OLLAMA_VULKAN", "1", 1);
      setenv ("OLLAMA_IGPU_ENABLE", "1", 1);
      unsetenv ("GGML_VK_VISIBLE_DEVICES");
    }
  else
    {
      /* Hide every GPU, which also skips their slow initialization.  */
      setenv ("OLLAMA_VULKAN", "0", 1);
      setenv ("OLLAMA_IGPU_ENABLE", "0", 1);
      setenv ("GGML_VK_VISIBLE_DEVICES", "-1", 1);
      setenv ("CUDA_VISIBLE_DEVICES", "-1", 1);
      setenv ("HIP_VISIBLE_DEVICES", "-1", 1);
      setenv ("ROCR_VISIBLE_DEVICES", "-1", 1);
    }
  /* Share the models of the system service when there is one; never
     delete anything from that store.  */
  if (!getenv ("OLLAMA_MODELS") && stat (SYSTEM_MODELS, &st) == 0)
    setenv ("OLLAMA_MODELS", SYSTEM_MODELS, 1);
  setenv ("OLLAMA_NOPRUNE", "1", 1);
  setenv ("OLLAMA_MAX_LOADED_MODELS", "1", 1);
  setenv ("OLLAMA_NUM_PARALLEL", "1", 1);
  setenv ("OLLAMA_KEEP_ALIVE", "-1", 1);

  execlp ("ollama", "ollama", "serve", (char *) NULL);
  /* Not through stdio, whose buffer _exit would discard.  */
  dprintf (STDERR_FILENO, "cannot run ollama: %s\n", strerror (errno));
  _exit (127);
}

/* Return true if the server at URL answers.  */
static bool
server_ready (CURL *curl, const char *url)
{
  struct buf b;
  long status = 0;

  buf_init (&b);
  buf_puts (&b, url);
  buf_puts (&b, "/api/version");
  curl_easy_reset (curl);
  curl_easy_setopt (curl, CURLOPT_URL, b.data);
  curl_easy_setopt (curl, CURLOPT_NOBODY, 1L);
  curl_easy_setopt (curl, CURLOPT_TIMEOUT, 1L);
  curl_easy_setopt (curl, CURLOPT_NOSIGNAL, 1L);
  if (curl_easy_perform (curl) == CURLE_OK)
    curl_easy_getinfo (curl, CURLINFO_RESPONSE_CODE, &status);
  buf_free (&b);
  return status == 200;
}

void
server_start (struct server *s, bool use_gpu)
{
  char *log = log_path ();
  char host[32];
  pid_t parent = getpid ();
  CURL *curl;
  int status;

  snprintf (host, sizeof host, "127.0.0.1:%d", free_port ());
  snprintf (s->url, sizeof s->url, "http://%s", host);
  make_parent_dirs (log);

  s->pid = fork ();
  if (s->pid < 0)
    error (EXIT_FAILURE, errno, "fork");
  if (s->pid == 0)
    exec_server (host, log, parent, use_gpu);
  /* Also set the group here, so that server_stop can signal it even if
     the child has not got that far yet.  */
  setpgid (s->pid, s->pid);

  curl = curl_easy_init ();
  if (!curl)
    error (EXIT_FAILURE, 0, "cannot initialize libcurl");
  for (int i = 0; i < START_TIMEOUT * 20; i++)
    {
      struct timespec delay = { 0, 50 * 1000 * 1000 };

      if (server_ready (curl, s->url))
        {
          curl_easy_cleanup (curl);
          free (log);
          return;
        }
      if (waitpid (s->pid, &status, WNOHANG) == s->pid)
        error (EXIT_FAILURE, 0, "ollama exited at startup; see %s", log);
      nanosleep (&delay, NULL);
    }
  server_stop (s);
  error (EXIT_FAILURE, 0, "ollama did not start within %d seconds; see %s",
         START_TIMEOUT, log);
}

void
server_stop (struct server *s)
{
  if (s->pid <= 0)
    return;

  kill (-s->pid, SIGTERM);
  for (int i = 0; i < 100; i++)
    {
      struct timespec delay = { 0, 50 * 1000 * 1000 };

      if (waitpid (s->pid, NULL, WNOHANG) != 0)
        break;
      nanosleep (&delay, NULL);
    }
  /* Kill whatever is left in the group, such as stuck model runners.  */
  kill (-s->pid, SIGKILL);
  waitpid (s->pid, NULL, WNOHANG);
  s->pid = 0;
}
