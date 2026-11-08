/* fake-ollama -- a stand-in for "ollama serve", for tests.
   Copyright (C) 2026 Martin Rixham

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   Installed as "ollama" in a directory on PATH, it serves the fake API
   of fake-http.c on the address in OLLAMA_HOST.  These environment
   variables control it:

     FAKE_OLLAMA_ENV     file to receive a copy of its environment
     FAKE_OLLAMA_LOG     file to receive the bodies of embed requests
     FAKE_OLLAMA_RUNNER  file to receive the pid of a child process,
                         standing in for a model runner
     FAKE_OLLAMA_MODE    "exit" to exit at once with status 3, "hang"
                         to never listen, "ignore-term" to ignore
                         SIGTERM  */

#include "fake-http.h"

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

extern char **environ;

int
main (int argc, char **argv)
{
  const char *mode = getenv ("FAKE_OLLAMA_MODE");
  const char *env_file = getenv ("FAKE_OLLAMA_ENV");
  const char *runner_file = getenv ("FAKE_OLLAMA_RUNNER");
  const char *host = getenv ("OLLAMA_HOST");
  const char *colon;
  int port, fd;
  pid_t runner;

  if (argc != 2 || strcmp (argv[1], "serve") != 0)
    {
      fprintf (stderr, "usage: ollama serve\n");
      return 2;
    }
  printf ("fake ollama starting\n");
  fflush (stdout);

  if (env_file)
    {
      FILE *f = fopen (env_file, "w");
      for (char **e = environ; f && *e; e++)
        fprintf (f, "%s\n", *e);
      if (f)
        fclose (f);
    }

  if (mode && strcmp (mode, "exit") == 0)
    return 3;
  if (mode && strcmp (mode, "ignore-term") == 0)
    signal (SIGTERM, SIG_IGN);

  runner = fork ();
  if (runner == 0)
    {
      signal (SIGTERM, SIG_DFL);
      for (;;)
        pause ();
    }
  if (runner_file)
    {
      FILE *f = fopen (runner_file, "w");
      if (f)
        {
          fprintf (f, "%d\n", (int) runner);
          fclose (f);
        }
    }

  if (mode && strcmp (mode, "hang") == 0)
    for (;;)
      pause ();

  colon = host ? strrchr (host, ':') : NULL;
  if (!colon)
    {
      fprintf (stderr, "fake ollama: bad OLLAMA_HOST\n");
      return 4;
    }
  port = atoi (colon + 1);
  fd = fake_listen (&port);
  fake_serve (fd, getenv ("FAKE_OLLAMA_LOG"));
}
