/* server.h -- run a private Ollama server on the GPU.
   Copyright (C) 2026 Martin Rixham

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.  */

#ifndef EMBED_SERVER_H
#define EMBED_SERVER_H

#include <stdbool.h>
#include <sys/types.h>

/* A private "ollama serve" process, independent of any system-wide
   Ollama service.  */
struct server
{
  pid_t pid;
  char url[64];                 /* Base URL, such as http://127.0.0.1:N.  */
};

/* Start a private server listening on a free loopback port, with the
   Vulkan GPU backend enabled.  Its output goes to the log file
   ~/.local/state/embed/ollama.log.  Exit with a diagnostic if it does
   not become ready.  */
void server_start (struct server *s);

/* Stop S and the model runners it started.  */
void server_stop (struct server *s);

#endif /* EMBED_SERVER_H */
