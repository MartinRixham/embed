/* fake-http.h -- a small file server, for testing downloads.
   Copyright (C) 2026 Martin Rixham

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.  */

#ifndef FAKE_HTTP_H
#define FAKE_HTTP_H

/* Accept connections on LISTEN_FD forever, answering GET /NAME with
   the contents of DIR/NAME, or 404 if there is no such file.  Paths
   starting with these prefixes do something else with the file named
   by the rest of the path:

     /redirect/   answer 302, redirecting to /NAME
     /truncated/  promise the whole file but send only half of it
     /stall/      send half of the file, then nothing for 10 minutes

   Each request line is appended to DIR/requests.log.  */
void fake_serve (int listen_fd, const char *dir) __attribute__ ((noreturn));

/* Bind a listening socket to a free loopback port.  Store the port in
   *PORT and return the socket.  */
int fake_listen (int *port);

#endif /* FAKE_HTTP_H */
