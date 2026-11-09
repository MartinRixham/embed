/* fetch.h -- download a file and check it.
   Copyright (C) 2026 Martin Rixham

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.  */

#ifndef EMBED_FETCH_H
#define EMBED_FETCH_H

#include <stdbool.h>

/* Download URL to PATH, unless PATH already exists, creating its
   directory if need be.  The download goes to PATH.part first, and is
   renamed to PATH only if its SHA-256 hash is SHA256 (in lowercase
   hexadecimal).  Report progress on standard error if it is a
   terminal, and problems in any case; return false on failure.  */
bool fetch_file (const char *url, const char *path, const char *sha256);

#endif /* EMBED_FETCH_H */
