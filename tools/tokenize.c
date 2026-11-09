/* tokenize.c -- print the token ids of texts, for checking tokenizer.c.
   Copyright (C) 2026 Martin Rixham

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   Usage: tokenize VOCAB < TEXTS

   Reads NUL terminated texts and writes the ids of each, without any
   length limit, on a line of its own.  */

#include "../src/common.h"
#include "../src/tokenizer.h"

#include <errno.h>
#include <error.h>
#include <stdio.h>
#include <stdlib.h>

int
main (int argc, char **argv)
{
  struct tokenizer t;
  char *text = NULL;
  size_t size = 0, max = 1;
  int64_t *ids = NULL;

  if (argc != 2)
    error (EXIT_FAILURE, 0, "usage: tokenize VOCAB < TEXTS");
  if (!tokenizer_load (&t, argv[1]))
    error (EXIT_FAILURE, errno, "%s", argv[1]);
  while (getdelim (&text, &size, '\0', stdin) > 0)
    {
      size_t n;

      /* A text has at most one token per byte, plus [CLS] and [SEP].  */
      if (size + 2 > max)
        ids = xrealloc (ids, (max = size + 2) * sizeof *ids);
      n = tokenize (&t, text, ids, max);
      for (size_t i = 0; i < n; i++)
        printf (i ? " %ld" : "%ld", (long) ids[i]);
      putchar ('\n');
    }
  return 0;
}
