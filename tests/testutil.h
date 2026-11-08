/* testutil.h -- helpers shared by the unit tests.
   Copyright (C) 2026 Martin Rixham

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.  */

#ifndef TESTUTIL_H
#define TESTUTIL_H

#include <stdbool.h>
#include <stddef.h>
#include <sys/types.h>

/* A fake HTTP server running in a child process.  */
struct fake
{
  pid_t pid;
  int port;
  char url[64];
};

/* Start a fake file server (see fake-http.h) serving DIR.  It dies
   with the calling process.  */
void fake_start (struct fake *f, const char *dir);
void fake_stop (struct fake *f);

/* Return a URL on which nothing is listening.  */
const char *dead_url (void);

/* Create a fresh temporary directory and return its name.  */
char *tmpdir_new (void);

/* Remove DIR and everything below it.  */
void tmpdir_remove (const char *dir);

/* Return DIR/NAME, freshly allocated.  */
char *path_join (const char *dir, const char *name);

/* Create the file PATH, and any missing parent directories, holding
   the N bytes at DATA (or strlen (DATA) if N is (size_t) -1).  */
void write_file (const char *path, const char *data, size_t n);

/* Return the contents of PATH, freshly allocated and NUL terminated,
   or NULL if it cannot be read.  */
char *read_file (const char *path);

/* Return the number of non-overlapping occurrences of NEEDLE in HAY.  */
size_t count_substr (const char *hay, const char *needle);

/* Return true if PROGRAM can be found on PATH.  */
bool have_program (const char *program);

/* Run FN (ARGV) in a child process with standard output and standard
   error sent to the files OUT and ERR (either may be NULL to discard).
   Return its exit status, or 128 plus the signal that killed it.  */
int run_main (int (*fn) (int, char **), char *const *argv, const char *out,
              const char *err);

/* Like run_main, but send SIGNO to the child after DELAY_MS
   milliseconds.  */
int run_main_signal (int (*fn) (int, char **), char *const *argv,
                     const char *out, const char *err, int delay_ms,
                     int signo);


/* Return true if process PID has gone, waiting up to MS milliseconds.
   Zombies count as gone.  */
bool process_gone (pid_t pid, int ms);

/* Return the current time in seconds.  */
double now (void);

#endif /* TESTUTIL_H */
