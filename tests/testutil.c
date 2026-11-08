/* testutil.c -- helpers shared by the unit tests.
   Copyright (C) 2026 Martin Rixham

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.  */

#include "testutil.h"
#include "fake-http.h"

#include <errno.h>
#include <fcntl.h>
#include <ftw.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static void
die (const char *what)
{
  perror (what);
  exit (99);
}

void
fake_start (struct fake *f, const char *dir)
{
  int fd;

  f->port = 0;
  fd = fake_listen (&f->port);
  snprintf (f->url, sizeof f->url, "http://127.0.0.1:%d", f->port);
  f->pid = fork ();
  if (f->pid < 0)
    die ("fork");
  if (f->pid == 0)
    {
      /* Its own group, so that fake_stop also gets the children that
         serve each connection.  */
      setpgid (0, 0);
      prctl (PR_SET_PDEATHSIG, SIGKILL);
      fake_serve (fd, dir);
    }
  close (fd);
}

void
fake_stop (struct fake *f)
{
  if (f->pid > 0)
    {
      kill (-f->pid, SIGKILL);
      kill (f->pid, SIGKILL);
      waitpid (f->pid, NULL, 0);
      f->pid = 0;
    }
}

const char *
dead_url (void)
{
  static char url[64];
  int port = 0;
  int fd = fake_listen (&port);

  /* Nothing listens once the socket is closed.  */
  close (fd);
  snprintf (url, sizeof url, "http://127.0.0.1:%d", port);
  return url;
}

char *
tmpdir_new (void)
{
  const char *base = getenv ("TMPDIR");
  char *dir;

  if (!base || !*base)
    base = "/tmp";
  if (asprintf (&dir, "%s/embed-test-XXXXXX", base) < 0 || !mkdtemp (dir))
    die ("mkdtemp");
  return dir;
}

static int
remove_one (const char *path, const struct stat *st, int flag,
            struct FTW *ftw)
{
  (void) st;
  (void) flag;
  (void) ftw;
  remove (path);
  return 0;
}

void
tmpdir_remove (const char *dir)
{
  nftw (dir, remove_one, 16, FTW_DEPTH | FTW_PHYS);
}

char *
path_join (const char *dir, const char *name)
{
  char *p;

  if (asprintf (&p, "%s/%s", dir, name) < 0)
    die ("asprintf");
  return p;
}

void
write_file (const char *path, const char *data, size_t n)
{
  char *copy = strdup (path);
  FILE *f;

  for (char *p = copy + 1; *p; p++)
    if (*p == '/')
      {
        *p = '\0';
        mkdir (copy, 0755);
        *p = '/';
      }
  free (copy);

  if (n == (size_t) -1)
    n = strlen (data);
  f = fopen (path, "wb");
  if (!f || fwrite (data, 1, n, f) != n || fclose (f) != 0)
    die (path);
}

char *
read_file (const char *path)
{
  FILE *f = fopen (path, "rb");
  char *data = NULL;
  size_t len = 0, cap = 0, n;

  if (!f)
    return NULL;
  do
    {
      if (len + 4096 + 1 > cap)
        {
          cap = 2 * (len + 4096 + 1);
          data = realloc (data, cap);
        }
      n = fread (data + len, 1, 4096, f);
      len += n;
    }
  while (n > 0);
  fclose (f);
  data[len] = '\0';
  return data;
}

size_t
count_substr (const char *hay, const char *needle)
{
  size_t n = 0, m = strlen (needle);

  if (!hay || m == 0)
    return 0;
  for (const char *p = hay; (p = strstr (p, needle)); p += m)
    n++;
  return n;
}

bool
have_program (const char *program)
{
  char *cmd;
  int status;

  if (asprintf (&cmd, "command -v '%s' >/dev/null 2>&1", program) < 0)
    die ("asprintf");
  status = system (cmd);
  free (cmd);
  return status == 0;
}

static pid_t
spawn_main (int (*fn) (int, char **), char *const *argv, const char *out,
            const char *err)
{
  pid_t pid;

  fflush (NULL);
  pid = fork ();
  if (pid < 0)
    die ("fork");
  if (pid == 0)
    {
      int argc = 0;

      if (!freopen (out ? out : "/dev/null", "w", stdout)
          || !freopen (err ? err : "/dev/null", "w", stderr))
        _exit (98);
      while (argv[argc])
        argc++;
      exit (fn (argc, (char **) argv));
    }
  return pid;
}

static int
wait_status (pid_t pid)
{
  int status;

  while (waitpid (pid, &status, 0) < 0)
    if (errno != EINTR)
      die ("waitpid");
  if (WIFEXITED (status))
    return WEXITSTATUS (status);
  return 128 + WTERMSIG (status);
}

int
run_main (int (*fn) (int, char **), char *const *argv, const char *out,
          const char *err)
{
  return wait_status (spawn_main (fn, argv, out, err));
}

int
run_main_signal (int (*fn) (int, char **), char *const *argv,
                 const char *out, const char *err, int delay_ms, int signo)
{
  pid_t pid = spawn_main (fn, argv, out, err);
  struct timespec delay = { delay_ms / 1000, (delay_ms % 1000) * 1000000L };

  nanosleep (&delay, NULL);
  kill (pid, signo);
  return wait_status (pid);
}

bool
process_gone (pid_t pid, int ms)
{
  for (int i = 0; i <= ms / 10; i++)
    {
      char path[64], *stat;
      struct timespec delay = { 0, 10 * 1000 * 1000 };

      /* Reap it if it is our own child.  */
      waitpid (pid, NULL, WNOHANG);
      snprintf (path, sizeof path, "/proc/%d/stat", (int) pid);
      stat = read_file (path);
      if (!stat)
        return true;
      /* The state follows the parenthesized command name.  */
      char *paren = strrchr (stat, ')');
      bool zombie = paren && paren[1] == ' ' && paren[2] == 'Z';
      free (stat);
      if (zombie)
        return true;
      nanosleep (&delay, NULL);
    }
  return false;
}

double
now (void)
{
  struct timespec ts;

  clock_gettime (CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec / 1e9;
}
