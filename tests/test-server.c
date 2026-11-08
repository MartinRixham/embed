/* test-server.c -- unit tests for server.c.
   Copyright (C) 2026 Martin Rixham

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   A fake "ollama" program stands in for the real one, so these tests
   need neither Ollama nor a GPU.  */

/* Keep the startup limit short, for the test of a server that never
   becomes ready.  */
#define START_TIMEOUT 2

/* Stand in for the system service's model store, so that the tests do
   not depend on what is installed.  */
static const char *test_system_models = "/nonexistent/embed-test/models";
#define SYSTEM_MODELS test_system_models

#include "../server.c"

#include "testutil.h"

#include <check.h>
#include <stdarg.h>
#include <stdio.h>

/* A temporary home directory, with the fake ollama first on PATH.  */
struct fixture
{
  char *home;
  char *env_file;
  char *runner_file;
};

static void
fixture_init (struct fixture *fx)
{
  fx->home = tmpdir_new ();
  fx->env_file = path_join (fx->home, "env");
  fx->runner_file = path_join (fx->home, "runner");
  setenv ("HOME", fx->home, 1);
  setenv ("FAKE_OLLAMA_ENV", fx->env_file, 1);
  setenv ("FAKE_OLLAMA_RUNNER", fx->runner_file, 1);
  unsetenv ("FAKE_OLLAMA_MODE");
  unsetenv ("OLLAMA_MODELS");
  install_fake_ollama (fx->home);
}

static void
fixture_free (struct fixture *fx)
{
  tmpdir_remove (fx->home);
}

/* Return the value of NAME in the server's environment.  */
static char *
server_env (struct fixture *fx, const char *name)
{
  char *dump = read_file (fx->env_file);
  char *value;

  ck_assert_ptr_nonnull (dump);
  value = env_lookup (dump, name);
  free (dump);
  return value;
}

static pid_t
runner_pid (struct fixture *fx)
{
  char *s = NULL;

  /* The fake writes it just after starting to listen, at the latest.  */
  for (int i = 0; i < 200 && !(s = read_file (fx->runner_file)); i++)
    usleep (10000);
  ck_assert_ptr_nonnull (s);
  pid_t pid = atoi (s);
  free (s);
  ck_assert_int_gt (pid, 0);
  return pid;
}

START_TEST (test_free_port)
{
  int a = free_port (), b = free_port ();

  ck_assert_int_gt (a, 1023);
  ck_assert_int_lt (a, 65536);
  ck_assert_int_gt (b, 0);
}
END_TEST

START_TEST (test_log_path)
{
  char *p;

  setenv ("HOME", "/home/x", 1);
  p = log_path ();
  ck_assert_str_eq (p, "/home/x/.local/state/embed/ollama.log");
  free (p);
}
END_TEST

START_TEST (test_start_answers_requests)
{
  struct fixture fx;
  struct server s;
  struct embedder e;
  float *v;
  size_t dim;

  fixture_init (&fx);
  server_start (&s, true);
  ck_assert_int_gt (s.pid, 0);
  ck_assert (strncmp (s.url, "http://127.0.0.1:", 17) == 0);

  embedder_init (&e, s.url, "m");
  ck_assert (embed_text (&e, "omega", 5, &v, &dim));
  ck_assert_uint_eq (dim, 8);
  free (v);
  embedder_free (&e);

  server_stop (&s);
  ck_assert_int_eq (s.pid, 0);
  fixture_free (&fx);
}
END_TEST

START_TEST (test_server_host_matches_url)
{
  struct fixture fx;
  struct server s;
  char *host;

  fixture_init (&fx);
  server_start (&s, true);
  host = server_env (&fx, "OLLAMA_HOST");
  ck_assert_ptr_nonnull (host);
  ck_assert_str_eq (s.url + strlen ("http://"), host);
  server_stop (&s);
  fixture_free (&fx);
}
END_TEST

START_TEST (test_gpu_environment)
{
  struct fixture fx;
  struct server s;

  fixture_init (&fx);
  /* The system service hides the GPU this way; we must undo it.  */
  setenv ("GGML_VK_VISIBLE_DEVICES", "-1", 1);
  setenv ("OLLAMA_IGPU_ENABLE", "0", 1);
  server_start (&s, true);
  ck_assert_str_eq (server_env (&fx, "OLLAMA_VULKAN"), "1");
  ck_assert_str_eq (server_env (&fx, "OLLAMA_IGPU_ENABLE"), "1");
  ck_assert_ptr_null (server_env (&fx, "GGML_VK_VISIBLE_DEVICES"));
  ck_assert_str_eq (server_env (&fx, "OLLAMA_NOPRUNE"), "1");
  ck_assert_str_eq (server_env (&fx, "OLLAMA_MAX_LOADED_MODELS"), "1");
  ck_assert_str_eq (server_env (&fx, "OLLAMA_KEEP_ALIVE"), "-1");
  server_stop (&s);
  fixture_free (&fx);
}
END_TEST

START_TEST (test_cpu_environment)
{
  struct fixture fx;
  struct server s;

  fixture_init (&fx);
  setenv ("OLLAMA_VULKAN", "1", 1);
  server_start (&s, false);
  ck_assert_str_eq (server_env (&fx, "OLLAMA_VULKAN"), "0");
  ck_assert_str_eq (server_env (&fx, "OLLAMA_IGPU_ENABLE"), "0");
  ck_assert_str_eq (server_env (&fx, "GGML_VK_VISIBLE_DEVICES"), "-1");
  ck_assert_str_eq (server_env (&fx, "CUDA_VISIBLE_DEVICES"), "-1");
  ck_assert_str_eq (server_env (&fx, "HIP_VISIBLE_DEVICES"), "-1");
  ck_assert_str_eq (server_env (&fx, "ROCR_VISIBLE_DEVICES"), "-1");
  ck_assert_str_eq (server_env (&fx, "OLLAMA_NOPRUNE"), "1");
  server_stop (&s);
  fixture_free (&fx);
}
END_TEST

/* Create a model store under the fixture's home holding a manifest for
   each of the NULL-terminated MODELS, and return its directory.  */
static char *
make_models (struct fixture *fx, const char *model, ...)
{
  char *dir = path_join (fx->home, "system-models");
  va_list ap;

  mkdir (dir, 0755);
  va_start (ap, model);
  for (const char *m = model; m; m = va_arg (ap, const char *))
    {
      char *rel, *path;

      ck_assert_int_ge (asprintf (&rel, "manifests/registry.ollama.ai/%s",
                                  m), 0);
      path = path_join (dir, rel);
      write_file (path, "{}", (size_t) -1);
      free (rel);
      free (path);
    }
  va_end (ap);
  return dir;
}

START_TEST (test_has_model)
{
  struct fixture fx;
  char *dir;

  fixture_init (&fx);
  dir = make_models (&fx, "library/nomic-embed-text/latest",
                     "library/qwen3-embedding/0.6b", "someone/thing/latest",
                     NULL);
  ck_assert (has_model (dir, "nomic-embed-text"));
  ck_assert (has_model (dir, "nomic-embed-text:latest"));
  ck_assert (has_model (dir, "qwen3-embedding:0.6b"));
  ck_assert (has_model (dir, "someone/thing"));
  ck_assert (!has_model (dir, "qwen3-embedding"));
  ck_assert (!has_model (dir, "qwen3-embedding:4b"));
  ck_assert (!has_model (dir, "nomic"));
  ck_assert (!has_model (dir, "thing"));
  ck_assert (!has_model ("/nonexistent", "nomic-embed-text"));
  fixture_free (&fx);
}
END_TEST

/* Start a server and return its OLLAMA_MODELS and its log.  */
static char *
models_used (struct fixture *fx, char **log)
{
  struct server s;
  char *models, *log_file;

  server_start (&s, false);
  models = server_env (fx, "OLLAMA_MODELS");
  server_stop (&s);
  log_file = path_join (fx->home, ".local/state/embed/ollama.log");
  *log = read_file (log_file);
  ck_assert_ptr_nonnull (*log);
  free (log_file);
  return models;
}

START_TEST (test_models_system_store)
{
  struct fixture fx;
  char *log;

  fixture_init (&fx);
  test_system_models = make_models (&fx, "library/" EMBED_MODEL "/latest",
                                    NULL);
  ck_assert_str_eq (models_used (&fx, &log), test_system_models);
  ck_assert_ptr_nonnull (strstr (log, "embed: using " EMBED_MODEL " in "));
  fixture_free (&fx);
}
END_TEST

/* The system store cannot be written, so a model missing from it is
   pulled into one of our own.  */
START_TEST (test_models_own_store_when_model_missing)
{
  struct fixture fx;
  struct stat st;
  char *own, *log;

  fixture_init (&fx);
  test_system_models = make_models (&fx, "library/other/latest", NULL);
  own = path_join (fx.home, ".local/share/embed/models");
  ck_assert_str_eq (models_used (&fx, &log), own);
  ck_assert_int_eq (stat (own, &st), 0);
  ck_assert (S_ISDIR (st.st_mode));
  ck_assert_ptr_nonnull (strstr (log, "is not in "));
  fixture_free (&fx);
}
END_TEST

/* As in a container, which does not see the host's /var/lib.  */
START_TEST (test_models_own_store_without_system_store)
{
  struct fixture fx;
  char *log;

  fixture_init (&fx);
  ck_assert_str_eq (models_used (&fx, &log),
                    path_join (fx.home, ".local/share/embed/models"));
  fixture_free (&fx);
}
END_TEST

START_TEST (test_models_directory_override)
{
  struct fixture fx;
  char *log;

  fixture_init (&fx);
  test_system_models = make_models (&fx, "library/" EMBED_MODEL "/latest",
                                    NULL);
  setenv ("OLLAMA_MODELS", "/somewhere/else", 1);
  ck_assert_str_eq (models_used (&fx, &log), "/somewhere/else");
  ck_assert_ptr_nonnull (strstr (log, "OLLAMA_MODELS=/somewhere/else from "
                                 "the environment"));
  fixture_free (&fx);
}
END_TEST

/* An empty OLLAMA_MODELS is the same as an unset one; ollama would
   otherwise fall back to an empty store in ~/.ollama.  */
START_TEST (test_models_directory_empty)
{
  struct fixture fx;
  char *log;

  fixture_init (&fx);
  test_system_models = make_models (&fx, "library/" EMBED_MODEL "/latest",
                                    NULL);
  setenv ("OLLAMA_MODELS", "", 1);
  ck_assert_str_eq (models_used (&fx, &log), test_system_models);
  fixture_free (&fx);
}
END_TEST

START_TEST (test_log_written)
{
  struct fixture fx;
  struct server s;
  char *log_file, *log;

  fixture_init (&fx);
  server_start (&s, true);
  server_stop (&s);
  log_file = path_join (fx.home, ".local/state/embed/ollama.log");
  log = read_file (log_file);
  ck_assert_ptr_nonnull (log);
  ck_assert_ptr_nonnull (strstr (log, "fake ollama starting"));

  /* The log is appended to, not replaced.  */
  server_start (&s, true);
  server_stop (&s);
  log = read_file (log_file);
  ck_assert_uint_eq (count_substr (log, "fake ollama starting"), 2);
  fixture_free (&fx);
}
END_TEST

START_TEST (test_own_process_group)
{
  struct fixture fx;
  struct server s;

  fixture_init (&fx);
  server_start (&s, true);
  /* So that Ctrl-C in the terminal reaches only us.  */
  ck_assert_int_eq (getpgid (s.pid), s.pid);
  ck_assert_int_ne (getpgid (s.pid), getpgrp ());
  server_stop (&s);
  fixture_free (&fx);
}
END_TEST

START_TEST (test_stop_kills_runners)
{
  struct fixture fx;
  struct server s;
  pid_t server, runner;

  fixture_init (&fx);
  server_start (&s, true);
  server = s.pid;
  runner = runner_pid (&fx);
  ck_assert (!process_gone (runner, 0));
  server_stop (&s);
  ck_assert (process_gone (server, 1000));
  ck_assert (process_gone (runner, 2000));
  fixture_free (&fx);
}
END_TEST

START_TEST (test_stop_kills_stubborn_server)
{
  struct fixture fx;
  struct server s;
  pid_t server;
  double start;

  fixture_init (&fx);
  setenv ("FAKE_OLLAMA_MODE", "ignore-term", 1);
  server_start (&s, true);
  server = s.pid;
  start = now ();
  server_stop (&s);
  ck_assert (process_gone (server, 1000));
  /* It waits for a graceful exit for about five seconds first.  */
  ck_assert_double_lt (now () - start, 7);
  fixture_free (&fx);
}
END_TEST

START_TEST (test_stop_twice_is_harmless)
{
  struct fixture fx;
  struct server s;
  struct server never = { 0 };

  fixture_init (&fx);
  server_start (&s, true);
  server_stop (&s);
  server_stop (&s);
  server_stop (&never);
  fixture_free (&fx);
}
END_TEST

START_TEST (test_dies_with_parent)
{
  struct fixture fx;
  int fds[2];
  pid_t child, server;

  fixture_init (&fx);
  ck_assert_int_eq (pipe (fds), 0);
  child = fork ();
  if (child == 0)
    {
      struct server s;

      server_start (&s, true);
      write (fds[1], &s.pid, sizeof s.pid);
      /* Die without stopping the server.  */
      _exit (0);
    }
  ck_assert_int_eq (read (fds[0], &server, sizeof server),
                    (ssize_t) sizeof server);
  waitpid (child, NULL, 0);
  ck_assert (process_gone (server, 3000));
  fixture_free (&fx);
}
END_TEST

START_TEST (test_server_exits_at_startup)
{
  struct fixture fx;
  struct server s;

  fixture_init (&fx);
  setenv ("FAKE_OLLAMA_MODE", "exit", 1);
  server_start (&s, true);
}
END_TEST

START_TEST (test_server_never_ready)
{
  struct fixture fx;
  struct server s;

  fixture_init (&fx);
  setenv ("FAKE_OLLAMA_MODE", "hang", 1);
  server_start (&s, true);
}
END_TEST

START_TEST (test_server_never_ready_is_cleaned_up)
{
  struct fixture fx;
  char *err = NULL;
  pid_t child, runner;
  int status;
  double start;

  fixture_init (&fx);
  setenv ("FAKE_OLLAMA_MODE", "hang", 1);
  start = now ();
  child = fork ();
  if (child == 0)
    {
      struct server s;

      ck_assert_ptr_nonnull (freopen (path_join (fx.home, "stderr"), "w",
                                      stderr));
      server_start (&s, true);
      _exit (0);
    }
  waitpid (child, &status, 0);
  ck_assert (WIFEXITED (status));
  ck_assert_int_eq (WEXITSTATUS (status), EXIT_FAILURE);
  ck_assert_double_lt (now () - start, 10);
  err = read_file (path_join (fx.home, "stderr"));
  ck_assert_ptr_nonnull (strstr (err, "did not start within 2 seconds"));
  runner = runner_pid (&fx);
  ck_assert (process_gone (runner, 3000));
  fixture_free (&fx);
}
END_TEST

START_TEST (test_ollama_missing)
{
  struct fixture fx;
  pid_t child;
  int status;
  char *log;

  fixture_init (&fx);
  setenv ("PATH", "/nonexistent", 1);
  child = fork ();
  if (child == 0)
    {
      struct server s;

      ck_assert_ptr_nonnull (freopen ("/dev/null", "w", stderr));
      server_start (&s, true);
      _exit (0);
    }
  waitpid (child, &status, 0);
  ck_assert (WIFEXITED (status));
  ck_assert_int_eq (WEXITSTATUS (status), EXIT_FAILURE);
  log = read_file (path_join (fx.home, ".local/state/embed/ollama.log"));
  ck_assert_msg (log && strstr (log, "cannot run ollama"), "log: %s", log ? log : "(none)");
  fixture_free (&fx);
}
END_TEST

START_TEST (test_two_servers_at_once)
{
  struct fixture fx;
  struct server a, b;

  fixture_init (&fx);
  server_start (&a, true);
  server_start (&b, false);
  ck_assert_str_ne (a.url, b.url);
  server_stop (&a);
  server_stop (&b);
  fixture_free (&fx);
}
END_TEST

static Suite *
server_suite (void)
{
  Suite *s = suite_create ("server");
  TCase *tc = tcase_create ("server");

  tcase_set_timeout (tc, 30);
  tcase_add_test (tc, test_free_port);
  tcase_add_test (tc, test_log_path);
  tcase_add_test (tc, test_start_answers_requests);
  tcase_add_test (tc, test_server_host_matches_url);
  tcase_add_test (tc, test_gpu_environment);
  tcase_add_test (tc, test_cpu_environment);
  tcase_add_test (tc, test_has_model);
  tcase_add_test (tc, test_models_system_store);
  tcase_add_test (tc, test_models_own_store_when_model_missing);
  tcase_add_test (tc, test_models_own_store_without_system_store);
  tcase_add_test (tc, test_models_directory_override);
  tcase_add_test (tc, test_models_directory_empty);
  tcase_add_test (tc, test_log_written);
  tcase_add_test (tc, test_own_process_group);
  tcase_add_test (tc, test_stop_kills_runners);
  tcase_add_test (tc, test_stop_kills_stubborn_server);
  tcase_add_test (tc, test_stop_twice_is_harmless);
  tcase_add_test (tc, test_dies_with_parent);
  tcase_add_exit_test (tc, test_server_exits_at_startup, EXIT_FAILURE);
  tcase_add_exit_test (tc, test_server_never_ready, EXIT_FAILURE);
  tcase_add_test (tc, test_server_never_ready_is_cleaned_up);
  tcase_add_test (tc, test_ollama_missing);
  tcase_add_test (tc, test_two_servers_at_once);
  suite_add_tcase (s, tc);
  return s;
}

int
main (void)
{
  SRunner *sr = srunner_create (server_suite ());
  int failed;

  srunner_run_all (sr, CK_NORMAL);
  failed = srunner_ntests_failed (sr);
  srunner_free (sr);
  return failed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
