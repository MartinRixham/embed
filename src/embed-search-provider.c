/* embed-search-provider -- offer embed-query's matches to GNOME Shell.
   Copyright (C) 2026 Martin Rixham

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   A D-Bus service implementing org.gnome.Shell.SearchProvider2, so
   that what is typed into the Activities overview is also looked up in
   the store, and the best matching files are listed with the other
   results.  D-Bus starts it on the first search; it keeps the text
   model open between searches, since opening it takes most of the
   time of a query, and exits when it has been idle for a while.

   The Shell asks again as each key is typed, so searches are run one
   at a time on a worker thread, and any that a newer one has made
   stale by the time it is reached is answered with no results without
   being run.  The store is read afresh for each search, so that what
   embed-index adds is found at once.

   Activating a result opens the file; launching the search from the
   provider's heading shows the best match in the file manager.  */

#include "common.h"
#include "model.h"
#include "search.h"

#include <errno.h>
#include <error.h>
#include <getopt.h>
#include <gio/gio.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BUS_NAME "embed.SearchProvider"
#define OBJECT_PATH "/embed/SearchProvider"

static const char introspection_xml[] =
  "<node>"
  "  <interface name='org.gnome.Shell.SearchProvider2'>"
  "    <method name='GetInitialResultSet'>"
  "      <arg type='as' name='terms' direction='in'/>"
  "      <arg type='as' name='results' direction='out'/>"
  "    </method>"
  "    <method name='GetSubsearchResultSet'>"
  "      <arg type='as' name='previous_results' direction='in'/>"
  "      <arg type='as' name='terms' direction='in'/>"
  "      <arg type='as' name='results' direction='out'/>"
  "    </method>"
  "    <method name='GetResultMetas'>"
  "      <arg type='as' name='identifiers' direction='in'/>"
  "      <arg type='aa{sv}' name='metas' direction='out'/>"
  "    </method>"
  "    <method name='ActivateResult'>"
  "      <arg type='s' name='identifier' direction='in'/>"
  "      <arg type='as' name='terms' direction='in'/>"
  "      <arg type='u' name='timestamp' direction='in'/>"
  "    </method>"
  "    <method name='LaunchSearch'>"
  "      <arg type='as' name='terms' direction='in'/>"
  "      <arg type='u' name='timestamp' direction='in'/>"
  "    </method>"
  "  </interface>"
  "</node>";

/* Settings from the command line.  */
static size_t count = 5;
static double min_score = 0.6;
static unsigned idle_seconds = 300;
static size_t min_query = 3;

static GMainLoop *loop;
static GThreadPool *pool;
static guint idle_source;

/* Searches queued or running, and the number of the newest.  Only the
   main thread touches PENDING; NEWEST is read by the worker too.  */
static unsigned pending;
static gint newest;

/* The model, opened by the worker on its first search.  */
static struct model *model;
static bool model_failed;

/* A search for the worker to run.  */
struct job
{
  GDBusMethodInvocation *invocation;
  char *text;
  gint number;
  GPtrArray *results;           /* Filled in by the worker.  */
};

static struct option const long_options[] = {
  {"count", required_argument, NULL, 'n'},
  {"min-score", required_argument, NULL, 'm'},
  {"idle", required_argument, NULL, 'i'},
  {"help", no_argument, NULL, 'h'},
  {"version", no_argument, NULL, 'V'},
  {NULL, 0, NULL, 0}
};

static _Noreturn void
usage (int status)
{
  if (status != EXIT_SUCCESS)
    {
      fprintf (stderr,
               "Try 'embed-search-provider --help' for more information.\n");
      exit (status);
    }
  printf ("Usage: embed-search-provider [OPTION]...\n"
          "Offer the files that best match what is typed in the GNOME "
          "Shell overview.\nD-Bus starts this program when it is "
          "needed.\n\n"
          "  -n, --count=N        offer at most N matches (default 5)\n"
          "  -m, --min-score=S    leave out matches scoring below S "
          "(default 0.6)\n"
          "  -i, --idle=SECONDS   exit after SECONDS without a search "
          "(default 300)\n"
          "      --help           display this help and exit\n"
          "      --version        output version information and exit\n");
  exit (status);
}

static gboolean
idle_timeout (gpointer data)
{
  (void) data;
  idle_source = 0;
  if (pending == 0)
    g_main_loop_quit (loop);
  return G_SOURCE_REMOVE;
}

/* Put off exiting for another IDLE_SECONDS.  */
static void
reset_idle (void)
{
  if (idle_source)
    g_source_remove (idle_source);
  idle_source = g_timeout_add_seconds (idle_seconds, idle_timeout, NULL);
}

/* Run the search for TEXT and append the matching paths to RESULTS.  */
static void
run_search (const char *text, GPtrArray *results)
{
  char *store = store_path (), *line = NULL, model_name[256];
  size_t line_size = 0, dim, qdim, n_latest, used;
  struct query q = { NULL, NULL, NULL, 0 };
  struct latest *latest;
  struct match *matches;
  struct buf query;
  long data_start;
  FILE *f;

  f = fopen (store, "r");
  if (!f)
    {
      g_warning ("%s: %s (run embed-index first)", store, strerror (errno));
      goto out;
    }
  if (getline (&line, &line_size, f) <= 0
      || !parse_header (line, model_name, sizeof model_name, &dim))
    {
      g_warning ("%s: missing or invalid header", store);
      goto out;
    }
  if (strcmp (model_name, EMBED_MODEL) != 0)
    {
      g_warning ("%s was built with %s; run embed-index to rebuild it",
                 store, model_name);
      goto out;
    }

  if (!model && !model_failed)
    {
      /* One short text at a time is quicker on the CPU.  */
      model = model_open (false);
      model_failed = !model;
    }
  if (!model)
    goto out;

  buf_init (&query);
  buf_puts (&query, QUERY_PREFIX);
  buf_puts (&query, text);
  if (!model_embed (model, query.data, &q.text, &qdim) || qdim != dim)
    {
      g_warning ("cannot embed the query");
      buf_free (&query);
      free (q.text);
      goto out;
    }
  buf_free (&query);
  query_prepare (&q, dim);

  data_start = ftell (f);
  latest = find_latest (f, &n_latest, NULL, NULL);
  if (fseek (f, data_start, SEEK_SET) == 0)
    {
      FILE *chunks = open_chunks (dim);

      matches = search_records (f, latest, n_latest, &q, dim, NULL, chunks,
                                count, &used);
      if (chunks)
        fclose (chunks);
      for (size_t i = 0; i < used; i++)
        if (matches[i].score >= min_score)
          g_ptr_array_add (results, g_strdup (matches[i].path));
      free_matches (matches, used);
    }
  free_latest (latest, n_latest);
  query_free (&q);

 out:
  if (f)
    fclose (f);
  free (line);
  free (store);
}

/* Back on the main thread: answer the search JOB.  */
static gboolean
finish_job (gpointer data)
{
  struct job *job = data;

  g_ptr_array_add (job->results, NULL);
  g_dbus_method_invocation_return_value
    (job->invocation,
     g_variant_new ("(^as)", (char **) job->results->pdata));
  g_ptr_array_free (job->results, TRUE);
  g_free (job->text);
  g_free (job);
  pending--;
  reset_idle ();
  return G_SOURCE_REMOVE;
}

static void
worker (gpointer data, gpointer user_data)
{
  struct job *job = data;

  (void) user_data;
  job->results = g_ptr_array_new_with_free_func (g_free);
  if (job->number == g_atomic_int_get (&newest))
    run_search (job->text, job->results);
  g_idle_add (finish_job, job);
}

/* Join TERMS into the text of a query.  */
static char *
join_terms (GVariant *terms)
{
  const char **v = g_variant_get_strv (terms, NULL);
  char *text = g_strstrip (g_strjoinv (" ", (char **) v));

  g_free (v);
  return text;
}

/* Queue a search for TERMS, to be answered on INVOCATION.  */
static void
start_search (GDBusMethodInvocation *invocation, GVariant *terms)
{
  struct job *job = g_new0 (struct job, 1);

  job->invocation = invocation;
  job->text = join_terms (terms);
  job->number = g_atomic_int_add (&newest, 1) + 1;
  if (g_utf8_strlen (job->text, -1) < (glong) min_query)
    {
      /* Too short to mean much: answer at once.  */
      g_atomic_int_inc (&newest);
      job->results = g_ptr_array_new_with_free_func (g_free);
      pending++;
      finish_job (job);
      return;
    }
  pending++;
  g_thread_pool_push (pool, job, NULL);
}

/* Return the meta data the Shell shows for the file PATH.  */
static GVariant *
result_meta (const char *path)
{
  GVariantBuilder b;
  GFile *file = g_file_new_for_path (path);
  GFileInfo *info;
  GIcon *icon = NULL;
  char *name = g_path_get_basename (path);
  char *dir = g_path_get_dirname (path);
  const char *home = g_get_home_dir ();
  size_t home_len = strlen (home);
  GVariant *serialized;

  g_variant_builder_init (&b, G_VARIANT_TYPE ("a{sv}"));
  g_variant_builder_add (&b, "{sv}", "id", g_variant_new_string (path));

  info = g_file_query_info (file,
                            G_FILE_ATTRIBUTE_STANDARD_DISPLAY_NAME ","
                            G_FILE_ATTRIBUTE_STANDARD_ICON ","
                            G_FILE_ATTRIBUTE_THUMBNAIL_PATH,
                            G_FILE_QUERY_INFO_NONE, NULL, NULL);
  if (info)
    {
      const char *thumb = g_file_info_get_attribute_byte_string
        (info, G_FILE_ATTRIBUTE_THUMBNAIL_PATH);
      const char *display = g_file_info_get_display_name (info);

      if (display)
        {
          g_free (name);
          name = g_strdup (display);
        }
      if (thumb)
        {
          GFile *t = g_file_new_for_path (thumb);
          icon = g_file_icon_new (t);
          g_object_unref (t);
        }
      else if (g_file_info_get_icon (info))
        icon = g_object_ref (g_file_info_get_icon (info));
    }
  g_variant_builder_add (&b, "{sv}", "name", g_variant_new_string (name));

  /* The folder, with the home directory shortened to ~.  */
  if (strncmp (dir, home, home_len) == 0
      && (dir[home_len] == '/' || dir[home_len] == '\0'))
    {
      char *short_dir = g_strconcat ("~", dir + home_len, NULL);
      g_free (dir);
      dir = short_dir;
    }
  g_variant_builder_add (&b, "{sv}", "description",
                         g_variant_new_string (dir));
  g_variant_builder_add (&b, "{sv}", "clipboardText",
                         g_variant_new_string (path));

  if (!icon)
    icon = g_themed_icon_new ("text-x-generic");
  serialized = g_icon_serialize (icon);
  if (serialized)
    {
      g_variant_builder_add (&b, "{sv}", "icon", serialized);
      g_variant_unref (serialized);
    }

  g_object_unref (icon);
  if (info)
    g_object_unref (info);
  g_object_unref (file);
  g_free (name);
  g_free (dir);
  return g_variant_builder_end (&b);
}

/* Open PATH with the default application for it.  */
static void
open_file (const char *path, guint32 timestamp)
{
  GAppLaunchContext *ctx = g_app_launch_context_new ();
  char *uri = g_filename_to_uri (path, NULL, NULL);
  char *ts = g_strdup_printf ("_TIME%u", timestamp);
  GError *err = NULL;

  g_app_launch_context_setenv (ctx, "DESKTOP_STARTUP_ID", ts);
  if (uri && !g_app_info_launch_default_for_uri (uri, ctx, &err))
    {
      g_warning ("cannot open %s: %s", path, err->message);
      g_error_free (err);
    }
  g_free (ts);
  g_free (uri);
  g_object_unref (ctx);
}

/* Show PATH in the file manager, selected in its folder.  */
static void
show_file (GDBusConnection *conn, const char *path)
{
  char *uri = g_filename_to_uri (path, NULL, NULL);
  const char *uris[] = { uri, NULL };

  if (!uri)
    return;
  g_dbus_connection_call (conn, "org.freedesktop.FileManager1",
                          "/org/freedesktop/FileManager1",
                          "org.freedesktop.FileManager1", "ShowItems",
                          g_variant_new ("(^ass)", uris, ""), NULL,
                          G_DBUS_CALL_FLAGS_NONE, -1, NULL, NULL, NULL);
  g_free (uri);
}

/* For LaunchSearch: show the best match, once its search is done.  */
static void
launch_done (GObject *source, GAsyncResult *res, gpointer data)
{
  GDBusConnection *conn = G_DBUS_CONNECTION (source);
  GVariant *reply = g_dbus_connection_call_finish (conn, res, NULL);
  const char **paths;

  (void) data;
  if (!reply)
    return;
  g_variant_get (reply, "(^a&s)", &paths);
  if (paths[0])
    show_file (conn, paths[0]);
  g_free (paths);
  g_variant_unref (reply);
}

static void
method_call (GDBusConnection *conn, const char *sender,
             const char *object_path, const char *interface_name,
             const char *method_name, GVariant *parameters,
             GDBusMethodInvocation *invocation, gpointer user_data)
{
  (void) sender;
  (void) object_path;
  (void) interface_name;
  (void) user_data;

  reset_idle ();
  if (strcmp (method_name, "GetInitialResultSet") == 0)
    {
      GVariant *terms = g_variant_get_child_value (parameters, 0);
      start_search (invocation, terms);
      g_variant_unref (terms);
    }
  else if (strcmp (method_name, "GetSubsearchResultSet") == 0)
    {
      /* Narrowing the earlier results would miss files that only the
         longer query brings up, so search afresh.  */
      GVariant *terms = g_variant_get_child_value (parameters, 1);
      start_search (invocation, terms);
      g_variant_unref (terms);
    }
  else if (strcmp (method_name, "GetResultMetas") == 0)
    {
      GVariantBuilder b;
      GVariantIter *iter;
      const char *id;

      g_variant_builder_init (&b, G_VARIANT_TYPE ("aa{sv}"));
      g_variant_get (parameters, "(as)", &iter);
      while (g_variant_iter_next (iter, "&s", &id))
        g_variant_builder_add_value (&b, result_meta (id));
      g_variant_iter_free (iter);
      g_dbus_method_invocation_return_value (invocation,
                                             g_variant_new ("(aa{sv})", &b));
    }
  else if (strcmp (method_name, "ActivateResult") == 0)
    {
      const char *id;
      guint32 timestamp;

      g_variant_get (parameters, "(&s@asu)", &id, NULL, &timestamp);
      open_file (id, timestamp);
      g_dbus_method_invocation_return_value (invocation, NULL);
    }
  else if (strcmp (method_name, "LaunchSearch") == 0)
    {
      /* Ask ourselves for the results, through the same queue.  */
      GVariant *terms = g_variant_get_child_value (parameters, 0);

      g_dbus_connection_call (conn, g_dbus_connection_get_unique_name (conn),
                              OBJECT_PATH, "org.gnome.Shell.SearchProvider2",
                              "GetInitialResultSet",
                              g_variant_new_tuple (&terms, 1),
                              G_VARIANT_TYPE ("(as)"),
                              G_DBUS_CALL_FLAGS_NONE, -1, NULL,
                              launch_done, NULL);
      g_variant_unref (terms);
      g_dbus_method_invocation_return_value (invocation, NULL);
    }
  else
    g_dbus_method_invocation_return_error (invocation, G_DBUS_ERROR,
                                           G_DBUS_ERROR_UNKNOWN_METHOD,
                                           "No such method: %s",
                                           method_name);
}

static const GDBusInterfaceVTable vtable = { method_call, NULL, NULL, { 0 } };

static void
bus_acquired (GDBusConnection *conn, const char *name, gpointer data)
{
  GDBusNodeInfo *info = data;
  GError *err = NULL;

  (void) name;
  if (!g_dbus_connection_register_object (conn, OBJECT_PATH,
                                          info->interfaces[0], &vtable,
                                          NULL, NULL, &err))
    {
      g_warning ("cannot register %s: %s", OBJECT_PATH, err->message);
      g_error_free (err);
      g_main_loop_quit (loop);
    }
}

static void
name_lost (GDBusConnection *conn, const char *name, gpointer data)
{
  (void) data;
  if (!conn)
    g_warning ("cannot connect to the session bus");
  else
    g_warning ("cannot own %s; is another provider running?", name);
  g_main_loop_quit (loop);
}

int
main (int argc, char **argv)
{
  GDBusNodeInfo *info;
  guint owner;
  int c;

  while ((c = getopt_long (argc, argv, "n:m:i:", long_options, NULL)) != -1)
    switch (c)
      {
      case 'n':
        {
          long n;
          if (!parse_positive (optarg, 1000, &n))
            error (EXIT_FAILURE, 0, "invalid count: %s", optarg);
          count = n;
        }
        break;
      case 'm':
        {
          char *end;
          errno = 0;
          min_score = strtod (optarg, &end);
          if (errno || end == optarg || *end)
            error (EXIT_FAILURE, 0, "invalid score: %s", optarg);
        }
        break;
      case 'i':
        {
          long n;
          if (!parse_positive (optarg, 86400, &n))
            error (EXIT_FAILURE, 0, "invalid idle time: %s", optarg);
          idle_seconds = n;
        }
        break;
      case 'h':
        usage (EXIT_SUCCESS);
      case 'V':
        printf ("embed-search-provider %s\n", PACKAGE_VERSION);
        return EXIT_SUCCESS;
      default:
        usage (EXIT_FAILURE);
      }
  if (optind < argc)
    {
      error (0, 0, "extra operand: %s", argv[optind]);
      usage (EXIT_FAILURE);
    }

  info = g_dbus_node_info_new_for_xml (introspection_xml, NULL);
  loop = g_main_loop_new (NULL, FALSE);
  pool = g_thread_pool_new (worker, NULL, 1, TRUE, NULL);
  owner = g_bus_own_name (G_BUS_TYPE_SESSION, BUS_NAME,
                          G_BUS_NAME_OWNER_FLAGS_NONE, bus_acquired, NULL,
                          name_lost, info, NULL);
  reset_idle ();
  g_main_loop_run (loop);

  g_bus_unown_name (owner);
  g_thread_pool_free (pool, FALSE, TRUE);
  if (model)
    model_close (model);
  g_main_loop_unref (loop);
  g_dbus_node_info_unref (info);
  return EXIT_SUCCESS;
}
