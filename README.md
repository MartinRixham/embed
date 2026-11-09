# embed

Find files by meaning rather than by name.  `embed-index` embeds every
file under your home directory with local models, and `embed-query`
and the GNOME Shell search provider look up the files that best match
what you type.  Nothing leaves the machine, apart from downloading the
models once.

- Text is embedded by [nomic-embed-text-v1.5].  Each file becomes a
  short description of its name, location and type and, where
  possible, its content: plain text, PDF text, office document text
  (OpenDocument and Office Open XML), EPUB book metadata and text, OCR
  of images, audio and video metadata, and a transcript of what is
  said.
- Images and videos are also embedded by what they show, with
  [nomic-embed-vision-v1.5], into the same space as the text, so a text
  query can find a photo.
- Speech in audio and video is transcribed by [Whisper] (base).

The models run on the GPU through OpenVINO when there is a usable one,
and otherwise on the CPU through ONNX Runtime.

[nomic-embed-text-v1.5]: https://huggingface.co/nomic-ai/nomic-embed-text-v1.5
[nomic-embed-vision-v1.5]: https://huggingface.co/nomic-ai/nomic-embed-vision-v1.5
[Whisper]: https://huggingface.co/onnx-community/whisper-base

## Programs

**embed-index** `[-a] [-v] [--cpu] [DIRECTORY]...` embeds every file
under each directory (by default `$HOME`), leaving out hidden files
unless `-a` is given.  Files whose size and modification time are
unchanged since the last run are skipped, so it is cheap to rerun.  An
interrupted run keeps what it has done.

**embed-query** `[-n N] [-s] QUERY...` lists the files that best match
the query, best first.  With `--file=FILE` it lists the files most like
`FILE` instead: a document, image, video or recording.

**embed-search-provider** offers the best matches in the GNOME Shell
Activities overview.  D-Bus starts it on the first search, and it exits
after five idle minutes.

Each program takes `--help`.

## Building

You need a C compiler, `make`, `pkg-config`, and the development files
of libcurl, OpenSSL (libcrypto), ONNX Runtime, OpenVINO (its C
library), utf8proc, libmagic and GLib/GIO.  At run time `embed-index`
also uses `ffmpeg` and `ffprobe`, `pdftotext` (poppler), `tesseract`
and `unzip`; a file it cannot read with them is embedded by its name
and type alone.  On Fedora:

    sudo dnf install gcc make pkgconf-pkg-config libcurl-devel \
      openssl-devel onnxruntime-devel libopenvino-devel \
      libopenvino-intel-gpu-plugin utf8proc-devel file-devel \
      glib2-devel ffmpeg poppler-utils tesseract unzip

Then:

    make
    sudo make install

`make install` installs into `/usr/local` by default (set `prefix` to
change it).  It must be system wide, since GNOME Shell reads search
providers only from the system data directories.  It also installs and
enables a systemd user timer that runs `embed-index` for every user 15
minutes after they log in and then hourly, at idle priority and only on
AC power.  To build the first index straight away, run `embed-index`
yourself.

## Files

| Path                                   | What                         |
|----------------------------------------|------------------------------|
| `~/.local/share/embed/embeddings.txt`  | the store, one file per line |
| `~/.local/share/embed/models/`         | downloaded models (~750 MB)  |
| `~/.cache/embed/openvino/`             | OpenVINO's compiled kernels  |

The models are pinned to one revision each and checked by their
SHA-256 hashes.

## Source

| Path                         | What                                       |
|------------------------------|--------------------------------------------|
| `src/embed-index.c`          | the indexer                                |
| `src/embed-query.c`          | the command-line search                    |
| `src/embed-search-provider.c`| the GNOME Shell search provider (D-Bus)    |
| `src/describe.c`             | turning a file into its text and picture   |
| `src/search.c`               | scoring the store against a query          |
| `src/model.c`                | the models: download, open, embed          |
| `src/onnx-embed.c`           | running a model with ONNX Runtime (CPU)    |
| `src/ov-embed.c`             | running a model with OpenVINO (GPU)        |
| `src/tokenizer.c`            | nomic-embed-text's WordPiece tokenizer     |
| `src/whisper.c`              | speech transcription                       |
| `src/fetch.c`                | downloading and checking files             |
| `src/common.c`               | shared helpers                             |
| `data/`                      | desktop, D-Bus and systemd files           |
| `tests/`                     | unit tests                                 |
| `tools/`                     | tokenizer checking tools                   |

## Testing

    make check

builds the tests with the address and undefined behaviour sanitizers
(they need [Check]) and runs them all; `make check TESTS=test-common`
runs just one.  The indexer and query tests use a fake model;
`test-model` checks the real models if they have been downloaded.

[Check]: https://libcheck.github.io/check/

## Licence

GNU General Public License, version 3 or later.
