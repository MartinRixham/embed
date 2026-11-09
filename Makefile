# Makefile for embed-index, embed-query and embed-search-provider.

CC ?= gcc
CFLAGS ?= -O2 -g -Wall -Wextra
PKGS = libcurl libcrypto libonnxruntime libutf8proc
CPPFLAGS += -D_GNU_SOURCE $(shell pkg-config --cflags $(PKGS))
# OpenVINO's pkg-config file names libraries that only its C++ API
# needs, so link its C library directly.
LDLIBS_MODEL = $(shell pkg-config --libs $(PKGS)) -lopenvino_c -lm
# Install system wide, with "sudo make install": GNOME Shell reads
# search providers only from the system data directories.
prefix ?= /usr/local
bindir ?= $(prefix)/bin
datadir ?= $(prefix)/share
searchproviderdir ?= $(datadir)/gnome-shell/search-providers
userunitdir ?= $(prefix)/lib/systemd/user
GIO_CFLAGS = $(shell pkg-config --cflags gio-2.0)
GIO_LIBS = $(shell pkg-config --libs gio-2.0)

PROGRAMS = embed-index embed-query embed-search-provider
MODEL_OBJS = src/model.o src/fetch.o src/tokenizer.o src/onnx-embed.o \
  src/ov-embed.o src/whisper.o

all: $(PROGRAMS)

embed-index: src/embed-index.o src/describe.o src/common.o $(MODEL_OBJS)
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS_MODEL) -lmagic

embed-query: src/embed-query.o src/search.o src/describe.o src/common.o \
  $(MODEL_OBJS)
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS_MODEL) -lmagic

embed-search-provider: src/embed-search-provider.o src/search.o \
  src/common.o $(MODEL_OBJS)
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS_MODEL) $(GIO_LIBS)

src/embed-search-provider.o: src/embed-search-provider.c
	$(CC) $(CPPFLAGS) $(GIO_CFLAGS) $(CFLAGS) -c -o $@ $<

src/common.o src/describe.o src/embed-index.o src/embed-query.o \
  src/embed-search-provider.o src/fetch.o src/model.o src/onnx-embed.o \
  src/ov-embed.o src/search.o src/tokenizer.o src/whisper.o: src/common.h
src/describe.o src/embed-index.o src/embed-query.o \
  src/embed-search-provider.o src/model.o: src/model.h
src/embed-query.o src/embed-search-provider.o src/search.o: src/search.h
src/describe.o src/embed-index.o src/embed-query.o: src/describe.h
src/model.o src/fetch.o: src/fetch.h
src/model.o src/onnx-embed.o: src/onnx-embed.h
src/model.o src/ov-embed.o: src/ov-embed.h
src/model.o src/tokenizer.o: src/tokenizer.h
src/model.o src/whisper.o: src/whisper.h
src/whisper.o: src/ov-embed.h

SUBST = sed 's|@bindir@|$(bindir)|g'

# GNOME Shell picks the search provider up, and each user's session
# starts indexing every hour, when they next log in.
install: all
	install -d $(DESTDIR)$(bindir) $(DESTDIR)$(datadir)/applications \
	  $(DESTDIR)$(datadir)/dbus-1/services $(DESTDIR)$(searchproviderdir) \
	  $(DESTDIR)$(userunitdir)
	install -m 755 $(PROGRAMS) $(DESTDIR)$(bindir)
	$(SUBST) data/embed-search.desktop.in \
	  > $(DESTDIR)$(datadir)/applications/embed-search.desktop
	$(SUBST) data/embed.SearchProvider.service.in \
	  > $(DESTDIR)$(datadir)/dbus-1/services/embed.SearchProvider.service
	install -m 644 data/embed-search-provider.ini \
	  $(DESTDIR)$(searchproviderdir)/embed-search-provider.ini
	$(SUBST) data/embed-index.service.in \
	  > $(DESTDIR)$(userunitdir)/embed-index.service
	install -m 644 data/embed-index.timer $(DESTDIR)$(userunitdir)
	if [ -z "$(DESTDIR)" ]; then \
	  systemctl --global enable embed-index.timer; \
	fi

# A tool for checking the tokenizer by hand.
tools/tokenize: tools/tokenize.c src/tokenizer.c src/common.c \
  src/tokenizer.h src/common.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -Isrc -o $@ tools/tokenize.c src/tokenizer.c \
	  src/common.c $(shell pkg-config --libs libutf8proc)

# Unit tests.  They are built with the address and undefined behaviour
# sanitizers, in their own directory so as not to mix with the
# programs' objects.  Most test files include the source file they
# test, so that static functions can be tested directly.  embed-index
# and embed-query are tested with a fake model (tests/fake-model.c);
# test-model checks the real one, if it has been downloaded.

TEST_DIR = tests
TEST_BUILD = $(TEST_DIR)/build
TEST_CFLAGS = -O1 -g -Wall -Wextra -fno-omit-frame-pointer \
  -fsanitize=address,undefined -fno-sanitize-recover=undefined
TEST_CPPFLAGS = $(CPPFLAGS) -Isrc \
  -DEMBED_INDEX='"$(abspath $(TEST_BUILD)/embed-index)"' \
  -DTEST_DATA='"$(abspath $(TEST_DIR)/data)"'
TEST_LIBS = $(shell pkg-config --libs check) $(LDLIBS_MODEL)
TESTS = test-common test-tokenizer test-whisper test-fetch test-index \
  test-query test-model
TEST_PROGRAMS = $(addprefix $(TEST_BUILD)/,$(TESTS))
TEST_SUPPORT = $(TEST_BUILD)/testutil.o $(TEST_BUILD)/fake-http.o
TEST_HEADERS = $(wildcard $(TEST_DIR)/*.h) $(wildcard src/*.h)

$(TEST_BUILD)/%.o: $(TEST_DIR)/%.c $(TEST_HEADERS) | $(TEST_BUILD)
	$(CC) $(TEST_CPPFLAGS) $(TEST_CFLAGS) -c -o $@ $<

$(TEST_BUILD)/src-%.o: src/%.c $(wildcard src/*.h) | $(TEST_BUILD)
	$(CC) $(TEST_CPPFLAGS) $(TEST_CFLAGS) -c -o $@ $<

# embed-index with the fake model, for the round trip in test-query.
$(TEST_BUILD)/embed-index: $(TEST_BUILD)/src-embed-index.o \
  $(TEST_BUILD)/src-describe.o $(TEST_BUILD)/src-common.o \
  $(TEST_BUILD)/fake-model.o
	$(CC) $(TEST_CFLAGS) -o $@ $^ -lmagic -lm

$(TEST_BUILD)/test-common: $(TEST_BUILD)/test-common.o $(TEST_SUPPORT)
	$(CC) $(TEST_CFLAGS) -o $@ $^ $(TEST_LIBS)

$(TEST_BUILD)/test-whisper: $(TEST_BUILD)/test-whisper.o \
  $(TEST_BUILD)/src-common.o $(TEST_BUILD)/src-ov-embed.o $(TEST_SUPPORT)
	$(CC) $(TEST_CFLAGS) -o $@ $^ $(TEST_LIBS)

$(TEST_BUILD)/test-tokenizer: $(TEST_BUILD)/test-tokenizer.o \
  $(TEST_BUILD)/src-common.o $(TEST_SUPPORT)
	$(CC) $(TEST_CFLAGS) -o $@ $^ $(TEST_LIBS)

$(TEST_BUILD)/test-fetch: $(TEST_BUILD)/test-fetch.o \
  $(TEST_BUILD)/src-common.o $(TEST_SUPPORT)
	$(CC) $(TEST_CFLAGS) -o $@ $^ $(TEST_LIBS)

$(TEST_BUILD)/test-index: $(TEST_BUILD)/test-index.o \
  $(TEST_BUILD)/src-common.o $(TEST_BUILD)/fake-model.o $(TEST_SUPPORT)
	$(CC) $(TEST_CFLAGS) -o $@ $^ $(TEST_LIBS) -lmagic -lutil

$(TEST_BUILD)/test-query: $(TEST_BUILD)/test-query.o \
  $(TEST_BUILD)/src-search.o $(TEST_BUILD)/src-describe.o $(TEST_BUILD)/src-common.o \
  $(TEST_BUILD)/fake-model.o $(TEST_SUPPORT)
	$(CC) $(TEST_CFLAGS) -o $@ $^ $(TEST_LIBS) -lmagic

# test-model runs the real ONNX Runtime and OpenVINO, which cannot run
# under the address sanitizer (OpenVINO loads its threading library
# with RTLD_DEEPBIND), so it is built without the sanitizers.
PLAIN_CFLAGS = -O1 -g -Wall -Wextra

$(TEST_BUILD)/plain-%.o: src/%.c $(wildcard src/*.h) | $(TEST_BUILD)
	$(CC) $(TEST_CPPFLAGS) $(PLAIN_CFLAGS) -c -o $@ $<

$(TEST_BUILD)/plain-test-%.o: $(TEST_DIR)/%.c $(TEST_HEADERS) | $(TEST_BUILD)
	$(CC) $(TEST_CPPFLAGS) $(PLAIN_CFLAGS) -c -o $@ $<

$(TEST_BUILD)/test-model: $(TEST_BUILD)/plain-test-test-model.o \
  $(addprefix $(TEST_BUILD)/plain-,common.o fetch.o tokenizer.o \
  onnx-embed.o ov-embed.o whisper.o) $(TEST_BUILD)/plain-test-testutil.o \
  $(TEST_BUILD)/plain-test-fake-http.o
	$(CC) $(PLAIN_CFLAGS) -o $@ $^ $(TEST_LIBS)

# Each test depends on the source file it includes.
$(TEST_BUILD)/test-common.o: src/common.c
$(TEST_BUILD)/test-tokenizer.o: src/tokenizer.c
$(TEST_BUILD)/test-whisper.o: src/whisper.c
$(TEST_BUILD)/test-fetch.o: src/fetch.c
$(TEST_BUILD)/test-index.o: src/embed-index.c src/describe.c
$(TEST_BUILD)/test-query.o: src/embed-query.c
$(TEST_BUILD)/plain-test-test-model.o: src/model.c

$(TEST_BUILD):
	mkdir -p $@

# Run every test program, even after one fails.  Set TESTS to run only
# some, for example "make check TESTS=test-common".
check: $(TEST_PROGRAMS) $(TEST_BUILD)/embed-index
	@status=0; \
	for t in $(TEST_PROGRAMS); do \
	  echo "== $$t"; \
	  ASAN_OPTIONS=detect_leaks=0:allocator_may_return_null=1 $$t || status=1; \
	done; \
	exit $$status

clean:
	rm -f $(PROGRAMS) src/*.o tools/tokenize
	rm -rf $(TEST_BUILD)

.PHONY: all install check clean
