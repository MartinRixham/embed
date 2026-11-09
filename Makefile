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
GIO_CFLAGS = $(shell pkg-config --cflags gio-2.0)
GIO_LIBS = $(shell pkg-config --libs gio-2.0)

PROGRAMS = embed-index embed-query embed-search-provider
MODEL_OBJS = model.o fetch.o tokenizer.o onnx-embed.o ov-embed.o \
  whisper.o

all: $(PROGRAMS)

embed-index: embed-index.o describe.o common.o $(MODEL_OBJS)
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS_MODEL) -lmagic

embed-query: embed-query.o search.o describe.o common.o $(MODEL_OBJS)
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS_MODEL) -lmagic

embed-search-provider: embed-search-provider.o search.o common.o \
  $(MODEL_OBJS)
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS_MODEL) $(GIO_LIBS)

embed-search-provider.o: embed-search-provider.c
	$(CC) $(CPPFLAGS) $(GIO_CFLAGS) $(CFLAGS) -c -o $@ $<

common.o describe.o embed-index.o embed-query.o embed-search-provider.o \
  fetch.o model.o onnx-embed.o ov-embed.o search.o tokenizer.o \
  whisper.o: common.h
describe.o embed-index.o embed-query.o embed-search-provider.o \
  model.o: model.h
embed-query.o embed-search-provider.o search.o: search.h
describe.o embed-index.o embed-query.o: describe.h
model.o fetch.o: fetch.h
model.o onnx-embed.o: onnx-embed.h
model.o ov-embed.o: ov-embed.h
model.o tokenizer.o: tokenizer.h
model.o whisper.o: whisper.h
whisper.o: ov-embed.h

SUBST = sed 's|@bindir@|$(bindir)|g'

# GNOME Shell picks the search provider up when you next log in.
install: all
	install -d $(DESTDIR)$(bindir) $(DESTDIR)$(datadir)/applications \
	  $(DESTDIR)$(datadir)/dbus-1/services $(DESTDIR)$(searchproviderdir)
	install -m 755 $(PROGRAMS) $(DESTDIR)$(bindir)
	$(SUBST) data/embed-search.desktop.in \
	  > $(DESTDIR)$(datadir)/applications/embed-search.desktop
	$(SUBST) data/embed.SearchProvider.service.in \
	  > $(DESTDIR)$(datadir)/dbus-1/services/embed.SearchProvider.service
	install -m 644 data/embed-search-provider.ini \
	  $(DESTDIR)$(searchproviderdir)/embed-search-provider.ini

# A tool for checking the tokenizer by hand.
tools/tokenize: tools/tokenize.c tokenizer.c common.c tokenizer.h common.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -I. -o $@ tools/tokenize.c tokenizer.c \
	  common.c $(shell pkg-config --libs libutf8proc)

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
TEST_CPPFLAGS = $(CPPFLAGS) -I. \
  -DEMBED_INDEX='"$(abspath $(TEST_BUILD)/embed-index)"' \
  -DTEST_DATA='"$(abspath $(TEST_DIR)/data)"'
TEST_LIBS = $(shell pkg-config --libs check) $(LDLIBS_MODEL)
TESTS = test-common test-tokenizer test-whisper test-fetch test-index \
  test-query test-model
TEST_PROGRAMS = $(addprefix $(TEST_BUILD)/,$(TESTS))
TEST_SUPPORT = $(TEST_BUILD)/testutil.o $(TEST_BUILD)/fake-http.o
TEST_HEADERS = $(wildcard $(TEST_DIR)/*.h) $(wildcard *.h)

$(TEST_BUILD)/%.o: $(TEST_DIR)/%.c $(TEST_HEADERS) | $(TEST_BUILD)
	$(CC) $(TEST_CPPFLAGS) $(TEST_CFLAGS) -c -o $@ $<

$(TEST_BUILD)/src-%.o: %.c $(wildcard *.h) | $(TEST_BUILD)
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

$(TEST_BUILD)/plain-%.o: %.c $(wildcard *.h) | $(TEST_BUILD)
	$(CC) $(TEST_CPPFLAGS) $(PLAIN_CFLAGS) -c -o $@ $<

$(TEST_BUILD)/plain-test-%.o: $(TEST_DIR)/%.c $(TEST_HEADERS) | $(TEST_BUILD)
	$(CC) $(TEST_CPPFLAGS) $(PLAIN_CFLAGS) -c -o $@ $<

$(TEST_BUILD)/test-model: $(TEST_BUILD)/plain-test-test-model.o \
  $(addprefix $(TEST_BUILD)/plain-,common.o fetch.o tokenizer.o \
  onnx-embed.o ov-embed.o whisper.o) $(TEST_BUILD)/plain-test-testutil.o \
  $(TEST_BUILD)/plain-test-fake-http.o
	$(CC) $(PLAIN_CFLAGS) -o $@ $^ $(TEST_LIBS)

# Each test depends on the source file it includes.
$(TEST_BUILD)/test-common.o: common.c
$(TEST_BUILD)/test-tokenizer.o: tokenizer.c
$(TEST_BUILD)/test-whisper.o: whisper.c
$(TEST_BUILD)/test-fetch.o: fetch.c
$(TEST_BUILD)/test-index.o: embed-index.c describe.c
$(TEST_BUILD)/test-query.o: embed-query.c
$(TEST_BUILD)/plain-test-test-model.o: model.c

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
	rm -f $(PROGRAMS) *.o tools/tokenize
	rm -rf $(TEST_BUILD)

.PHONY: all install check clean
