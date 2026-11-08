# Makefile for embed-index and embed-query.

CC ?= gcc
CFLAGS ?= -O2 -g -Wall -Wextra
CPPFLAGS += -D_GNU_SOURCE $(shell pkg-config --cflags libcurl)
LDLIBS_COMMON = $(shell pkg-config --libs libcurl) -lm
prefix ?= $(HOME)/.local
bindir ?= $(prefix)/bin

PROGRAMS = embed-index embed-query

all: $(PROGRAMS)

embed-index: embed-index.o common.o server.o
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS_COMMON) -lmagic

embed-query: embed-query.o common.o server.o
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS_COMMON)

embed-index.o embed-query.o common.o server.o: common.h
embed-index.o embed-query.o server.o: server.h

install: all
	install -d $(DESTDIR)$(bindir)
	install -m 755 $(PROGRAMS) $(DESTDIR)$(bindir)

# Unit tests.  They are built with the address and undefined behaviour
# sanitizers, in their own directory so as not to mix with the
# programs' objects.  Each test file includes the source file it tests,
# so that static functions can be tested directly.

TEST_DIR = tests
TEST_BUILD = $(TEST_DIR)/build
TEST_CFLAGS = -O1 -g -Wall -Wextra -fno-omit-frame-pointer \
  -fsanitize=address,undefined -fno-sanitize-recover=undefined
TEST_CPPFLAGS = $(CPPFLAGS) -I. \
  -DFAKE_OLLAMA='"$(abspath $(TEST_BUILD)/fake-ollama)"' \
  -DEMBED_INDEX='"$(abspath embed-index)"'
TEST_LIBS = $(shell pkg-config --libs check) $(LDLIBS_COMMON)
TESTS = test-common test-server test-index test-query
TEST_PROGRAMS = $(addprefix $(TEST_BUILD)/,$(TESTS))
TEST_SUPPORT = $(TEST_BUILD)/testutil.o $(TEST_BUILD)/fake-http.o
TEST_HEADERS = $(TEST_DIR)/testutil.h $(TEST_DIR)/fake-http.h common.h \
  server.h

$(TEST_BUILD)/%.o: $(TEST_DIR)/%.c $(TEST_HEADERS) | $(TEST_BUILD)
	$(CC) $(TEST_CPPFLAGS) $(TEST_CFLAGS) -c -o $@ $<

$(TEST_BUILD)/src-%.o: %.c common.h server.h | $(TEST_BUILD)
	$(CC) $(TEST_CPPFLAGS) $(TEST_CFLAGS) -c -o $@ $<

$(TEST_BUILD)/fake-ollama: $(TEST_BUILD)/fake-ollama.o \
  $(TEST_BUILD)/fake-http.o
	$(CC) $(TEST_CFLAGS) -o $@ $^

$(TEST_BUILD)/test-common: $(TEST_BUILD)/test-common.o $(TEST_SUPPORT)
	$(CC) $(TEST_CFLAGS) -o $@ $^ $(TEST_LIBS)

$(TEST_BUILD)/test-server: $(TEST_BUILD)/test-server.o \
  $(TEST_BUILD)/src-common.o $(TEST_SUPPORT)
	$(CC) $(TEST_CFLAGS) -o $@ $^ $(TEST_LIBS)

$(TEST_BUILD)/test-index: $(TEST_BUILD)/test-index.o \
  $(TEST_BUILD)/src-common.o $(TEST_BUILD)/src-server.o $(TEST_SUPPORT)
	$(CC) $(TEST_CFLAGS) -o $@ $^ $(TEST_LIBS) -lmagic -lutil

$(TEST_BUILD)/test-query: $(TEST_BUILD)/test-query.o \
  $(TEST_BUILD)/src-common.o $(TEST_BUILD)/src-server.o $(TEST_SUPPORT)
	$(CC) $(TEST_CFLAGS) -o $@ $^ $(TEST_LIBS)

# Each test depends on the source file it includes.
$(TEST_BUILD)/test-common.o: common.c
$(TEST_BUILD)/test-server.o: server.c
$(TEST_BUILD)/test-index.o: embed-index.c
$(TEST_BUILD)/test-query.o: embed-query.c

$(TEST_BUILD):
	mkdir -p $@

# Run every test program, even after one fails.  Set TESTS to run only
# some, for example "make check TESTS=test-common".
check: $(TEST_PROGRAMS) $(TEST_BUILD)/fake-ollama embed-index
	@status=0; \
	for t in $(TEST_PROGRAMS); do \
	  echo "== $$t"; \
	  ASAN_OPTIONS=detect_leaks=0:allocator_may_return_null=1 $$t || status=1; \
	done; \
	exit $$status

clean:
	rm -f $(PROGRAMS) *.o
	rm -rf $(TEST_BUILD)

.PHONY: all install check clean
