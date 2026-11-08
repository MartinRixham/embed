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

clean:
	rm -f $(PROGRAMS) *.o

.PHONY: all install clean
