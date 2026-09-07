// reader.c - safetensors mmap store with manifest bsearch
#define _GNU_SOURCE
#include "qwenflash.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

static int cmp_entry(const void *a, const void *b) {
    return strcmp(((const QfEntry *)a)->name, ((const QfEntry *)b)->name);
}

int qf_store_open(QfStore *st, const char *dir) {
    memset(st, 0, sizeof(*st));
    char path[1024];
    snprintf(path, sizeof(path), "%s/shards.txt", dir);
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    int cap = 256;
    st->shard_paths = (char (*)[1024])calloc(cap, 1024);
    char line[1024];
    while (fgets(line, sizeof(line), f)) {
        line[strcspn(line, "\n")] = 0;
        if (!line[0]) continue;
        if (st->n_shards == cap) {
            cap *= 2;
            st->shard_paths = (char (*)[1024])realloc(st->shard_paths, cap * 1024);
        }
        memcpy(st->shard_paths[st->n_shards], line, 1024);
        st->n_shards++;
    }
    fclose(f);

    snprintf(path, sizeof(path), "%s/manifest.bin", dir);
    f = fopen(path, "rb");
    if (!f) return -2;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    st->n_entries = sz / QF_REC_SIZE;
    st->entries = (QfEntry *)malloc(sz);
    fread(st->entries, 1, sz, f);
    fclose(f);
    qsort(st->entries, st->n_entries, sizeof(QfEntry), cmp_entry);

    st->shard_fds = (int *)calloc(st->n_shards, sizeof(int));
    st->shard_maps = (void **)calloc(st->n_shards, sizeof(void *));
    st->shard_sizes = (size_t *)calloc(st->n_shards, sizeof(size_t));
    for (int i = 0; i < st->n_shards; i++) {
        st->shard_fds[i] = open(st->shard_paths[i], O_RDONLY);
        if (st->shard_fds[i] < 0) return -3;
        struct stat sb;
        fstat(st->shard_fds[i], &sb);
        st->shard_sizes[i] = sb.st_size;
        st->shard_maps[i] = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE,
                                 st->shard_fds[i], 0);
        if (st->shard_maps[i] == MAP_FAILED) return -4;
    }
    return 0;
}

const QfEntry *qf_find(const QfStore *st, const char *name) {
    int lo = 0, hi = st->n_entries - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        int c = strcmp(st->entries[mid].name, name);
        if (c == 0) return &st->entries[mid];
        if (c < 0) lo = mid + 1; else hi = mid - 1;
    }
    return NULL;
}

void qf_store_close(QfStore *st) {
    for (int i = 0; i < st->n_shards; i++) {
        if (st->shard_maps[i]) munmap(st->shard_maps[i], st->shard_sizes[i]);
        if (st->shard_fds[i] > 0) close(st->shard_fds[i]);
    }
    free(st->entries);
}

// Drop-behind for the full-resident loader (QF_EXPERT_MODE=full). GB10 is
// unified memory: pages faulted in from the mmap'd shards come out of the
// same pool as device allocations, so without dropping, reading ~63 GiB of
// expert weights would leave a second copy of all of them in the page cache.
// The loader stages each tensor through the fixed pinned buffer and drops the
// source range here immediately after the copy.
void qf_store_drop_range(QfStore *st, int file_idx, uint64_t off, uint64_t len) {
    if (!st || file_idx < 0 || file_idx >= st->n_shards || !len) return;
    if (!st->shard_maps[file_idx] || st->shard_fds[file_idx] <= 0) return;
    long pg = sysconf(_SC_PAGESIZE);
    if (pg <= 0) pg = 4096;
    uint64_t a = off & ~(uint64_t)(pg - 1);              // madvise: page-align down
    uint64_t end = off + len;
    if (end > st->shard_sizes[file_idx]) end = st->shard_sizes[file_idx];
    if (end > a)
        madvise((char *)st->shard_maps[file_idx] + a, end - a, MADV_DONTNEED);
    // The mapping is read-only MAP_PRIVATE, so DONTNEED can only discard clean
    // pages we no longer need. posix_fadvise drops the page-cache copy held via
    // the fd as well; it is Linux-specific (the target).
#ifdef POSIX_FADV_DONTNEED
    posix_fadvise(st->shard_fds[file_idx], (off_t)off, (off_t)len,
                  POSIX_FADV_DONTNEED);
#endif
}

void qf_store_drop_all(QfStore *st) {
    if (!st) return;
    for (int i = 0; i < st->n_shards; i++) {
        if (st->shard_maps[i])
            madvise(st->shard_maps[i], st->shard_sizes[i], MADV_DONTNEED);
#ifdef POSIX_FADV_DONTNEED
        if (st->shard_fds[i] > 0)
            posix_fadvise(st->shard_fds[i], 0, 0, POSIX_FADV_DONTNEED);
#endif
    }
}
