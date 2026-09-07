#include "qwenflash.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static int cmp_entry(const void *a, const void *b) {
    return strcmp(((const QfEntry *) a)->name, ((const QfEntry *) b)->name);
}

int qf_store_open(QfStore *st, const char *dir) {
    memset(st, 0, sizeof(*st));
    char path[1024];
    snprintf(path, sizeof(path), "%s/shards.txt", dir);
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    int cap = 256;
    st->shard_paths = (char (*)[1024]) calloc(cap, 1024);
    char line[1024];
    while (fgets(line, sizeof(line), f)) {
        line[strcspn(line, "\n")] = 0;
        if (!line[0]) continue;
        if (st->n_shards == cap) {
            cap *= 2;
            st->shard_paths = (char (*)[1024]) realloc(st->shard_paths, (size_t) cap * 1024);
        }
        snprintf(st->shard_paths[st->n_shards++], 1024, "%s", line);
    }
    fclose(f);

    snprintf(path, sizeof(path), "%s/manifest.bin", dir);
    f = fopen(path, "rb");
    if (!f) return -2;
    fseek(f, 0, SEEK_END);
    const long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    st->n_entries = (int) (sz / QF_REC_SIZE);
    st->entries = (QfEntry *) malloc((size_t) sz);
    if (!st->entries || fread(st->entries, 1, (size_t) sz, f) != (size_t) sz) {
        fclose(f);
        return -2;
    }
    fclose(f);
    qsort(st->entries, st->n_entries, sizeof(QfEntry), cmp_entry);

    st->shard_fds = (int *) malloc((size_t) st->n_shards * sizeof(int));
    st->shard_maps = (void **) calloc(st->n_shards, sizeof(void *));
    st->shard_sizes = (size_t *) calloc(st->n_shards, sizeof(size_t));
    for (int i = 0; i < st->n_shards; ++i) st->shard_fds[i] = -1;
    for (int i = 0; i < st->n_shards; ++i) {
        st->shard_fds[i] = open(st->shard_paths[i], O_RDONLY);
        if (st->shard_fds[i] < 0) return -3;
        struct stat sb {};
        if (fstat(st->shard_fds[i], &sb) != 0) return -3;
        st->shard_sizes[i] = (size_t) sb.st_size;
        st->shard_maps[i] = mmap(nullptr, st->shard_sizes[i], PROT_READ, MAP_PRIVATE, st->shard_fds[i], 0);
        if (st->shard_maps[i] == MAP_FAILED) return -4;
    }
    return 0;
}

const QfEntry *qf_find(const QfStore *st, const char *name) {
    int lo = 0, hi = st->n_entries - 1;
    while (lo <= hi) {
        const int mid = (lo + hi) >> 1;
        const int c = strcmp(st->entries[mid].name, name);
        if (c == 0) return &st->entries[mid];
        if (c < 0) lo = mid + 1; else hi = mid - 1;
    }
    return nullptr;
}

void qf_store_close(QfStore *st) {
    if (!st) return;
    for (int i = 0; i < st->n_shards; ++i) {
        if (st->shard_maps && st->shard_maps[i] && st->shard_maps[i] != MAP_FAILED)
            munmap(st->shard_maps[i], st->shard_sizes[i]);
        if (st->shard_fds && st->shard_fds[i] >= 0) close(st->shard_fds[i]);
    }
    free(st->shard_paths);
    free(st->shard_fds);
    free(st->shard_maps);
    free(st->shard_sizes);
    free(st->entries);
    memset(st, 0, sizeof(*st));
}
