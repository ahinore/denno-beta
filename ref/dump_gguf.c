/* dump_gguf.c — llama.cpp 本家 gguf.c でファイルを検査する (正の検証用) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ggml.h"
#include "gguf.h"

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s file.gguf [names]\n", argv[0]); return 1; }
    struct gguf_init_params p = { .no_alloc = true, .ctx = NULL };
    struct gguf_context *g = gguf_init_from_file(argv[1], p);
    if (!g) { fprintf(stderr, "gguf_init_from_file FAILED\n"); return 1; }
    printf("version: %d\n", gguf_get_version(g));
    printf("alignment: %zu\n", gguf_get_alignment(g));
    printf("data_offset: %zu\n", gguf_get_data_offset(g));
    int64_t n = gguf_get_n_tensors(g);
    printf("n_tensors: %lld\n", (long long)n);
    /* 総データ終端 = data_offset + max(offset+size) */
    size_t doff = gguf_get_data_offset(g);
    size_t max_end = 0;
    for (int64_t i = 0; i < n; i++) {
        const char *name = gguf_get_tensor_name(g, i);
        size_t off = gguf_get_tensor_offset(g, i);
        size_t sz = gguf_get_tensor_size(g, i);
        size_t end = doff + off + sz;
        if (end > max_end) max_end = end;
        if (argc > 2) /* 全名前出力モード: "name size type" */
            printf("%s %zu %s\n", name, sz, ggml_type_name(gguf_get_tensor_type(g, i)));
        else if (i < 3 || i >= n - 3)
            printf("  [%lld] %-40s off=%zu size=%zu end=%zu\n", (long long)i, name, off, sz, end);
    }
    FILE *f = fopen(argv[1], "rb");
    fseek(f, 0, SEEK_END);
    long flen = ftell(f);
    fclose(f);
    printf("tensor_data_end: %zu\n", max_end);
    printf("file_size: %ld\n", flen);
    printf("trailing: %ld bytes\n", flen - (long)max_end);
    gguf_free(g);
    return 0;
}
