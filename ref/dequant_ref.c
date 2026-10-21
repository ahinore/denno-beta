/* dequant_ref.c — denno-beta kquant.rs の検証用リファレンス。
 * ggml-quants.c の dequantize_row_* を、決定論的 LCG で作った疑似ランダム
 * ブロック列に適用し、(生バイト, f32 出力) をバイナリにダンプする。
 *
 * 出力形式 (LE):
 *   u32 n_sections
 *   各 section:
 *     u32 name_len, name bytes (ASCII, NUL 無し)
 *     u32 n_bytes, raw bytes
 *     u32 n_f32, f32 LE * n_f32
 *
 * ビルド (実測で通るコマンド。llama.cpp 側は -O2 のままだと
 * dequantize_row_tq1_0 / iq1_m の演算が変わらないことを確認済み):
 *   gcc -O2 -I<llama.cpp>/ggml/include -I<llama.cpp>/ggml/src \
 *       -o dequant_ref ref/dequant_ref.c <llama.cpp>/ggml/src/ggml-quants.c -lm
 *
 * 詳細と再生成手順は ref/README.md を参照。
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "ggml.h"
#include "ggml-quants.h"

static uint32_t lcg_state = 0x12345678u;
static uint8_t lcg_byte(void) {
    lcg_state = lcg_state * 1103515245u + 12345u;
    return (uint8_t)((lcg_state >> 16) & 0xff);
}

/* --- ggml.c からの最小スタブ (リンク用。dequant 経路では未使用) --- */
void ggml_abort(const char *file, int line, const char *fmt, ...) {
    fprintf(stderr, "ABORT at %s:%d\n", file, line);
    exit(1);
}
const char *ggml_type_name(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32: return "f32";
        case GGML_TYPE_F16: return "f16";
        case GGML_TYPE_BF16: return "bf16";
        case GGML_TYPE_Q4_0: return "q4_0";
        case GGML_TYPE_Q4_1: return "q4_1";
        case GGML_TYPE_Q5_0: return "q5_0";
        case GGML_TYPE_Q5_1: return "q5_1";
        case GGML_TYPE_Q8_0: return "q8_0";
        case GGML_TYPE_Q2_K: return "q2_K";
        case GGML_TYPE_Q3_K: return "q3_K";
        case GGML_TYPE_Q4_K: return "q4_K";
        case GGML_TYPE_Q5_K: return "q5_K";
        case GGML_TYPE_Q6_K: return "q6_K";
        case GGML_TYPE_IQ2_XXS: return "iq2_xxs";
        case GGML_TYPE_IQ2_XS: return "iq2_xs";
        case GGML_TYPE_IQ2_S: return "iq2_s";
        case GGML_TYPE_IQ3_XXS: return "iq3_xxs";
        case GGML_TYPE_IQ3_S: return "iq3_s";
        case GGML_TYPE_IQ1_S: return "iq1_s";
        case GGML_TYPE_IQ4_NL: return "iq4_nl";
        case GGML_TYPE_IQ4_XS: return "iq4_xs";
        case GGML_TYPE_TQ1_0: return "tq1_0";
        case GGML_TYPE_IQ1_M: return "iq1_m";
        default: return "?";
    }
}
static size_t ggml_type_size_impl(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32: return 4;
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16: return 2;
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_IQ4_NL: return 18;
        case GGML_TYPE_Q2_K: return 84;
        case GGML_TYPE_Q3_K: return 110;
        case GGML_TYPE_Q4_K: return 144;
        case GGML_TYPE_Q5_K: return 176;
        case GGML_TYPE_Q6_K: return 210;
        case GGML_TYPE_IQ2_XXS: return 66;
        case GGML_TYPE_IQ2_XS: return 74;
        case GGML_TYPE_IQ2_S: return 82;
        case GGML_TYPE_IQ3_XXS: return 98;
        case GGML_TYPE_IQ3_S: return 110;
        case GGML_TYPE_IQ1_S: return 50;
        case GGML_TYPE_IQ4_XS: return 136;
        case GGML_TYPE_TQ1_0: return 54;  /* qs[48] + qh[4] + d(f16) */
        case GGML_TYPE_IQ1_M: return 56;  /* qs[32] + qh[16] + scales[8] */
        default: return 0;
    }
}
size_t ggml_type_size(enum ggml_type type) {
    return ggml_type_size_impl(type);
}
int64_t ggml_blck_size(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16: return 1;
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_IQ4_NL: return 32;
        default: return 256; /* K/IQ 系 */
    }
}
size_t ggml_row_size(enum ggml_type type, int64_t n) {
    size_t blck = (size_t)ggml_blck_size(type);
    size_t ts = ggml_type_size_impl(type);
    return (n / blck + (n % blck != 0)) * ts;
}

static void wr(FILE *f, const void *p, size_t n) {
    if (fwrite(p, 1, n, f) != n) { perror("fwrite"); exit(1); }
}
static void wr_u32(FILE *f, uint32_t v) { wr(f, &v, 4); }

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s out.bin\n", argv[0]); return 1; }
    FILE *f = fopen(argv[1], "wb");
    if (!f) { perror("fopen"); return 1; }

    const int N_BLOCKS = 16; /* 各型 16 ブロック (K 系なら 4096 要素) */
    struct { const char *name; size_t type_size; int blck;
             void (*fn)(const void *, float *, int64_t); } secs[] = {
        { "Q8_0",    sizeof(block_q8_0),    QK8_0,    (void *)dequantize_row_q8_0 },
        { "Q4_0",    sizeof(block_q4_0),    QK4_0,    (void *)dequantize_row_q4_0 },
        { "Q4_1",    sizeof(block_q4_1),    QK4_1,    (void *)dequantize_row_q4_1 },
        { "Q5_0",    sizeof(block_q5_0),    QK5_0,    (void *)dequantize_row_q5_0 },
        { "Q5_1",    sizeof(block_q5_1),    QK5_1,    (void *)dequantize_row_q5_1 },
        { "Q2_K",    sizeof(block_q2_K),    QK_K,     (void *)dequantize_row_q2_K },
        { "Q3_K",    sizeof(block_q3_K),    QK_K,     (void *)dequantize_row_q3_K },
        { "Q4_K",    sizeof(block_q4_K),    QK_K,     (void *)dequantize_row_q4_K },
        { "Q5_K",    sizeof(block_q5_K),    QK_K,     (void *)dequantize_row_q5_K },
        { "Q6_K",    sizeof(block_q6_K),    QK_K,     (void *)dequantize_row_q6_K },
        { "IQ2_XXS", sizeof(block_iq2_xxs), QK_K,     (void *)dequantize_row_iq2_xxs },
        { "IQ2_XS",  sizeof(block_iq2_xs),  QK_K,     (void *)dequantize_row_iq2_xs },
        { "IQ2_S",   sizeof(block_iq2_s),   QK_K,     (void *)dequantize_row_iq2_s },
        { "IQ3_XXS", sizeof(block_iq3_xxs), QK_K,     (void *)dequantize_row_iq3_xxs },
        { "IQ3_S",   sizeof(block_iq3_s),   QK_K,     (void *)dequantize_row_iq3_s },
        { "IQ1_S",   sizeof(block_iq1_s),   QK_K,     (void *)dequantize_row_iq1_s },
        { "IQ4_NL",  sizeof(block_iq4_nl),  QK4_NL,   (void *)dequantize_row_iq4_nl },
        { "IQ4_XS",  sizeof(block_iq4_xs),  QK_K,     (void *)dequantize_row_iq4_xs },
        /* Task K2: K1 で追加した 2 型。llama.cpp の dequantize_row_* をそのまま使う
         * (TQ1_0 は pow3 展開、IQ1_M は iq1s_grid 参照。iq1s_grid は ggml-quants.c が
         *  GGML_COMMON_IMPL_C で include する ggml-common.h の static const テーブル)。
         * 既存 18 型の後ろに追記する (先頭セクションのバイト列を変えないため)。 */
        { "TQ1_0",   sizeof(block_tq1_0),   QK_K,     (void *)dequantize_row_tq1_0 },
        { "IQ1_M",   sizeof(block_iq1_m),   QK_K,     (void *)dequantize_row_iq1_m },
    };
    const int n_secs = (int)(sizeof(secs) / sizeof(secs[0]));
    wr_u32(f, (uint32_t)n_secs);

    /* Q8_0 だけは qs が int8 なので専用生成 (他の型は符号ビットも使う) */
    for (int s = 0; s < n_secs; s++) {
        size_t ts = secs[s].type_size;
        int blck = secs[s].blck;
        size_t nb = (size_t)N_BLOCKS * ts;
        uint8_t *raw = malloc(nb);
        float *y = malloc(sizeof(float) * N_BLOCKS * blck);
        if (!raw || !y) { perror("malloc"); return 1; }

        for (size_t i = 0; i < nb; i++) raw[i] = lcg_byte();

        secs[s].fn(raw, y, (int64_t)N_BLOCKS * blck);

        uint32_t nl = (uint32_t)strlen(secs[s].name);
        wr_u32(f, nl); wr(f, secs[s].name, nl);
        wr_u32(f, (uint32_t)nb); wr(f, raw, nb);
        wr_u32(f, (uint32_t)(N_BLOCKS * blck));
        wr(f, y, sizeof(float) * N_BLOCKS * blck);
        free(raw); free(y);
        fprintf(stderr, "%s: ok (%zu bytes, %d elems)\n", secs[s].name, nb, N_BLOCKS * blck);
    }
    fclose(f);
    return 0;
}
