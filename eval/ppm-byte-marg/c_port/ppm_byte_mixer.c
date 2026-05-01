/* PPM-D byte mixer — C implementation of proper_ppm_mixer_rigorous.py's
 * `ppm_mixer_with_norm_modes` function.
 *
 * Inputs (per token, length N):
 *   target_ids[N]       int32   realized target token id
 *   prev_ids[N]         int32   realized previous token id
 *   nll_nats[N]         float   NN's NLL of realized token (nats)
 *   p_full_b0[N]        float   NN P(byte_0=realized_b0) for prev_is_boundary case
 *   p_non_ls_b0[N]      float   NN P(byte_0=realized_b0) for prev_not_bdy, b0!=SPACE case
 *   p_leading_sp[N]     float   NN P(byte_0=SPACE) for prev_not_bdy case
 *   sum_check_full[N]   float   Σ_b byte_dist[b]                (Case A renorm denom)
 *   sum_check_caseB[N]  float   p_ls + Σ_b non_ls_dist[b]       (Case B renorm denom)
 *
 * Tokenizer LUTs (precomputed by Python from sentencepiece):
 *   token_bytes_concat[]      uint8   all token UTF-8 byte expansions concatenated
 *   token_bytes_offset[V+1]   int32   offset into token_bytes_concat for token i
 *   token_has_leading_space[V] uint8  1 if piece starts with ▁
 *   token_is_boundary[V]      uint8   1 if token is BOS/control/unknown/unused
 *
 * Output:
 *   Result struct (see header).
 *
 * Build:
 *   cc -O3 -fPIC -shared -Wall -o libppm_byte_mixer.so ppm_byte_mixer.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#define MAX_ORDER 5
#define HASH_SLOTS (1u << 24)        /* 16 M slots, ~256 MB */
#define HASH_MASK  (HASH_SLOTS - 1u)
#define EMPTY_KEY  0xFFFFFFFFFFFFFFFFULL

/* Pack context bytes (window suffix of length K, K=0..5) into a 64-bit key.
 * Layout: bits [56..58] = K, bits [0..K*8) = bytes (LSB = oldest of the K).
 * K=0 uses key = 0 (special-cased).
 */
static inline uint64_t pack_ctx(const uint8_t *win, int win_len, int K) {
    if (K == 0) return 1ULL << 60;   /* unique marker for empty context */
    uint64_t key = (uint64_t)K << 56;
    for (int i = 0; i < K; i++) {
        key |= (uint64_t)win[win_len - K + i] << (i * 8);
    }
    return key;
}

/* Per-context entry: sparse map byte->count, stored as parallel arrays. */
typedef struct {
    uint64_t key;             /* context key, EMPTY_KEY if free */
    uint32_t total;           /* sum of counts */
    uint16_t unique;          /* number of distinct bytes seen */
    uint16_t cap;             /* allocated capacity for bytes/counts */
    uint8_t  *bytes;          /* dynamic array of seen bytes */
    uint32_t *counts;         /* parallel counts */
} ctx_entry_t;

static ctx_entry_t *g_table = NULL;

static void table_init(void) {
    g_table = (ctx_entry_t*)calloc(HASH_SLOTS, sizeof(ctx_entry_t));
    if (!g_table) { fprintf(stderr, "ppm_byte_mixer: OOM init\n"); abort(); }
    for (uint32_t i = 0; i < HASH_SLOTS; i++) g_table[i].key = EMPTY_KEY;
}

static void table_free(void) {
    if (!g_table) return;
    for (uint32_t i = 0; i < HASH_SLOTS; i++) {
        if (g_table[i].key != EMPTY_KEY) {
            free(g_table[i].bytes); free(g_table[i].counts);
        }
    }
    free(g_table); g_table = NULL;
}

/* MurmurHash3-style finalizer to mix bits. */
static inline uint32_t hash64(uint64_t k) {
    k ^= k >> 33; k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33; k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return (uint32_t)k & HASH_MASK;
}

/* Lookup or insert. Returns slot index. */
static inline uint32_t table_get_or_insert(uint64_t key) {
    uint32_t slot = hash64(key);
    while (1) {
        if (g_table[slot].key == key) return slot;
        if (g_table[slot].key == EMPTY_KEY) {
            g_table[slot].key = key;
            g_table[slot].total = 0;
            g_table[slot].unique = 0;
            g_table[slot].cap = 4;
            g_table[slot].bytes = (uint8_t*)malloc(4);
            g_table[slot].counts = (uint32_t*)malloc(4 * sizeof(uint32_t));
            return slot;
        }
        slot = (slot + 1) & HASH_MASK;
    }
}

/* Lookup only. Returns slot index or 0xFFFFFFFFu if not found. */
static inline uint32_t table_lookup(uint64_t key) {
    uint32_t slot = hash64(key);
    while (1) {
        if (g_table[slot].key == key) return slot;
        if (g_table[slot].key == EMPTY_KEY) return 0xFFFFFFFFu;
        slot = (slot + 1) & HASH_MASK;
    }
}

/* Increment count for byte `b` in context at `slot`. */
static inline void ctx_increment(uint32_t slot, uint8_t b) {
    ctx_entry_t *e = &g_table[slot];
    /* search */
    for (uint16_t i = 0; i < e->unique; i++) {
        if (e->bytes[i] == b) { e->counts[i]++; e->total++; return; }
    }
    /* new byte */
    if (e->unique == e->cap) {
        e->cap *= 2;
        e->bytes  = (uint8_t*)realloc(e->bytes, e->cap);
        e->counts = (uint32_t*)realloc(e->counts, e->cap * sizeof(uint32_t));
    }
    e->bytes[e->unique]  = b;
    e->counts[e->unique] = 1;
    e->unique++;
    e->total++;
}

/* PPM-D scoring: returns log P_PPM(b | window) and writes confidence to *conf_out.
 * Uses Method-D escape: at each order, escape mass = unique/(total+unique),
 * remaining mass distributed as count[b]/(total+unique).
 */
static double ppm_score(uint8_t b, const uint8_t *win, int win_len, double *conf_out) {
    static const double LN2 = 0.6931471805599453;
    (void)LN2;
    double escape_log = 0.0;
    double conf = 0.0;
    int seen_any = 0;
    int order_max = win_len < MAX_ORDER ? win_len : MAX_ORDER;
    for (int K = order_max; K >= 0; K--) {
        uint64_t key = pack_ctx(win, win_len, K);
        uint32_t slot = table_lookup(key);
        if (slot == 0xFFFFFFFFu) continue;
        ctx_entry_t *e = &g_table[slot];
        double denom = (double)e->total + (double)e->unique;
        if (!seen_any) {
            uint32_t maxc = 0;
            for (uint16_t i = 0; i < e->unique; i++) if (e->counts[i] > maxc) maxc = e->counts[i];
            conf = (double)maxc / denom;
            seen_any = 1;
        }
        /* check if b in this ctx */
        for (uint16_t i = 0; i < e->unique; i++) {
            if (e->bytes[i] == b) {
                *conf_out = conf;
                return escape_log + log((double)e->counts[i] / denom);
            }
        }
        /* escape */
        escape_log += log((double)e->unique / denom);
    }
    *conf_out = conf;
    return escape_log + log(1.0 / 256.0);
}

/* Confidence-only lookup (no scoring). */
static double ppm_confidence(const uint8_t *win, int win_len) {
    int order_max = win_len < MAX_ORDER ? win_len : MAX_ORDER;
    for (int K = order_max; K >= 0; K--) {
        uint64_t key = pack_ctx(win, win_len, K);
        uint32_t slot = table_lookup(key);
        if (slot == 0xFFFFFFFFu) continue;
        ctx_entry_t *e = &g_table[slot];
        double denom = (double)e->total + (double)e->unique;
        uint32_t maxc = 0;
        for (uint16_t i = 0; i < e->unique; i++) if (e->counts[i] > maxc) maxc = e->counts[i];
        return (double)maxc / denom;
    }
    return 0.0;
}

/* Update PPM state with byte b at every order 0..min(MAX_ORDER, win_len). */
static void ppm_update(uint8_t b, const uint8_t *win, int win_len) {
    int order_max = win_len < MAX_ORDER ? win_len : MAX_ORDER;
    for (int K = 0; K <= order_max; K++) {
        uint64_t key = pack_ctx(win, win_len, K);
        uint32_t slot = table_get_or_insert(key);
        ctx_increment(slot, b);
    }
}

/* Sigmoid (numerically safe). */
static inline double safe_sigmoid(double z) {
    if (z >= 0) return 1.0 / (1.0 + exp(-z));
    double e = exp(z);
    return e / (1.0 + e);
}

/* Result struct. */
typedef struct {
    double total_mix_nll_nats;
    int64_t total_canonical_bytes;
    int64_t total_scored_tokens;
    double  total_renorm_correction_nats;
    /* Decomposition tracking: NN, PPM, mix, oracle at byte_0 and remainder. */
    double  total_nn_b0_nll;
    double  total_ppm_b0_nll;
    double  total_mix_b0_nll;
    double  total_oracle_b0_nll;
    double  total_nn_rem_nll;
    double  total_ppm_rem_nll;
    double  total_mix_rem_nll;
    double  total_oracle_rem_nll;
    int64_t total_b0_positions;       /* = number of byte-content tokens */
    int64_t total_rem_bytes;          /* sum of (k-1) over multi-byte tokens */
    int64_t total_multibyte_tokens;
} mixer_result_t;

/* Main entry point.
 *
 * `renormalize` = 0 → loose, 1 → rigorous.
 * `alpha`, `beta` are mixer hyperparams (default 15.0, 0.80).
 *
 * The token byte expansion at position i is:
 *   bytes_i = ([0x20] if include_space else []) ++ token_bytes_concat[off..off+nb]
 *   include_space = token_has_leading_space[target] AND NOT prev_is_boundary
 *   prev_is_boundary = (prev < 0) OR token_is_boundary[prev]
 */
void ppm_mixer_run(
    const int32_t *target_ids, const int32_t *prev_ids, const float *nll_nats,
    const float *p_full_b0, const float *p_non_ls_b0, const float *p_leading_sp,
    const float *sum_check_full, const float *sum_check_caseB,
    int64_t N,
    /* tokenizer LUTs */
    const uint8_t *token_bytes_concat,
    const int32_t *token_bytes_offset,
    int32_t V,
    const uint8_t *token_has_leading_space,
    const uint8_t *token_is_boundary,
    /* hyperparams */
    double alpha, double beta, int renormalize,
    /* output */
    mixer_result_t *out
) {
    table_init();

    uint8_t window[MAX_ORDER];
    int win_len = 0;

    double total_nll = 0.0;
    double total_renorm = 0.0;
    int64_t total_bytes = 0;
    int64_t total_tokens = 0;
    /* Decomposition accumulators. */
    double total_nn_b0_nll = 0.0, total_ppm_b0_nll = 0.0;
    double total_mix_b0_nll_acc = 0.0, total_oracle_b0_nll = 0.0;
    double total_nn_rem_nll_acc = 0.0, total_ppm_rem_nll = 0.0;
    double total_mix_rem_nll_acc = 0.0, total_oracle_rem_nll = 0.0;
    int64_t total_b0_positions = 0;
    int64_t total_rem_bytes = 0;
    int64_t total_multibyte = 0;

    for (int64_t i = 0; i < N; i++) {
        int32_t tid = target_ids[i];
        int32_t pid = prev_ids[i];
        int has_space = (tid >= 0 && tid < V) ? token_has_leading_space[tid] : 0;
        int prev_is_bdy = (pid < 0) || (pid < V && token_is_boundary[pid]);
        int include_space = has_space && !prev_is_bdy;

        int32_t off  = token_bytes_offset[tid];
        int32_t off1 = token_bytes_offset[tid + 1];
        int32_t nb   = off1 - off;
        int total_bytes_tok = nb + (include_space ? 1 : 0);

        if (total_bytes_tok == 0) {
            /* control token: full NLL, no decomposition */
            total_nll += (double)nll_nats[i];
            total_tokens++;
            continue;
        }

        /* Build the realized byte sequence for this token (max 1 + nb bytes; small). */
        uint8_t toklen = (uint8_t)total_bytes_tok;
        uint8_t b0;
        if (include_space) b0 = 0x20;
        else b0 = token_bytes_concat[off];

        /* PPM score on byte_0 (BEFORE update). */
        double conf_b0;
        double ppm_logp_b0 = ppm_score(b0, window, win_len, &conf_b0);

        /* NN's raw P(byte_0 = b0 | history, prev). */
        double nn_p_raw, sc;
        if (prev_is_bdy) {
            nn_p_raw = (double)p_full_b0[i];
            sc       = (double)sum_check_full[i];
        } else if (include_space) {
            nn_p_raw = (double)p_leading_sp[i];
            sc       = (double)sum_check_caseB[i];
        } else {
            nn_p_raw = (double)p_non_ls_b0[i];
            sc       = (double)sum_check_caseB[i];
        }
        if (nn_p_raw < 1e-30) nn_p_raw = 1e-30;
        if (nn_p_raw > 1.0)   nn_p_raw = 1.0;
        if (sc < 1e-30)       sc = 1e-30;
        if (sc > 1.0)         sc = 1.0;

        double nn_p_use = nn_p_raw;
        double renorm_corr = 0.0;
        if (renormalize) {
            nn_p_use = nn_p_raw / sc;
            if (nn_p_use > 1.0) nn_p_use = 1.0;
            renorm_corr = -log(sc);
            total_renorm += renorm_corr;
        }

        double ppm_p_b0 = exp(ppm_logp_b0);
        if (ppm_p_b0 > 1.0) ppm_p_b0 = 1.0;
        if (ppm_p_b0 < 0.0) ppm_p_b0 = 0.0;
        double lam0 = 1.0 - safe_sigmoid(alpha * (conf_b0 - beta));
        double p_mix_b0 = lam0 * nn_p_use + (1.0 - lam0) * ppm_p_b0;
        if (p_mix_b0 < 1e-30) p_mix_b0 = 1e-30;
        double nll_b0 = -log(p_mix_b0);

        /* Decomposition tracking at byte_0. */
        double nn_b0_nll = -log(nn_p_use);
        double ppm_b0_nll = -ppm_logp_b0;
        double oracle_b0_nll = nn_b0_nll < ppm_b0_nll ? nn_b0_nll : ppm_b0_nll;
        total_nn_b0_nll += nn_b0_nll;
        total_ppm_b0_nll += ppm_b0_nll;
        total_mix_b0_nll_acc += nll_b0;
        total_oracle_b0_nll += oracle_b0_nll;
        total_b0_positions++;

        /* Update PPM state with realized b0 (score-before-update). */
        ppm_update(b0, window, win_len);
        if (win_len < MAX_ORDER) window[win_len++] = b0;
        else {
            memmove(window, window + 1, MAX_ORDER - 1);
            window[MAX_ORDER - 1] = b0;
        }

        /* Remainder: chain rule on the FULL token's joint probability.
         * For multi-byte: actual byte-by-byte remainder (mixed with PPM).
         * For single-byte: empty byte sequence, but the conditional
         *   P(this 1-byte token | b0) = P(token)/P(b0) is NOT 1 (other tokens
         *   share the same b0). PPM has nothing to mix at the byte level, so
         *   mix at remainder = NN at remainder for single-byte tokens.
         * In BOTH cases, total_nll += nll_b0 + nll_rem satisfies chain rule:
         *   nll_b0 + nll_rem = -log(P_mix_b0 * P_mix_rem)
         */
        double nll_rem = 0.0;
        double nn_token_p = exp(-(double)nll_nats[i]);
        double nn_rem_p = nn_token_p / nn_p_raw;
        if (nn_rem_p < 1e-30) nn_rem_p = 1e-30;
        if (nn_rem_p > 1.0)   nn_rem_p = 1.0;
        double nn_rem_nll = -log(nn_rem_p);
        double ppm_rem_nll = 0.0;  /* PPM contributes 0 NLL for empty rem */
        double oracle_rem_nll = nn_rem_nll;  /* default: PPM = NN for empty rem */

        if (toklen > 1) {
            double ppm_rem_log = 0.0;
            double conf_sum = 0.0;
            int conf_n = 0;
            for (int j = 1; j < toklen; j++) {
                uint8_t bj;
                if (include_space) {
                    /* j=1 corresponds to token_bytes_concat[off+0], etc. */
                    bj = token_bytes_concat[off + j - 1];
                } else {
                    bj = token_bytes_concat[off + j];
                }
                double cf;
                ppm_rem_log += ppm_score(bj, window, win_len, &cf);
                conf_sum += cf; conf_n++;
                ppm_update(bj, window, win_len);
                if (win_len < MAX_ORDER) window[win_len++] = bj;
                else { memmove(window, window + 1, MAX_ORDER - 1); window[MAX_ORDER - 1] = bj; }
            }
            double ppm_rem_p = (ppm_rem_log > -700.0) ? exp(ppm_rem_log) : 0.0;
            double mean_conf = conf_n > 0 ? conf_sum / (double)conf_n : 0.0;
            double lam_r = 1.0 - safe_sigmoid(alpha * (mean_conf - beta));
            double p_mix_r = lam_r * nn_rem_p + (1.0 - lam_r) * ppm_rem_p;
            if (p_mix_r < 1e-30) p_mix_r = 1e-30;
            nll_rem = -log(p_mix_r);
            ppm_rem_nll = -ppm_rem_log;
            oracle_rem_nll = nn_rem_nll < ppm_rem_nll ? nn_rem_nll : ppm_rem_nll;
            total_rem_bytes += (toklen - 1);
            total_multibyte++;
        } else {
            /* single-byte: no PPM bytes to mix; mix_rem = NN_rem */
            nll_rem = nn_rem_nll;
        }
        /* Accumulate decomposition (for both single-byte and multi-byte). */
        total_nn_rem_nll_acc += nn_rem_nll;
        total_ppm_rem_nll += ppm_rem_nll;
        total_mix_rem_nll_acc += nll_rem;
        total_oracle_rem_nll += oracle_rem_nll;

        total_nll += nll_b0 + nll_rem + renorm_corr;
        total_bytes += toklen;
        total_tokens++;
    }

    out->total_mix_nll_nats = total_nll;
    out->total_canonical_bytes = total_bytes;
    out->total_scored_tokens = total_tokens;
    out->total_renorm_correction_nats = total_renorm;
    out->total_nn_b0_nll = total_nn_b0_nll;
    out->total_ppm_b0_nll = total_ppm_b0_nll;
    out->total_mix_b0_nll = total_mix_b0_nll_acc;
    out->total_oracle_b0_nll = total_oracle_b0_nll;
    out->total_nn_rem_nll = total_nn_rem_nll_acc;
    out->total_ppm_rem_nll = total_ppm_rem_nll;
    out->total_mix_rem_nll = total_mix_rem_nll_acc;
    out->total_oracle_rem_nll = total_oracle_rem_nll;
    out->total_b0_positions = total_b0_positions;
    out->total_rem_bytes = total_rem_bytes;
    out->total_multibyte_tokens = total_multibyte;

    table_free();
}

/* Self-test: run on a tiny synthetic input. */
#ifdef PPM_BYTE_MIXER_MAIN
int main(void) {
    /* Minimal sanity: feed 3 tokens, all with 1-byte expansions, prev_not_bdy. */
    int32_t target_ids[3] = {2, 2, 2};
    int32_t prev_ids[3]   = {2, 2, 2};
    float nll_nats[3]     = {1.0f, 1.0f, 1.0f};
    float p_full[3]       = {0.5f, 0.5f, 0.5f};
    float p_nonls[3]      = {0.5f, 0.5f, 0.5f};
    float p_ls[3]         = {0.0f, 0.0f, 0.0f};
    float sc_full[3]      = {0.99f, 0.99f, 0.99f};
    float sc_caseB[3]     = {0.999f, 0.999f, 0.999f};
    /* token V=3: 0=control(0 bytes), 1=control, 2='a' */
    uint8_t bytes[1] = {'a'};
    int32_t off[4] = {0, 0, 0, 1};
    uint8_t hls[3] = {0, 0, 0};
    uint8_t bdy[3] = {1, 1, 0};
    mixer_result_t res;
    ppm_mixer_run(target_ids, prev_ids, nll_nats, p_full, p_nonls, p_ls,
                  sc_full, sc_caseB, 3,
                  bytes, off, 3, hls, bdy,
                  15.0, 0.80, 0, &res);
    printf("LOOSE   nll=%.4f bytes=%lld tok=%lld bpb=%.4f\n",
           res.total_mix_nll_nats, (long long)res.total_canonical_bytes,
           (long long)res.total_scored_tokens,
           res.total_mix_nll_nats / (double)res.total_canonical_bytes / log(2.0));
    ppm_mixer_run(target_ids, prev_ids, nll_nats, p_full, p_nonls, p_ls,
                  sc_full, sc_caseB, 3,
                  bytes, off, 3, hls, bdy,
                  15.0, 0.80, 1, &res);
    printf("RIGOROUS nll=%.4f renorm=%.4f bpb=%.4f\n",
           res.total_mix_nll_nats, res.total_renorm_correction_nats,
           res.total_mix_nll_nats / (double)res.total_canonical_bytes / log(2.0));
    return 0;
}
#endif
