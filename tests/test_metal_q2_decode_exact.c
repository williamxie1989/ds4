#define _DARWIN_C_SOURCE
#include "ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); goto done; \
} } while (0)

enum { INPUT = 4096, MID = 2048, OUTPUT = 4096, MAX_OUTPUT = 4097,
       EXPERTS = 8, SELECTED = 6, OUTPUTS = 5, GUARD = 16, ROUTES = 4 };
/* Same packed layouts as test_metal_moe_prefill.c; all payload bit patterns
 * are valid grid/sign/scale codes when the half scales are finite. */
typedef struct { uint16_t d; uint8_t qs[64]; } iq2_block;
typedef struct { uint8_t scales[16], qs[64]; uint16_t d, dmin; } q2_block;

typedef struct {
    void *model;
    uint64_t model_bytes, up_off, down_off;
    ds4_gpu_tensor *x, *ids, *weights, *addend, *result[OUTPUTS];
} fixture;

static uint32_t random_u32(uint32_t *state) {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    return *state;
}

static int run_moe(const fixture *f, uint32_t width, bool addend) {
    const uint64_t gate_row = INPUT / 256 * sizeof(iq2_block);
    const uint64_t down_row = MID / 256 * sizeof(q2_block);
    return ds4_gpu_routed_moe_one_tensor(
        f->result[4], f->result[0], f->result[1], f->result[2], f->result[3],
        f->model, f->model_bytes, 0, f->up_off, f->down_off, 16, 10,
        MID * gate_row, gate_row, width * down_row, down_row,
        INPUT, MID, width, f->ids, f->weights, EXPERTS, SELECTED, 7.0f,
        f->x, addend ? f->addend : NULL, 0, true);
}

int main(int argc, char **argv) {
    const bool reference_only = argc == 2 && strcmp(argv[1], "--reference-only") == 0;
    if (argc != 1 && !reference_only) {
        fprintf(stderr, "usage: %s [--reference-only]\n", argv[0]);
        return 2;
    }
    /* Do not silently compare two fallback paths. */
    const char *conflicts[] = {
        "DS4_METAL_DISABLE_ROUTED_PAIR_SWIGLU_FUSION",
        "DS4_METAL_MOE_WRITE_CLAMPED_ACT",
    };
    for (unsigned i = 0; i < sizeof(conflicts) / sizeof(*conflicts); i++) {
        if (getenv(conflicts[i])) {
            fprintf(stderr, "unset %s for this test\n", conflicts[i]);
            return 2;
        }
    }
    const char *disable = "DS4_METAL_DISABLE_M5_Q2_SUM6_TUNING";
    const char *saved_env = getenv(disable);
    char *saved = saved_env ? strdup(saved_env) : NULL;
    if (saved_env && !saved) return 1;

    fixture f = {0};
    ds4_gpu_tensor *slabs[OUTPUTS] = {0};
    float *reference[OUTPUTS] = {0}, *actual = NULL;
    const char *names[OUTPUTS] = {"gate-scratch", "up-scratch", "mid", "experts-unused", "out"};
    const uint32_t capacity[OUTPUTS] = {
        SELECTED * MID, SELECTED * MID, SELECTED * MID,
        SELECTED * MAX_OUTPUT, MAX_OUTPUT,
    };
    const uint32_t widths[] = {OUTPUT, OUTPUT - 1, OUTPUT + 1, OUTPUT};
    const int32_t ids[ROUTES][SELECTED] = {
        {7, 7, 7, 7, 7, 7}, {7, 0, 7, 0, 3, 3},
        {0, 7, 1, 6, 2, 5}, {0, 1, 2, 3, 6, 7},
    };
    const float weights[ROUTES][SELECTED] = {
        {-0.75f, 0.5f, 0.0f, -0.125f, 0.25f, 0.0625f},
        {-0.5f, 0.0f, 0.25f, -0.0f, 1.0f, -1.25f},
        {0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f},
        {0.125f, 0.25f, 0.5f, 0.75f, 1.0f, 1.5f},
    };
    float input[INPUT], addend[MAX_OUTPUT];
    const uint32_t sentinel = 0x7fc12345u;
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t gate_bytes = (uint64_t)EXPERTS * MID * INPUT / 256 * sizeof(iq2_block);
    const uint64_t down_blocks = (uint64_t)EXPERTS * MAX_OUTPUT * MID / 256;
    uint64_t checksum = UINT64_C(14695981039346656037);
    int rc = 1;

    /* Never request the new shader when loading an original MoE source. */
    if (reference_only) CHECK(setenv(disable, "1", 1) == 0);
    CHECK(sizeof(iq2_block) == 66 && sizeof(q2_block) == 84);
    CHECK(ds4_gpu_init());
    if (!ds4_gpu_device_is_m5_apple_silicon()) {
        puts("Q2 decode tuning: SKIP (requires M5 for distinct A/B paths)");
        rc = 0;
        goto done;
    }
    ds4_gpu_set_quality(false);
    ds4_gpu_set_ssd_streaming(false);
    f.up_off = (gate_bytes + page - 1) / page * page;
    f.down_off = 2 * f.up_off;
    f.model_bytes = (f.down_off + down_blocks * sizeof(q2_block) + page - 1) / page * page;
    CHECK(posix_memalign(&f.model, (size_t)page, (size_t)f.model_bytes) == 0);
    memset(f.model, 0, (size_t)f.model_bytes);
    uint32_t state = 7919;
    for (unsigned tensor = 0; tensor < 2; tensor++) {
        iq2_block *b = (iq2_block *)((char *)f.model + tensor * f.up_off);
        for (uint64_t i = 0; i < gate_bytes / sizeof(*b); i++) {
            b[i].d = (uint16_t)(0x1000u + (random_u32(&state) & 0x7ffu));
            for (unsigned j = 0; j < sizeof(b[i].qs); j++) b[i].qs[j] = random_u32(&state);
        }
    }
    q2_block *b = (q2_block *)((char *)f.model + f.down_off);
    for (uint64_t i = 0; i < down_blocks; i++) {
        b[i].d = (uint16_t)(0x1c00u + (random_u32(&state) & 0x7ffu));
        b[i].dmin = (uint16_t)(0x1800u + (random_u32(&state) & 0x7ffu));
        for (unsigned j = 0; j < sizeof(b[i].scales); j++) b[i].scales[j] = random_u32(&state);
        for (unsigned j = 0; j < sizeof(b[i].qs); j++) b[i].qs[j] = random_u32(&state);
    }
    /* Each width uses a densely packed prefix with its own expert stride. */
    CHECK(ds4_gpu_set_model_map(f.model, f.model_bytes));
    f.x = ds4_gpu_tensor_alloc(sizeof(input));
    f.ids = ds4_gpu_tensor_alloc(sizeof(ids[0]));
    f.weights = ds4_gpu_tensor_alloc(sizeof(weights[0]));
    f.addend = ds4_gpu_tensor_alloc(sizeof(addend));
    actual = malloc((SELECTED * MAX_OUTPUT + 2 * GUARD) * sizeof(float));
    CHECK(f.x && f.ids && f.weights && f.addend && actual);
    for (unsigned i = 0; i < OUTPUTS; i++) {
        const uint64_t bytes = (capacity[i] + 2 * GUARD) * sizeof(float);
        slabs[i] = ds4_gpu_tensor_alloc(bytes);
        reference[i] = malloc(bytes);
        CHECK(slabs[i] && reference[i]);
        f.result[i] = ds4_gpu_tensor_view(slabs[i], GUARD * sizeof(float), capacity[i] * sizeof(float));
        CHECK(f.result[i]);
    }
    for (unsigned i = 0; i < MAX_OUTPUT; i++)
        addend[i] = ((int)(random_u32(&state) % 101) - 50) / 256.0f;
    CHECK(ds4_gpu_tensor_write(f.addend, 0, addend, sizeof(addend)));

    for (unsigned c = 0; c < sizeof(widths) / sizeof(*widths); c++) {
        const bool with_addend = c == 3;
        /* gate ne11=1 broadcasts x. The fused IQ2 kernel also uses idx%ne11
         * for gate/up stores: only their first MID entries are shared scratch,
         * with no defined winning expert. Mid instead uses idx and saves all
         * six rows; only mid/out are deterministic observable outputs. */
        const uint32_t count[OUTPUTS] = {MID, MID, SELECTED * MID, 0, widths[c]};
        for (unsigned route = 0; route < ROUTES; route++) {
            state = 65537u + route * 104729u;
            for (unsigned i = 0; i < INPUT; i++)
                input[i] = ldexpf(((int)(random_u32(&state) % 1025) - 512) / 1024.0f, (int)route - 2);
            CHECK(ds4_gpu_tensor_write(f.x, 0, input, sizeof(input)));
            CHECK(ds4_gpu_tensor_write(f.ids, 0, ids[route], sizeof(ids[route])));
            CHECK(ds4_gpu_tensor_write(f.weights, 0, weights[route], sizeof(weights[route])));
            for (unsigned run = 0; run < 4; run++) {
                const bool rollback = reference_only || run == 0 || run == 3;
                const char *rollback_values[] = {"1", "0", ""}; /* Presence, not truthiness. */
                CHECK(rollback ? setenv(disable, rollback_values[(route + run) % 3], 1) == 0
                               : unsetenv(disable) == 0);
                for (unsigned i = 0; i < OUTPUTS; i++) {
                    for (uint32_t j = 0; j < capacity[i] + 2 * GUARD; j++) {
                        const uint32_t bits = j >= GUARD && j < GUARD + count[i] ?
                            0x7fc00001u + run : sentinel;
                        memcpy(&actual[j], &bits, sizeof(bits));
                    }
                    CHECK(ds4_gpu_tensor_write(slabs[i], 0, actual, (capacity[i] + 2 * GUARD) * sizeof(float)));
                }
                CHECK(ds4_gpu_begin_commands());
                CHECK(run_moe(&f, widths[c], with_addend));
                CHECK(ds4_gpu_end_commands());
                for (unsigned i = 0; i < OUTPUTS; i++) {
                    const uint64_t bytes = (capacity[i] + 2 * GUARD) * sizeof(float);
                    CHECK(ds4_gpu_tensor_read(slabs[i], 0, actual, bytes));
                    for (uint32_t j = 0; j < capacity[i] + 2 * GUARD; j++) {
                        uint32_t bits, expected;
                        memcpy(&bits, &actual[j], sizeof(bits));
                        const bool active = j >= GUARD && j < GUARD + count[i];
                        if ((active && !isfinite(actual[j])) || (!active && bits != sentinel)) {
                            fprintf(stderr, "Q2 invalid/guard width=%u addend=%d route=%u run=%u %s[%u]=0x%08x\n",
                                    widths[c], with_addend, route, run, names[i], j, bits);
                            goto done;
                        }
                        if (run && !(i < 2 && active) &&
                            memcmp(&reference[i][j], &actual[j], sizeof(float))) {
                            memcpy(&expected, &reference[i][j], sizeof(expected));
                            fprintf(stderr, "Q2 mismatch width=%u addend=%d route=%u run=%u %s[%u] ref=0x%08x actual=0x%08x\n",
                                    widths[c], with_addend, route, run, names[i], j, expected, bits);
                            goto done;
                        }
                        if (i == 4 && active && route == 2)
                            CHECK(actual[j] == (with_addend ? addend[j - GUARD] : 0.0f));
                        /* FNV-1a, canonical low-to-high bytes of every defined
                         * output word, once per case, independent of run mode. */
                        if (run == 0 && i >= 2 && active) {
                            for (unsigned byte = 0; byte < sizeof(bits); byte++) {
                                checksum ^= (bits >> (8 * byte)) & 0xffu;
                                checksum *= UINT64_C(1099511628211);
                            }
                        }
                    }
                    if (run == 0) memcpy(reference[i], actual, bytes);
                }
            }
        }
        printf("Q2 decode width=%u addend=%d: PASS (4 routes x %s, mid/out bitwise exact, scratch finite, sentinels intact)\n",
               widths[c], with_addend, reference_only ? "AAAA" : "ABBA");
    }
    printf("Q2 defined outputs checksum=%016llx (all mid/out bits, once per case)\n",
           (unsigned long long)checksum);

    rc = 0;
done:
    if (ds4_gpu_commands_active() && !ds4_gpu_end_commands()) rc = 1;
    for (unsigned i = 0; i < OUTPUTS; i++) {
        ds4_gpu_tensor_free(f.result[i]);
        ds4_gpu_tensor_free(slabs[i]);
        free(reference[i]);
    }
    ds4_gpu_tensor_free(f.x);
    ds4_gpu_tensor_free(f.ids);
    ds4_gpu_tensor_free(f.weights);
    ds4_gpu_tensor_free(f.addend);
    ds4_gpu_cleanup();
    free(f.model);
    free(actual);
    if (saved ? setenv(disable, saved, 1) != 0 : unsetenv(disable) != 0) rc = 1;
    free(saved);
    return rc;
}
