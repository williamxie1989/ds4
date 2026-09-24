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

enum { INPUT = 4096, MID = 2048, OUTPUT = 4096,
       EXPERTS = 8, FULL_EXPERTS = 256, GLM53_EXPERTS = 288,
       DEFAULT_SELECTED = 6, MAX_SELECTED = 8, OUTPUTS = 5,
       GUARD = 16, ROUTES = 4, PROFILE_DEFAULT = 0, PROFILE_V41, PROFILE_PRO, PROFILE_GLM53 };
/* Same packed layouts as test_metal_moe_prefill.c; all payload bit patterns
 * are valid grid/sign/scale codes when the half scales are finite. */
typedef struct { uint16_t d; uint8_t qs[64]; } iq2_block;
typedef struct { uint8_t scales[16], qs[64]; uint16_t d, dmin; } q2_block;

typedef struct {
    void *model;
    uint64_t model_bytes, up_off, down_off;
    uint32_t experts, selected, input_dim, mid_dim;
    ds4_gpu_tensor *x, *ids, *weights, *addend, *result[OUTPUTS];
} fixture;

static uint32_t random_u32(uint32_t *state) {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    return *state;
}

static int run_moe(const fixture *f, uint32_t width, bool addend) {
    const uint64_t gate_row = f->input_dim / 256 * sizeof(iq2_block);
    const uint64_t down_row = f->mid_dim / 256 * sizeof(q2_block);
    return ds4_gpu_routed_moe_one_tensor(
        f->result[4], f->result[0], f->result[1], f->result[2], f->result[3],
        f->model, f->model_bytes, 0, f->up_off, f->down_off, 16, 10,
        f->mid_dim * gate_row, gate_row, width * down_row, down_row,
        f->input_dim, f->mid_dim, width, f->ids, f->weights, f->experts, f->selected, 7.0f,
        f->x, addend ? f->addend : NULL, 0, true);
}

int main(int argc, char **argv) {
    bool reference_only = false, full_experts = false;
    unsigned profile = PROFILE_DEFAULT;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--reference-only") && !reference_only) reference_only = true;
        else if (!strcmp(argv[i], "--full-experts") && !full_experts) full_experts = true;
        else if (!strcmp(argv[i], "--profile") && profile == PROFILE_DEFAULT && i + 1 < argc &&
                 (!strcmp(argv[i + 1], "v41") || !strcmp(argv[i + 1], "pro") || !strcmp(argv[i + 1], "glm53"))) {
            const char *value = argv[++i];
            profile = !strcmp(value, "v41") ? PROFILE_V41 :
                      !strcmp(value, "pro") ? PROFILE_PRO : PROFILE_GLM53;
        } else {
            fprintf(stderr, "usage: %s [--reference-only] [--full-experts | --profile v41|pro|glm53]\n", argv[0]);
            return 2;
        }
    }
    if (full_experts && profile != PROFILE_DEFAULT) {
        fputs("--profile cannot be combined with --full-experts\n", stderr);
        return 2;
    }
    const bool glm53 = profile == PROFILE_GLM53;
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
    const char *disable[] = {
        "DS4_METAL_DISABLE_M5_IQ2_PACK2_SHAPE",
        "DS4_METAL_DISABLE_M5_Q2_SUM6_TUNING",
        "DS4_METAL_DISABLE_M5_IQ2_PAIR_PACK2",
        "DS4_METAL_DISABLE_M5_IQ2_GLM53_SHAPE",
        "DS4_METAL_DISABLE_M5_GLM53_Q2_DOWN_SHAPE",
    };
    if (full_experts && !reference_only && getenv(disable[2])) {
        fprintf(stderr, "unset %s for --full-experts\n", disable[2]);
        return 2;
    }
    char *saved[sizeof(disable) / sizeof(*disable)] = {0};
    for (unsigned i = 0; i < sizeof(disable) / sizeof(*disable); i++) {
        const char *value = getenv(disable[i]);
        if (value) saved[i] = strdup(value);
        if (value && !saved[i]) {
            for (unsigned j = 0; j < sizeof(saved) / sizeof(*saved); j++) free(saved[j]);
            return 1;
        }
    }

    /* Enum selection keys on these projection dimensions and strides, not expert count. */
    const struct { uint32_t input, mid, output; } dimensions[] = {
        {INPUT, MID, OUTPUT}, {5120, 2304, 5120}, {7168, 3072, 7168}, {INPUT, MID, OUTPUT},
    };
    fixture f = {0};
    f.experts = glm53 ? GLM53_EXPERTS : full_experts ? FULL_EXPERTS : EXPERTS;
    f.selected = glm53 ? MAX_SELECTED : DEFAULT_SELECTED;
    f.input_dim = dimensions[profile].input;
    f.mid_dim = dimensions[profile].mid;
    const uint32_t output_dim = dimensions[profile].output;
    const uint32_t max_output = output_dim + 1;
    ds4_gpu_tensor *slabs[OUTPUTS] = {0};
    float *reference[OUTPUTS] = {0}, *actual = NULL;
    const char *names[OUTPUTS] = {"gate-scratch", "up-scratch", "mid",
                                  glm53 ? "experts-scratch" : "experts-unused", "out"};
    const uint32_t capacity[OUTPUTS] = {
        f.selected * f.mid_dim, f.selected * f.mid_dim, f.selected * f.mid_dim,
        f.selected * max_output, max_output,
    };
    const uint32_t widths[] = {output_dim, output_dim - 1, output_dim + 1, output_dim};
    const int32_t small_ids[ROUTES][MAX_SELECTED] = {
        {7, 7, 7, 7, 7, 7}, {7, 0, 7, 0, 3, 3},
        {0, 7, 1, 6, 2, 5}, {0, 1, 2, 3, 6, 7},
    };
    const int32_t large_ids[ROUTES][MAX_SELECTED] = {
        {255, 255, 255, 255, 255, 255}, {255, 0, 255, 0, 128, 128},
        {0, 255, 1, 254, 63, 127}, {0, 1, 63, 127, 254, 255},
    };
    const int32_t glm53_ids[ROUTES][MAX_SELECTED] = {
        {287, 287, 287, 287, 287, 287, 287, 287},
        {287, 0, 287, 0, 128, 128, 286, 286},
        {0, 287, 1, 286, 63, 127, 191, 255},
        {0, 1, 63, 127, 254, 255, 286, 287},
    };
    const int32_t (*ids)[MAX_SELECTED] = glm53 ? glm53_ids : full_experts ? large_ids : small_ids;
    bool used[GLM53_EXPERTS] = {0};
    for (unsigned route = 0; route < ROUTES; route++)
        for (unsigned i = 0; i < f.selected; i++) used[ids[route][i]] = true;
    const float weights[ROUTES][MAX_SELECTED] = {
        {-0.75f, 0.5f, 0.0f, -0.125f, 0.25f, 0.0625f, -0.375f, 0.875f},
        {-0.5f, 0.0f, 0.25f, -0.0f, 1.0f, -1.25f, 0.125f, -0.625f},
        {0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f},
        {0.125f, 0.25f, 0.5f, 0.75f, 1.0f, 1.5f, -0.375f, 2.0f},
    };
    float input[f.input_dim], addend[max_output];
    const uint32_t sentinel = 0x7fc12345u;
    const uint64_t page = (uint64_t)getpagesize();
    const uint64_t gate_bytes = (uint64_t)f.experts * f.mid_dim * f.input_dim / 256 * sizeof(iq2_block);
    const uint64_t down_blocks = (uint64_t)f.experts * max_output * f.mid_dim / 256;
    uint64_t checksum = UINT64_C(14695981039346656037);
    int rc = 1;

    /* Never request the new shader when loading an original MoE source. */
    if (reference_only) {
        CHECK(setenv(disable[0], "1", 1) == 0);
        CHECK(setenv(disable[1], "1", 1) == 0);
        if (full_experts) CHECK(setenv(disable[2], "1", 1) == 0);
    }
    if (glm53) {
        /* Eight routes cannot use Q2 direct sum. Disable both GLM shapes before init. */
        CHECK(setenv(disable[1], "1", 1) == 0);
    }
    if (reference_only || glm53) {
        CHECK(setenv(disable[3], "1", 1) == 0);
        CHECK(setenv(disable[4], "1", 1) == 0);
    }
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
        for (unsigned e = 0; e < f.experts; e++) {
            if (f.experts != EXPERTS && !used[e]) continue;
            for (uint64_t i = (uint64_t)e * f.mid_dim * f.input_dim / 256;
                 i < (uint64_t)(e + 1) * f.mid_dim * f.input_dim / 256; i++) {
                b[i].d = (uint16_t)(0x1000u + (random_u32(&state) & 0x7ffu));
                for (unsigned j = 0; j < sizeof(b[i].qs); j++) b[i].qs[j] = random_u32(&state);
            }
        }
    }
    q2_block *b = (q2_block *)((char *)f.model + f.down_off);
    for (unsigned e = 0; e < f.experts; e++) {
        if (f.experts != EXPERTS && !used[e]) continue;
        /* Dense prefixes use width-dependent expert strides, including at the highest ID. */
        const uint64_t first = (uint64_t)e * (f.experts != EXPERTS ? output_dim - 1 : max_output) * f.mid_dim / 256;
        const uint64_t end = (uint64_t)(e + 1) * max_output * f.mid_dim / 256;
        for (uint64_t i = first; i < end; i++) {
            b[i].d = (uint16_t)(0x1c00u + (random_u32(&state) & 0x7ffu));
            b[i].dmin = (uint16_t)(0x1800u + (random_u32(&state) & 0x7ffu));
            for (unsigned j = 0; j < sizeof(b[i].scales); j++) b[i].scales[j] = random_u32(&state);
            for (unsigned j = 0; j < sizeof(b[i].qs); j++) b[i].qs[j] = random_u32(&state);
        }
    }
    /* Each width uses a densely packed prefix with its own expert stride. */
    CHECK(ds4_gpu_set_model_map(f.model, f.model_bytes));
    f.x = ds4_gpu_tensor_alloc(sizeof(input));
    f.ids = ds4_gpu_tensor_alloc(f.selected * sizeof(int32_t));
    f.weights = ds4_gpu_tensor_alloc(f.selected * sizeof(float));
    f.addend = ds4_gpu_tensor_alloc(sizeof(addend));
    actual = malloc((f.selected * max_output + 2 * GUARD) * sizeof(float));
    CHECK(f.x && f.ids && f.weights && f.addend && actual);
    for (unsigned i = 0; i < OUTPUTS; i++) {
        const uint64_t bytes = (capacity[i] + 2 * GUARD) * sizeof(float);
        slabs[i] = ds4_gpu_tensor_alloc(bytes);
        reference[i] = malloc(bytes);
        CHECK(slabs[i] && reference[i]);
        f.result[i] = ds4_gpu_tensor_view(slabs[i], GUARD * sizeof(float), capacity[i] * sizeof(float));
        CHECK(f.result[i]);
    }
    for (unsigned i = 0; i < max_output; i++)
        addend[i] = ((int)(random_u32(&state) % 101) - 50) / 256.0f;
    CHECK(ds4_gpu_tensor_write(f.addend, 0, addend, sizeof(addend)));

    for (unsigned c = 0; c < sizeof(widths) / sizeof(*widths); c++) {
        const bool with_addend = c == 3;
        if (glm53 && with_addend) {
            printf("Q2 decode profile=glm53 experts=%u width=%u addend=1: SKIP (non-TP eight-route addend unsupported)\n",
                   f.experts, widths[c]);
            continue;
        }
        /* gate ne11=1 broadcasts x. The fused IQ2 kernel also uses idx%ne11
         * for gate/up stores: only their first mid_dim entries are shared scratch,
         * with no defined winning expert. Mid instead uses idx and saves all
         * selected rows; mid/out are deterministic observable outputs. */
        /* Eight routes consume all per-expert Q2 rows, including with the tuned down path. */
        const uint32_t count[OUTPUTS] = {f.mid_dim, f.mid_dim, f.selected * f.mid_dim,
                                         glm53 ? f.selected * widths[c] : 0, widths[c]};
        const unsigned variants = (full_experts || glm53) && c == 0 && !reference_only ? 2 : 1;
        for (unsigned variant = 0; variant < variants; variant++) {
            const bool pair_case = (full_experts || glm53) && c == 0 && variant == 0 && !reference_only;
            const bool down_case = (full_experts || glm53) && c == 0 && variant == 1 && !reference_only;
            const char *name = full_experts && pair_case ? "IQ2 pack2 shape eligible (Q2 off)" :
                               full_experts && down_case ? "Q2 sum6 static eligible (shape off)" :
                               reference_only ? (glm53 ? "reference (Q2 generic down)" : "reference") :
                               glm53 && pair_case ? "IQ2 glm53 shape eligible (Q2 generic down)" :
                               glm53 && down_case ? "Q2 glm53 down shape eligible (IQ2 generic pair)" :
                               glm53 ? "IQ2 glm53 / Q2 down guard probes" :
                               profile && c == 0 ? (profile == PROFILE_V41 ? "Q2 v41 enum eligible" : "Q2 pro enum eligible") :
                               profile ? "Q2 fallback (guarded)" : "Q2 fallback (shape off)";
            const unsigned runs = !full_experts || reference_only || c == 0 ? 4 : 1;
            for (unsigned route = 0; route < ROUTES; route++) {
                state = 65537u + route * 104729u;
                for (unsigned i = 0; i < f.input_dim; i++)
                    input[i] = ldexpf(((int)(random_u32(&state) % 1025) - 512) / 1024.0f, (int)route - 2);
                CHECK(ds4_gpu_tensor_write(f.x, 0, input, sizeof(input)));
                CHECK(ds4_gpu_tensor_write(f.ids, 0, ids[route], f.selected * sizeof(int32_t)));
                CHECK(ds4_gpu_tensor_write(f.weights, 0, weights[route], f.selected * sizeof(float)));
                for (unsigned run = 0; run < runs; run++) {
                    const bool rollback = reference_only || run == 0 || ((full_experts || glm53) ? run == 2 : run == 3);
                    const char *rollback_values[] = {"1", "0", ""}; /* Presence, not truthiness. */
                    if (glm53) {
                        CHECK(setenv(disable[1], "1", 1) == 0);
                        CHECK(!rollback && !down_case ? unsetenv(disable[3]) == 0 : setenv(disable[3], "1", 1) == 0);
                        CHECK(!rollback && !pair_case ? unsetenv(disable[4]) == 0 : setenv(disable[4], "1", 1) == 0);
                    } else if (full_experts) {
                        CHECK(pair_case && !rollback ? unsetenv(disable[0]) == 0 : setenv(disable[0], "1", 1) == 0);
                        /* Non-eligible widths/addend request tuning once, then check its fallback guards. */
                        CHECK((down_case && !rollback) || (!reference_only && c != 0) ?
                              unsetenv(disable[1]) == 0 : setenv(disable[1], "1", 1) == 0);
                    } else {
                        CHECK(rollback ? setenv(disable[1], rollback_values[(route + run) % 3], 1) == 0
                                       : unsetenv(disable[1]) == 0);
                    }
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
                                fprintf(stderr, "Q2 invalid/guard experts=%u variant=%s width=%u addend=%d route=%u run=%u %s[%u]=0x%08x\n",
                                        f.experts, full_experts || profile ? name : "default", widths[c], with_addend, route, run, names[i], j, bits);
                                goto done;
                            }
                            if (run && !(i < 2 && active) &&
                                memcmp(&reference[i][j], &actual[j], sizeof(float))) {
                                memcpy(&expected, &reference[i][j], sizeof(expected));
                                fprintf(stderr, "Q2 mismatch experts=%u variant=%s width=%u addend=%d route=%u run=%u %s[%u] ref=0x%08x actual=0x%08x\n",
                                        f.experts, full_experts || profile ? name : "default", widths[c], with_addend, route, run, names[i], j, expected, bits);
                                goto done;
                            }
                            if (i == 4 && active && route == 2)
                                CHECK(actual[j] == (with_addend ? addend[j - GUARD] : 0.0f));
                            /* FNV-1a, canonical low-to-high bytes of every defined
                             * output word, once per case, independent of run mode. */
                            if (run == 0 && variant == 0 && (i == 2 || i == 4) && active) {
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
            if (full_experts)
                printf("Q2 decode experts=%u variant=%s width=%u addend=%d: PASS (4 routes x %s, mid/out %s, scratch finite, sentinels intact)\n",
                       f.experts, name, widths[c], with_addend, reference_only ? "AAAA" : c == 0 ? "ABAB" : "A",
                       runs == 4 ? "bitwise exact" : "finite");
            else if (profile)
                printf("Q2 decode profile=%s experts=%u input=%u mid=%u variant=%s width=%u addend=%d: PASS (4 routes x %s, %s, sentinels intact)\n",
                       profile == PROFILE_V41 ? "v41" : profile == PROFILE_PRO ? "pro" : "glm53",
                       f.experts, f.input_dim, f.mid_dim, name, widths[c], with_addend,
                       reference_only ? "AAAA" : glm53 ? "ABAB" : "ABBA",
                       glm53 ? "8 expert-output scratch rows/mid/out bitwise exact" : "mid/out bitwise exact, scratch finite");
            else
                printf("Q2 decode width=%u addend=%d: PASS (4 routes x %s, mid/out bitwise exact, scratch finite, sentinels intact)\n",
                       widths[c], with_addend, reference_only ? "AAAA" : "ABBA");
        }
    }
    printf("Q2 defined outputs checksum=%016llx (%s)\n",
           (unsigned long long)checksum,
           glm53 ? "mid/out bits, once per executed case; addend skipped" :
                   "all mid/out bits, once per case");

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
    for (unsigned i = 0; i < sizeof(disable) / sizeof(*disable); i++) {
        if (saved[i] ? setenv(disable[i], saved[i], 1) != 0 : unsetenv(disable[i]) != 0) rc = 1;
        free(saved[i]);
    }
    return rc;
}
