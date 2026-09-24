#define _DARWIN_C_SOURCE
#include "ds4_gpu.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { K = 8192, M = 4096, MAX_K = K + 32, MAX_M = M + 1, GUARD = 16 };
typedef struct { uint16_t d; int8_t qs[32]; } q8_block;

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); goto done; \
} } while (0)

int main(int argc, char **argv) {
    const bool reference_only = argc == 2 && !strcmp(argv[1], "--reference-only");
    if (argc != 1 && !reference_only) {
        fprintf(stderr, "usage: %s [--reference-only]\n", argv[0]);
        return 2;
    }
    const char *disable = "DS4_METAL_DISABLE_M5_GLM53_KDA_Q8_SHAPE";
    if (getenv("DS4_METAL_Q8_MV_NSG")) {
        fprintf(stderr, "unset DS4_METAL_Q8_MV_NSG before running\n");
        return 2;
    }
    const size_t model_bytes = (size_t)MAX_M * (MAX_K / 32) * sizeof(q8_block);
    const size_t slab_bytes = (MAX_M + 2 * GUARD) * sizeof(float);
    const uint32_t sentinel = 0x7fc12345u;
    const struct { int k, m; } shapes[] = {
        {K, M}, {K, M - 1}, {K, M + 1}, {K - 32, M}, {K + 32, M},
    };
    void *model = NULL;
    ds4_gpu_tensor *x = NULL, *slab = NULL, *out = NULL;
    uint32_t *actual = NULL, *baseline = NULL;
    float input[MAX_K];
    char timeline_path[] = "/tmp/ds4-q8-shape-XXXXXX";
    uint64_t checksum = UINT64_C(14695981039346656037);
    int rc = 1;

    CHECK(sizeof(q8_block) == 34);
    CHECK(!getenv("DS4_METAL_ENCODER_TIMELINE"));
    int timeline_fd = mkstemp(timeline_path);
    CHECK(timeline_fd >= 0);
    CHECK(close(timeline_fd) == 0);
    CHECK(setenv("DS4_METAL_ENCODER_TIMELINE", timeline_path, 1) == 0);
    CHECK(posix_memalign(&model, (size_t)getpagesize(), model_bytes) == 0);
    q8_block *blocks = model;
    for (size_t i = 0; i < model_bytes / sizeof(*blocks); i++) {
        blocks[i].d = 0x3c00;
        for (unsigned j = 0; j < 32; j++)
            blocks[i].qs[j] = (int8_t)((int)((i * 13 + j * 17) % 127) - 63);
    }
    for (int i = 0; i < MAX_K; i++) input[i] = ((i * 29) % 257 - 128) / 256.0f;
    CHECK(ds4_gpu_init());
    if (!ds4_gpu_device_is_m5_apple_silicon()) {
        puts("Q8 decode shape: SKIP (M5 only)");
        rc = 0;
        goto done;
    }
    CHECK(ds4_gpu_set_model_map(model, model_bytes));
    x = ds4_gpu_tensor_alloc(sizeof(input));
    slab = ds4_gpu_tensor_alloc(slab_bytes);
    out = slab ? ds4_gpu_tensor_view(slab, GUARD * sizeof(float), MAX_M * sizeof(float)) : NULL;
    actual = malloc(slab_bytes);
    baseline = malloc(slab_bytes);
    CHECK(x && slab && out && actual && baseline);
    CHECK(ds4_gpu_tensor_write(x, 0, input, sizeof(input)));

    unsigned shaped_dispatches = 0;
    for (unsigned c = 0; c < sizeof(shapes) / sizeof(*shapes); c++) {
        for (unsigned run = 0; run < 4; run++) {
            const bool generic = reference_only || run == 0 || run == 3;
            CHECK(generic ? setenv(disable, "1", 1) == 0 : unsetenv(disable) == 0);
            for (int j = 0; j < MAX_M + 2 * GUARD; j++) actual[j] = sentinel;
            CHECK(ds4_gpu_tensor_write(slab, 0, actual, slab_bytes));
            CHECK(ds4_gpu_begin_commands());
            CHECK(ds4_gpu_matmul_q8_0_tensor(out, model, model_bytes, 0,
                                             shapes[c].k, shapes[c].m, x, 1));
            CHECK(ds4_gpu_end_commands());
            FILE *timeline = fopen(timeline_path, "r");
            CHECK(timeline);
            char line[512];
            shaped_dispatches = 0;
            while (fgets(line, sizeof(line), timeline)) {
                if (strstr(line, "kernel_mul_mv_q8_0_glm53_kda_output_f32"))
                    shaped_dispatches++;
            }
            const int read_ok = !ferror(timeline);
            const int timeline_ok = fclose(timeline) == 0 && read_ok;
            const unsigned expected = reference_only ? 0u :
                c == 0 ? (run == 0 ? 0u : run == 1 ? 1u : 2u) : 2u;
            if (!timeline_ok || shaped_dispatches != expected) {
                fprintf(stderr, "Q8 shaped dispatch count case=%u run=%u actual=%u expected=%u\n",
                        c, run, shaped_dispatches, expected);
                goto done;
            }
            CHECK(ds4_gpu_tensor_read(slab, 0, actual, slab_bytes));
            for (int j = 0; j < MAX_M + 2 * GUARD; j++) {
                const bool active = j >= GUARD && j < GUARD + shapes[c].m;
                float value;
                memcpy(&value, &actual[j], sizeof(value));
                if (active ? !isfinite(value) : actual[j] != sentinel) {
                    fprintf(stderr, "invalid output case=%u run=%u index=%d bits=%08x\n",
                            c, run, j, actual[j]);
                    goto done;
                }
                if (run && actual[j] != baseline[j]) {
                    fprintf(stderr, "Q8 mismatch case=%u run=%u index=%d generic=%08x actual=%08x\n",
                            c, run, j, baseline[j], actual[j]);
                    goto done;
                }
                if (run == 0 && active) {
                    for (unsigned byte = 0; byte < sizeof(actual[j]); byte++) {
                        checksum ^= (actual[j] >> (8 * byte)) & 0xffu;
                        checksum *= UINT64_C(1099511628211);
                    }
                }
            }
            if (run == 0) memcpy(baseline, actual, slab_bytes);
        }
        printf("Q8 decode K=%d M=%d: PASS (%s bitwise, guards intact)\n",
               shapes[c].k, shapes[c].m, reference_only ? "AAAA" : "ABBA");
    }
    printf("Q8 shaped dispatches=%u\n", shaped_dispatches);
    printf("Q8 defined outputs checksum=%016llx\n", (unsigned long long)checksum);
    rc = 0;
done:
    unsetenv(disable);
    unsetenv("DS4_METAL_ENCODER_TIMELINE");
    unlink(timeline_path);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(slab);
    ds4_gpu_tensor_free(x);
    ds4_gpu_cleanup();
    free(actual);
    free(baseline);
    free(model);
    return rc;
}
