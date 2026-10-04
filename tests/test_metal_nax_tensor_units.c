/*
 * M4-3 (G2f) acceptance: NAX tensor-API >= 2 GiB addressing units
 * (tests/test_metal_nax_tensor_units.c, build pattern copied from
 * tests/test_metal_graph_capture: single .c linked against ds4_metal.o,
 * NO model loaded).
 *
 * Upstream llama.cpp #28748 pinned the failure mode: a Metal tensor
 * slice's byte offset is int32 and wraps at 2 GiB.  Every NAX family in
 * this repo (direct_rhs / mpp / packed / mla / indexer) lives on buffers
 * whose real offsets ARE multi-GiB (whole GGUF map, expert blobs with
 * >2 GiB per-expert strides, KV/index caches at long ctx), so each family
 * gets a runnable >= 2 GiB offset vector through
 * ds4_gpu_test_nax_2gib_units() in ds4_metal.m.  See the audit ledger row
 * in LOCAL_INVENTORY.md section D for the static audit these units pin.
 *
 * Memory posture: each unit allocates 2..4.3 GiB of SHARED MTLBuffer
 * (CPU-writable) serially and frees it; the preflight below refuses to
 * start below a composite-available-memory floor (override with
 * DS4_NAX_2GIB_MIN_AVAIL_GIB; set 0 in a loaded-model window where the
 * operator has already gated memory, per ITERATION_PLAN_2026Q4 M1-6).
 */
#define _DARWIN_C_SOURCE
#include "ds4_gpu.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }

extern int ds4_gpu_test_nax_2gib_units(void);

int main(void) {
    const char *min_env = getenv("DS4_NAX_2GIB_MIN_AVAIL_GIB");
    const long floor_gib = min_env ? atol(min_env) : 24;
    if (floor_gib > 0) {
        FILE *p = popen("vm_stat", "r");
        if (!p) { perror("vm_stat"); return 2; }
        char line[256];
        double avail = 0;
        while (fgets(line, sizeof(line), p)) {
            double v = 0;
            if (sscanf(line, "Pages free: %lf", &v) == 1 ||
                sscanf(line, "Pages inactive: %lf", &v) == 1 ||
                sscanf(line, "Pages speculative: %lf", &v) == 1 ||
                sscanf(line, "Pages purgeable: %lf", &v) == 1)
                avail += v;
        }
        pclose(p);
        long avail_gib = (long)(avail * 16384.0 / (1ull << 30));
        if (avail_gib < floor_gib) {
            fprintf(stderr, "nax 2GiB units: composite avail %ld GiB < %ld "
                    "GiB floor -- refusing to allocate multi-GiB buffers "
                    "(ITERATION_PLAN rule: check memory before big allocs)\n",
                    avail_gib, floor_gib);
            return 2;
        }
    }
    int fails = ds4_gpu_test_nax_2gib_units();
    printf("nax-tensor-units: %s\n", fails ? "FAILED" : "ok");
    return fails ? 1 : 0;
}
