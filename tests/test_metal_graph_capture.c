/*
 * M1-1 acceptance: Metal decode-graph skeleton
 * (tests/test_metal_graph_capture.c, build pattern copied from
 * tests/test_glm53_router_shared: single .c linked against ds4_metal.o,
 * NO model loaded).
 *
 * Ruling (this session, LOCAL_INVENTORY.md section D): macOS 26.5.1 on M5
 * Max has NO usable record-then-replay substrate -- MTLGraph is removed
 * from the OS and MTLIndirectCommandBuffer segfaults on every recorded-bind
 * variant of the AGX driver (silent no-op on the only non-crashing variant).
 * The ds4_gpu_decode_graph_* machinery therefore ships as a full skeleton:
 * key table + LRU + capture probe through the recording proxy (which
 * validates the recordable surface and keeps the poison discipline) +
 * per-key param buffer + log anchors, while every well-formed capture
 * retires to EAGER (end() == -1, island byte-exact once).  Replay/replay-
 * count assertions are impossible until M2-3 lands a substrate; this suite
 * locks in exactly the guarantees that protect the bit contract today:
 *   - default OFF; begin without an open batch stays eager (first mine);
 *   - a capture that sees a batch rotation mid-island poisons and retires;
 *   - an unrecordable encoder selector poisons (fail-closed, never fake);
 *   - abort drops the capture and the key stays recapturable;
 *   - no captured work ever executes (eager retries see exact results);
 *   - the param buffer is live while the capture owns the key;
 *   - the key table evicts at capacity 64 (retirement reclaim included)
 *     and invalidate() reopens every key.
 */
#define _DARWIN_C_SOURCE
#include "ds4_gpu.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }

extern int ds4_gpu_decode_graph_param_map(const ds4_decode_graph_key *key,
                                          void                      **ptr_out,
                                          uint64_t                   *size_out);
extern void ds4_gpu_decode_graph_stats(uint64_t *captures,
                                       uint64_t *replays,
                                       uint64_t *captures_failed,
                                       uint64_t *evictions);
extern void ds4_gpu_decode_graph_test_set_enabled(int on);
extern void ds4_gpu_decode_graph_test_reset(void);
extern void ds4_gpu_decode_graph_test_poison_probe(void);
extern int ds4_gpu_decode_graph_test_encode_add_one(ds4_gpu_tensor *x);
extern int ds4_gpu_decode_graph_test_encode_add_param(ds4_gpu_tensor *x);

static int g_fail;

static void req(int ok, const char *what, int line) {
    if (!ok) {
        fprintf(stderr, "metal-graph-capture FAIL (line %d): %s\n", line, what);
        g_fail++;
    }
}
#define REQ(ok, what) req((ok) != 0, (what), __LINE__)

static uint32_t read_u32(ds4_gpu_tensor *t) {
    uint32_t v = 0;
    if (ds4_gpu_tensor_read(t, 0, &v, sizeof(v)) == 0) {
        fprintf(stderr, "metal-graph-capture FAIL: tensor read\n");
        g_fail++;
    }
    return v;
}

static void key_init(ds4_decode_graph_key *k, uint32_t il, void *identity) {
    memset(k, 0, sizeof(*k));
    k->il = il;
    k->island = 1u;
    k->cur_hc = identity;
}

static uint64_t stat_caps, stat_replays, stat_failed, stat_evict;
static void stats(void) {
    ds4_gpu_decode_graph_stats(&stat_caps, &stat_replays, &stat_failed, &stat_evict);
}

static ds4_gpu_tensor *alloc_u32(void) {
    ds4_gpu_tensor *t = ds4_gpu_tensor_alloc(64);
    REQ(t != NULL, "tensor alloc");
    return t;
}

int main(void) {
    if (!ds4_gpu_init()) {
        fprintf(stderr, "metal-graph-capture: GPU init failed\n");
        return 1;
    }

    /* --- default OFF: unset env means no capture, everything eager ----- */
    REQ(ds4_gpu_decode_graphs_supported() == 0,
        "graphs are OFF without DS4_METAL_DECODE_GRAPHS");
    ds4_gpu_decode_graph_test_set_enabled(1);
    ds4_gpu_decode_graph_test_reset();

    /* --- first mine: no open batch -> never capture --------------------- */
    {
        ds4_gpu_decode_graph_test_reset();
        ds4_decode_graph_key k;
        ds4_gpu_tensor *x = alloc_u32();
        key_init(&k, 1, x);
        REQ(ds4_gpu_decode_graph_begin(&k) == -1,
            "begin without an active command batch stays eager");
    }

    /* --- capture probe records, retires to eager, never double-runs ----- */
    {
        ds4_gpu_decode_graph_test_reset();
        ds4_decode_graph_key k;
        ds4_gpu_tensor *x = alloc_u32();
        key_init(&k, 2, x);
        REQ(ds4_gpu_begin_commands(), "batch open");
        REQ(ds4_gpu_decode_graph_begin(&k) == 0, "first encounter opens a capture");
        REQ(ds4_gpu_decode_graph_test_encode_add_one(x), "record add_one #1");
        REQ(ds4_gpu_decode_graph_test_encode_add_one(x), "record add_one #2");
        REQ(ds4_gpu_decode_graph_end(&k) == -1,
            "capture retires without a replay substrate (end reports -1)");
        REQ(ds4_gpu_decode_graph_begin(&k) == -1, "retired key stays eager");
        REQ(ds4_gpu_decode_graph_test_encode_add_one(x), "eager re-encode");
        REQ(ds4_gpu_end_commands(), "batch close");
        REQ(read_u32(x) == 1u, "captured dispatches never executed; eager ran once");
        stats();
        REQ(stat_caps == 0, "no capture commits without a substrate");
        REQ(stat_replays == 0, "no replays without a substrate");
        REQ(stat_failed == 1, "retired capture counted as failed");
    }

    /* --- first mine: batch rotation mid-capture poisons ------------------ */
    {
        ds4_gpu_decode_graph_test_reset();
        ds4_decode_graph_key k;
        ds4_gpu_tensor *x = alloc_u32();
        key_init(&k, 3, x);
        REQ(ds4_gpu_begin_commands(), "batch open");
        REQ(ds4_gpu_decode_graph_begin(&k) == 0, "capture opens");
        REQ(ds4_gpu_decode_graph_test_encode_add_one(x), "record #1");
        REQ(ds4_gpu_flush_commands(), "mid-capture flush rotates the batch");
        REQ(ds4_gpu_decode_graph_test_encode_add_one(x), "record after rotation");
        REQ(ds4_gpu_decode_graph_end(&k) == -1, "poisoned capture end fails");
        REQ(ds4_gpu_decode_graph_begin(&k) == -1, "poisoned key is retired");
        REQ(ds4_gpu_decode_graph_test_encode_add_one(x), "eager re-encode");
        REQ(ds4_gpu_end_commands(), "batch close");
        REQ(read_u32(x) == 1u, "only the eager re-encode ran");
        stats();
        REQ(stat_failed == 1, "poisoned capture counted once");
    }

    /* --- unknown encoder selector poisons (fail-closed) ------------------ */
    {
        ds4_gpu_decode_graph_test_reset();
        ds4_decode_graph_key k;
        ds4_gpu_tensor *x = alloc_u32();
        key_init(&k, 4, x);
        REQ(ds4_gpu_begin_commands(), "batch open");
        REQ(ds4_gpu_decode_graph_begin(&k) == 0, "capture opens");
        REQ(ds4_gpu_decode_graph_test_encode_add_one(x), "record one");
        ds4_gpu_decode_graph_test_poison_probe();
        REQ(ds4_gpu_decode_graph_end(&k) == -1,
            "unrecordable selector fails the capture, no crash");
        REQ(ds4_gpu_decode_graph_test_encode_add_one(x), "eager fallback");
        REQ(ds4_gpu_end_commands(), "batch close");
        REQ(read_u32(x) == 1u, "poisoned capture ran eagerly only");
    }

    /* --- abort drops the capture; the key stays recapturable ------------- */
    {
        ds4_gpu_decode_graph_test_reset();
        ds4_decode_graph_key k;
        ds4_gpu_tensor *x = alloc_u32();
        key_init(&k, 5, x);
        REQ(ds4_gpu_begin_commands(), "batch open");
        REQ(ds4_gpu_decode_graph_begin(&k) == 0, "capture opens");
        REQ(ds4_gpu_decode_graph_test_encode_add_one(x), "record before abort");
        ds4_gpu_decode_graph_abort(&k);
        REQ(ds4_gpu_decode_graph_begin(&k) == 0, "aborted key recaptures");
        REQ(ds4_gpu_decode_graph_test_encode_add_one(x), "record after re-record");
        REQ(ds4_gpu_decode_graph_end(&k) == -1, "re-recorded capture retires");
        REQ(ds4_gpu_end_commands(), "batch close");
        REQ(read_u32(x) == 0u, "neither capture executed; nothing ran");
        REQ(ds4_gpu_begin_commands(), "batch open (eager)");
        REQ(ds4_gpu_decode_graph_test_encode_add_one(x), "eager add_one");
        REQ(ds4_gpu_end_commands(), "batch close (eager)");
        REQ(read_u32(x) == 1u, "eager island after the aborted capture ran once");
        ds4_gpu_decode_graph_abort(&k);  /* abort with no capture is a no-op */
    }

    /* --- param buffer is live while the capture owns the key ------------- */
    {
        ds4_gpu_decode_graph_test_reset();
        ds4_decode_graph_key k;
        ds4_gpu_tensor *x = alloc_u32();
        key_init(&k, 6, x);
        void *p = NULL;
        uint64_t psz = 0;
        REQ(ds4_gpu_decode_graph_param_map(&k, &p, &psz) == 0,
            "unknown key has no param buffer");
        REQ(ds4_gpu_begin_commands(), "batch open");
        REQ(ds4_gpu_decode_graph_begin(&k) == 0, "capture opens");
        REQ(ds4_gpu_decode_graph_test_encode_add_param(x), "record add_param");
        REQ(ds4_gpu_decode_graph_param_map(&k, &p, &psz) == 1,
            "capturing key maps its param buffer");
        REQ(p != NULL && psz == 8192u, "param buffer sized");
        ((uint32_t *)p)[0] = 100u;   /* host write must not crash mid-capture */
        REQ(((uint32_t *)p)[0] == 100u, "param write visible in the same map");
        REQ(ds4_gpu_decode_graph_end(&k) == -1, "capture retires");
        REQ(ds4_gpu_decode_graph_param_map(&k, &p, &psz) == 0,
            "retired key releases its param buffer");
        REQ(ds4_gpu_end_commands(), "batch close");
    }

    /* --- key table capacity: retirement reclaim past 64 ------------------ */
    {
        ds4_gpu_decode_graph_test_reset();
        enum { CAP = 64, OVER = 70 };
        ds4_gpu_tensor *pool[OVER];
        ds4_decode_graph_key pk[OVER];
        for (int i = 0; i < OVER; i++) {
            pool[i] = alloc_u32();
            key_init(&pk[i], (uint32_t)(100 + i), pool[i]);
            REQ(ds4_gpu_begin_commands(), "pool batch open");
            REQ(ds4_gpu_decode_graph_begin(&pk[i]) == 0,
                "every fresh key gets a capture slot (retirement reclaim past 64)");
            REQ(ds4_gpu_decode_graph_test_encode_add_one(pool[i]), "pool record");
            REQ(ds4_gpu_decode_graph_end(&pk[i]) == -1, "pool capture retires");
            REQ(ds4_gpu_decode_graph_begin(&pk[i]) == -1, "pool key retired");
            REQ(ds4_gpu_decode_graph_test_encode_add_one(pool[i]), "pool eager");
            REQ(ds4_gpu_end_commands(), "pool batch close");
            REQ(read_u32(pool[i]) == 1u, "pool island ran eagerly exactly once");
        }
        stats();
        REQ(stat_evict == OVER - CAP,
            "retirements past capacity reclaimed exactly one slot each");
    }

    /* --- invalidate drops retirements and reopens keys ------------------- */
    {
        ds4_gpu_decode_graph_test_reset();
        ds4_decode_graph_key k;
        ds4_gpu_tensor *x = alloc_u32();
        key_init(&k, 7, x);
        REQ(ds4_gpu_begin_commands(), "batch open");
        REQ(ds4_gpu_decode_graph_begin(&k) == 0, "capture opens");
        REQ(ds4_gpu_decode_graph_test_encode_add_one(x), "record");
        REQ(ds4_gpu_decode_graph_end(&k) == -1, "capture retires");
        REQ(ds4_gpu_decode_graph_begin(&k) == -1, "key retired");
        ds4_gpu_decode_graphs_invalidate();
        REQ(ds4_gpu_decode_graph_begin(&k) == 0,
            "invalidate reopens the retired key for a fresh capture");
        ds4_gpu_decode_graph_abort(&k);
        REQ(ds4_gpu_end_commands(), "batch close");
    }

    if (g_fail != 0) {
        fprintf(stderr, "metal-graph-capture: %d FAILED checks\n", g_fail);
        return 1;
    }
    printf("metal-graph-capture: all checks passed (skeleton substrate)\n");
    return 0;
}
