/* V4.1 speculative verification. Resident real-model fixture; run it only on a
 * dedicated Metal host:
 *   test_deepseek41_dspark MODEL --verify-parity PROMPT   checks
 *   test_deepseek41_dspark MODEL --short-prefill PROMPT   checks
 *   test_deepseek41_dspark MODEL --verify-timing PROMPT   measures
 *   test_deepseek41_dspark MODEL --verify-scan[-ssd] PROMPT
 *                                                          measures the cost
 *   slope of a verified row (a) across n = 2..8, the number draftless n-gram
 *   speculative decoding turned on; DS4_TEST_VERIFY_WINDOW=same verifies one
 *   token n times for the expert-union floor.
 *   test_deepseek41_dspark MODEL --stage-timing PROMPT    measures
 * The --verify-parity-ssd and --short-prefill-ssd variants run the same checks
 * with the engine in SSD-streaming mode, where verification rows bind their
 * routed experts through the streaming selected-address gather. These run on a
 * host that cannot hold the resident model.
 *   test_deepseek41_dspark MODEL --moe-bind-parity   checks one layer's batch
 *   MoE twice on identical inputs: whole-map bind vs the streaming address
 *   table, bit-exact (HANDOFF-V41 12.5). Rows sweep past the mm-id and
 *   packed-TensorOps thresholds to 1024, so both arms run the same expert
 *   matmul family with only the weight bind differing (HANDOFF-V41 13.4). */
#include "../ds4.c"
#include <libproc.h>

/* Bytes this process has read from disk, cumulative. The decode step's cost is
 * either weight traffic from RAM or weight traffic from SSD, and this is what
 * tells them apart: bytes/token against the ~3.5 GB a token's weights occupy
 * gives the expert-cache miss rate directly, with no ambiguity from other
 * processes the way iostat has. */
static uint64_t disk_read_bytes(void) {
    struct rusage_info_v4 ri;
    if (proc_pid_rusage(getpid(), RUSAGE_INFO_V4, (rusage_info_t *)&ri) != 0) return 0;
    return ri.ri_diskio_bytesread;
}

#define REQUIRE(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); goto done; \
} } while (0)

static char *read_text(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    fseek(fp, 0, SEEK_END);
    const long size = ftell(fp);
    rewind(fp);
    char *text = size > 0 ? malloc((size_t)size + 1u) : NULL;
    if (text && fread(text, 1, (size_t)size, fp) != (size_t)size) { free(text); text = NULL; }
    if (text) text[size] = 0;
    fclose(fp);
    return text;
}

/* Streaming runs carry an explicit expert-cache budget. The target covers a
 * prefill headroom (~7.1 GiB at ctx 4096) first; the dynamic expert cache is
 * what remains, and it must hold one layer's experts at once (DS4_N_EXPERT *
 * ~9.49 MiB ~= 3.64 GiB on V4.1 Q2) for the streaming gather kernels to be
 * admitted at all. DS4_TEST_STREAMING_CACHE_GIB overrides it. */
static uint64_t ds41_test_streaming_cache_bytes(void) {
    const char *env = getenv("DS4_TEST_STREAMING_CACHE_GIB");
    uint32_t gib = env ? (uint32_t)atoi(env) : 16u;
    if (gib < 8u) gib = 8u;
    return (uint64_t)gib * 1024ull * 1024ull * 1024ull;
}

/* A verified suffix must be what one-token decode computes, and any kept
 * prefix must leave the session that decode would have left: every logit of
 * every row, the serialized state, and the tokens that follow. */
static int check_verify_parity(const char *model, const char *prompt_path, bool streaming) {
    ds4_engine *engine = NULL;
    ds4_session *scalar = NULL, *verified = NULL;
    ds4_session_snapshot base = {0}, want = {0}, got = {0};
    ds4_tokens prompt = {0}, prefix = {0};
    ds41_spec spec = {0};
    float *rows = NULL;
    char err[256] = {0};
    char *text = read_text(prompt_path);
    int rc = 1;
    const uint32_t targets[] = {37, 38, 39};
    ds4_engine_options opt = {.model_path = model, .backend = DS4_BACKEND_METAL,
        .context_size = 4096, .power_percent = 100};
    if (streaming) {
        opt.ssd_streaming = true;
        opt.ssd_streaming_cache_bytes = ds41_test_streaming_cache_bytes();
    }
    REQUIRE(text);
    REQUIRE(ds4_engine_open(&engine, &opt) == 0);
    ds4_tokenize_text(engine, text, &prompt);
    REQUIRE(prompt.len > 1100);
    REQUIRE(ds4_session_create(&scalar, engine, 4096) == 0);
    REQUIRE(ds4_session_create(&verified, engine, 4096) == 0);
    REQUIRE(ds41_spec_alloc(&spec, targets, 3));
    rows = malloc((size_t)DS4_TP_BATCH_MAX_ROWS * DS4_N_VOCAB * sizeof(float));
    REQUIRE(rows);
    /* Below and across the 128-row window, odd and even frontiers, and a
     * frontier whose rows straddle a window wrap. */
    const struct { int prefix; uint32_t count; } cases[] = {
        {40, 2}, {40, 3}, {40, 6}, {41, 5}, {126, 6}, {250, 8}, {251, 2}, {1021, 6}, {1022, 3},
    };
    /* DS4_TEST_VERIFY_CASE=PREFIX,ROWS runs that one frontier instead. */
    int only_prefix = 0, only_count = 0;
    const char *only = getenv("DS4_TEST_VERIFY_CASE");
    if (only && sscanf(only, "%d,%d", &only_prefix, &only_count) != 2) only = NULL;
    for (size_t c = 0; c < (only ? 1u : sizeof(cases) / sizeof(cases[0])); c++) {
        const int start = only ? only_prefix : cases[c].prefix;
        const uint32_t count = only ? (uint32_t)only_count : cases[c].count;
        REQUIRE(start > 0 && count >= 2 && count <= DS4_TP_BATCH_MAX_ROWS);
        const int *suffix = prompt.v + start;
        prefix.len = 0;
        for (int i = 0; i < start; i++) ds4_tokens_push(&prefix, prompt.v[i]);
        REQUIRE(ds4_session_sync(scalar, &prefix, err, sizeof(err)) == 0);
        REQUIRE(ds4_session_save_snapshot(scalar, &base, err, sizeof(err)) == 0);
        /* One-token reference rows. */
        for (uint32_t i = 0; i < count; i++) {
            REQUIRE(ds4_session_eval(scalar, suffix[i], err, sizeof(err)) == 0);
            memcpy(rows + (size_t)i * DS4_N_VOCAB, scalar->logits, DS4_N_VOCAB * sizeof(float));
        }
        for (uint32_t keep = 1; keep <= count; keep++) {
            REQUIRE(ds4_session_load_snapshot(verified, &base, err, sizeof(err)) == 0);
            ds41_gpu_graph *g = &verified->ds41_graph;
            REQUIRE(g->pos == (uint32_t)start);
            REQUIRE(ds41_graph_verify(g, &engine->model, &engine->weights, suffix, count, &spec));
            REQUIRE(g->pos == (uint32_t)start);
            const float *logits = ds4_gpu_tensor_contents(spec.logits);
            REQUIRE(logits);
            bool exact = true;
            for (uint32_t i = 0; i < count; i++) {
                if (memcmp(logits + (size_t)i * DS4_N_VOCAB, rows + (size_t)i * DS4_N_VOCAB,
                           DS4_N_VOCAB * sizeof(float)) != 0) {
                    double worst = 0;
                    for (uint32_t v = 0; v < DS4_N_VOCAB; v++)
                        worst = fmax(worst, fabs((double)logits[(size_t)i * DS4_N_VOCAB + v] -
                                                 rows[(size_t)i * DS4_N_VOCAB + v]));
                    fprintf(stderr, "prefix %d rows %u: row %u differs from one-token decode (max %.9g)\n",
                            start, count, i, worst);
                    exact = false;
                }
            }
            if (!exact) goto done;
            REQUIRE(ds4_gpu_begin_commands() && ds41_spec_commit(g, &spec, keep) &&
                    ds4_gpu_end_commands());
            REQUIRE(g->pos == (uint32_t)start + keep);
            for (uint32_t i = 0; i < keep; i++) token_vec_push(&verified->checkpoint, suffix[i]);
            memcpy(verified->logits, logits + (size_t)(keep - 1u) * DS4_N_VOCAB,
                   DS4_N_VOCAB * sizeof(float));
            /* The reference for this prefix, and three tokens past it. */
            REQUIRE(ds4_session_load_snapshot(scalar, &base, err, sizeof(err)) == 0);
            for (uint32_t i = 0; i < keep; i++)
                REQUIRE(ds4_session_eval(scalar, suffix[i], err, sizeof(err)) == 0);
            REQUIRE(ds4_session_save_snapshot(scalar, &want, err, sizeof(err)) == 0);
            REQUIRE(ds4_session_save_snapshot(verified, &got, err, sizeof(err)) == 0);
            REQUIRE(want.len == got.len);
            if (memcmp(want.ptr, got.ptr, want.len) != 0) {
                size_t at = 0;
                while (((const uint8_t *)want.ptr)[at] == ((const uint8_t *)got.ptr)[at]) at++;
                fprintf(stderr, "prefix %d rows %u keep %u: state differs at byte %zu of %zu\n",
                        start, count, keep, at, (size_t)want.len);
                goto done;
            }
            for (uint32_t i = 0; i < 3; i++) {
                const int next = suffix[keep + i];
                REQUIRE(ds4_session_eval(scalar, next, err, sizeof(err)) == 0);
                REQUIRE(ds4_session_eval(verified, next, err, sizeof(err)) == 0);
                REQUIRE(memcmp(scalar->logits, verified->logits, DS4_N_VOCAB * sizeof(float)) == 0);
            }
        }
        fprintf(stderr, "V4.1 verify parity prefix=%d rows=%u: every row, kept prefix and continuation exact\n",
                start, count);
        REQUIRE(ds4_session_load_snapshot(scalar, &base, err, sizeof(err)) == 0);
    }
    puts("V4.1 speculative verification: PASS");
    rc = 0;
done:
    if (err[0]) fprintf(stderr, "error: %s\n", err);
    if (ds4_gpu_commands_active()) ds4_gpu_end_commands();
    free(rows);
    ds41_spec_free(&spec);
    ds4_session_snapshot_free(&base); ds4_session_snapshot_free(&want); ds4_session_snapshot_free(&got);
    ds4_tokens_free(&prefix); ds4_tokens_free(&prompt);
    ds4_session_free(verified); ds4_session_free(scalar);
    ds4_engine_close(engine);
    free(text);
    return rc;
}

/* The production continued prefill of two to eight rows is one-token decode:
 * every routed recipe keeps scalar reductions through eight rows. */
static int check_short_prefill(const char *model, const char *prompt_path, bool streaming) {
    ds4_engine *engine = NULL;
    ds4_session *scalar = NULL, *batched = NULL;
    ds4_tokens prompt = {0}, prefix = {0};
    char err[256] = {0};
    char *text = read_text(prompt_path);
    int rc = 1;
    ds4_engine_options opt = {.model_path = model, .backend = DS4_BACKEND_METAL,
        .context_size = 4096, .power_percent = 100};
    if (streaming) {
        opt.ssd_streaming = true;
        opt.ssd_streaming_cache_bytes = ds41_test_streaming_cache_bytes();
    }
    REQUIRE(text && ds4_engine_open(&engine, &opt) == 0);
    ds4_tokenize_text(engine, text, &prompt);
    REQUIRE(ds4_session_create(&scalar, engine, 4096) == 0);
    REQUIRE(ds4_session_create(&batched, engine, 4096) == 0);
    for (int start = 40; start <= 41; start++) {
        for (int extra = 2; extra <= 8; extra += extra < 5 ? 3 : 1) {
            prefix.len = 0;
            for (int i = 0; i < start; i++) ds4_tokens_push(&prefix, prompt.v[i]);
            ds4_session_invalidate(scalar); ds4_session_invalidate(batched);
            REQUIRE(ds4_session_sync(scalar, &prefix, err, sizeof(err)) == 0);
            REQUIRE(ds4_session_sync(batched, &prefix, err, sizeof(err)) == 0);
            REQUIRE(memcmp(scalar->logits, batched->logits, DS4_N_VOCAB * sizeof(float)) == 0);
            for (int i = 0; i < extra; i++) {
                REQUIRE(ds4_session_eval(scalar, prompt.v[start + i], err, sizeof(err)) == 0);
                ds4_tokens_push(&prefix, prompt.v[start + i]);
            }
            REQUIRE(ds4_session_sync(batched, &prefix, err, sizeof(err)) == 0);
            double worst = 0;
            for (uint32_t v = 0; v < DS4_N_VOCAB; v++)
                worst = fmax(worst, fabs((double)scalar->logits[v] - batched->logits[v]));
            fprintf(stderr, "prefix %d + %d continued rows: max logit difference from decode %.9g\n",
                    start, extra, worst);
            REQUIRE(worst == 0);
        }
    }
    rc = 0;
done:
    if (err[0]) fprintf(stderr, "error: %s\n", err);
    ds4_tokens_free(&prefix); ds4_tokens_free(&prompt);
    ds4_session_free(batched); ds4_session_free(scalar);
    ds4_engine_close(engine);
    free(text);
    return rc;
}

/* E0.3 (M3-7): registered-red contract arm. Batched append versus
 * per-token stepping for rows {2,9,16,31,32,223,831}. Above eight rows the
 * batched path rides the whole-layer mv family while stepping rides the
 * fused tiny-pair scalar path, and at >=32 it rides mm-id (f16 mid) --
 * DRIFT is the registered expectation for the mm-id band. Measured here:
 * [9,31] came back bit-identical (the mv band is row-count independent,
 * which is E0.1's mathematical basis for E1) while >=32 drifts ~2-3 logits
 * across families. Every row reports its worst logit difference; this arm
 * never fails the gate; it registers. */
static int check_short_prefill_rows(const char *model, const char *prompt_path) {
    ds4_engine *engine = NULL;
    ds4_session *scalar = NULL, *batched = NULL;
    ds4_tokens prompt = {0}, prefix = {0};
    char err[256] = {0};
    char *text = read_text(prompt_path);
    int rc = 1, drifts = 0;
    ds4_engine_options opt = {.model_path = model, .backend = DS4_BACKEND_METAL,
        .context_size = 4096, .power_percent = 100,
        .ssd_streaming = true,
        .ssd_streaming_cache_bytes = ds41_test_streaming_cache_bytes()};
    REQUIRE(text && ds4_engine_open(&engine, &opt) == 0);
    ds4_tokenize_text(engine, text, &prompt);
    REQUIRE(ds4_session_create(&scalar, engine, 4096) == 0);
    REQUIRE(ds4_session_create(&batched, engine, 4096) == 0);
    const uint32_t rows_list[] = {2, 9, 16, 31, 32, 223, 831};
    const uint32_t start = 40;
    for (size_t li = 0; li < sizeof(rows_list) / sizeof(rows_list[0]); li++) {
        const uint32_t rows = rows_list[li];
        REQUIRE((uint32_t)prompt.len >= start + rows);
        prefix.len = 0;
        for (uint32_t i = 0; i < start; i++) ds4_tokens_push(&prefix, prompt.v[i]);
        ds4_session_invalidate(scalar); ds4_session_invalidate(batched);
        REQUIRE(ds4_session_sync(scalar, &prefix, err, sizeof(err)) == 0);
        REQUIRE(ds4_session_sync(batched, &prefix, err, sizeof(err)) == 0);
        for (uint32_t i = 0; i < rows; i++) {
            REQUIRE(ds4_session_eval(scalar, prompt.v[start + i], err, sizeof(err)) == 0);
            ds4_tokens_push(&prefix, prompt.v[start + i]);
        }
        REQUIRE(ds4_session_sync(batched, &prefix, err, sizeof(err)) == 0);
        double worst = 0;
        for (uint32_t v = 0; v < DS4_N_VOCAB; v++)
            worst = fmax(worst, fabs((double)scalar->logits[v] - batched->logits[v]));
        if (worst > 0) drifts++;
        fprintf(stderr, "E0.3 rows=%u batched-append vs step: max logit diff %.9g -> %s\n",
                rows, worst, worst > 0 ? "REGISTERED-RED (contract does not exist above eight rows)"
                                       : "IDENTICAL (contract widens to this row set)");
    }
    fprintf(stderr, "E0.3 registered-red summary: %d of %zu row sets drift\n",
            drifts, sizeof(rows_list) / sizeof(rows_list[0]));
    rc = 0;
done:
    if (err[0]) fprintf(stderr, "error: %s\n", err);
    ds4_tokens_free(&prefix); ds4_tokens_free(&prompt);
    ds4_session_free(batched); ds4_session_free(scalar);
    ds4_engine_close(engine);
    free(text);
    return rc;
}

/* Measurement, not a check: what a verified row costs next to a decoded token. */
static int time_verify(const char *model, const char *prompt_path) {
    ds4_engine *engine = NULL;
    ds4_session *session = NULL;
    ds4_session_snapshot base = {0};
    ds4_tokens prompt = {0}, prefix = {0};
    ds41_spec spec = {0};
    char err[256] = {0};
    char *text = read_text(prompt_path);
    int rc = 1;
    const uint32_t targets[] = {37, 38, 39};
    ds4_engine_options opt = {.model_path = model, .backend = DS4_BACKEND_METAL,
        .context_size = 4096, .power_percent = 100};
    REQUIRE(text && ds4_engine_open(&engine, &opt) == 0);
    ds4_tokenize_text(engine, text, &prompt);
    REQUIRE(ds4_session_create(&session, engine, 4096) == 0);
    REQUIRE(ds41_spec_alloc(&spec, targets, 3));
    const int start = 1024;
    for (int i = 0; i < start; i++) ds4_tokens_push(&prefix, prompt.v[i]);
    REQUIRE(ds4_session_sync(session, &prefix, err, sizeof(err)) == 0);
    REQUIRE(ds4_session_save_snapshot(session, &base, err, sizeof(err)) == 0);
    for (int warm = 0; warm < 8; warm++)
        REQUIRE(ds4_session_eval(session, prompt.v[start + warm], err, sizeof(err)) == 0);
    double t = now_sec();
    for (int i = 0; i < 32; i++)
        REQUIRE(ds4_session_eval(session, prompt.v[start + 8 + i], err, sizeof(err)) == 0);
    const double step_ms = (now_sec() - t) * 1000.0 / 32.0;
    fprintf(stderr, "one-token decode: %.2f ms\n", step_ms);
    for (uint32_t rows = 2; rows <= 8; rows += 2) {
        const char *ab_env = getenv("DS4_TEST_VERIFY_AB_ENV");
        if (ab_env && !strncmp(ab_env,"DS4_METAL_DISABLE_V41_",sizeof("DS4_METAL_DISABLE_V41_")-1)) {
            const size_t bytes = (size_t)rows * DS4_N_VOCAB * sizeof(float);
            float *reference = malloc(bytes), *actual = malloc(bytes);
            REQUIRE(reference && actual);
            double totals[2] = {0,0};
            for (unsigned rep=0;rep<20;rep++) {
                const unsigned slot=rep%4, cycle=rep/4;
                const unsigned rollback=(slot==1 || slot==2) ^ (cycle%2);
                if (rollback) setenv(ab_env,"1",1); else unsetenv(ab_env);
                REQUIRE(ds4_session_load_snapshot(session,&base,err,sizeof(err))==0);
                const double begin=now_sec();
                REQUIRE(ds41_graph_verify(&session->ds41_graph,&engine->model,&engine->weights,
                                          prompt.v+start,rows,&spec));
                const double ms=(now_sec()-begin)*1000.0;
                if(rep>=4) totals[rollback]+=ms;
                REQUIRE(ds4_gpu_tensor_read(spec.logits,0,actual,bytes));
                if(!rep) memcpy(reference,actual,bytes);
                else REQUIRE(!memcmp(reference,actual,bytes));
            }
            unsetenv(ab_env);
            fprintf(stderr,"verify_ab env=%s rows=%u control_ms=%.6f rollback_ms=%.6f saved_percent=%.6f exact_floats=%llu\n",
                ab_env,rows,totals[0]/8.0,totals[1]/8.0,
                100.0*(totals[1]-totals[0])/totals[1],
                (unsigned long long)rows*DS4_N_VOCAB*20u);
            free(reference);free(actual);
            continue;
        }
        double total = 0;
        for (int rep = 0; rep < 5; rep++) {
            REQUIRE(ds4_session_load_snapshot(session, &base, err, sizeof(err)) == 0);
            t = now_sec();
            REQUIRE(ds41_graph_verify(&session->ds41_graph, &engine->model, &engine->weights,
                                      prompt.v + start, rows, &spec));
            if (rep) total += (now_sec() - t) * 1000.0;
        }
        fprintf(stderr, "verify %u rows: %.2f ms = %.2f decode steps, %.2f ms per row\n",
                rows, total / 4.0, total / 4.0 / step_ms, total / 4.0 / rows);
    }
    rc = 0;
done:
    if (err[0]) fprintf(stderr, "error: %s\n", err);
    ds41_spec_free(&spec);
    ds4_session_snapshot_free(&base);
    ds4_tokens_free(&prefix); ds4_tokens_free(&prompt);
    ds4_session_free(session);
    ds4_engine_close(engine);
    free(text);
    return rc;
}

static int cmp_double(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y ? 1 : 0;
}

/* Measurement, not a check: the cost slope of a verified row.
 *
 * A speculative step that submits c draft rows runs an (c+1)-row verification
 * in place of one plain decode step. Write V(n) for the wall clock of an n-row
 * verification and D for one plain decode step; then
 *
 *     a(n) = (V(n)/D - 1) / (n - 1)
 *
 * is the slope the draftless n-gram decision turns on, because the offline cost
 * model prices such a step as 1 + a*c decode steps. The scan also times what a
 * step pays beyond the verify itself -- the host argmax over n vocabularies and
 * the commit pass -- since a plain decode step pays neither.
 *
 * Successive reps verify a shifted window so they route to different experts:
 * re-verifying the same eight tokens would leave their experts warm and flatter
 * a streaming run.  Spread across reps is reported because that is what says
 * whether a is cache-limited. */
static int time_verify_scan(const char *model, const char *prompt_path, bool streaming) {
    ds4_engine *engine = NULL;
    ds4_session *session = NULL;
    ds4_session_snapshot base = {0};
    ds4_tokens prompt = {0}, prefix = {0};
    ds41_spec spec = {0};
    char err[256] = {0};
    char *text = read_text(prompt_path);
    int rc = 1;
    const uint32_t targets[] = {37, 38, 39};
    ds4_engine_options opt = {.model_path = model, .backend = DS4_BACKEND_METAL,
        .context_size = 4096, .power_percent = 100};
    if (streaming) {
        opt.ssd_streaming = true;
        opt.ssd_streaming_cache_bytes = ds41_test_streaming_cache_bytes();
        const char *experts = getenv("DS4_TEST_STREAMING_CACHE_EXPERTS");
        opt.ssd_streaming_cache_experts = experts ? (uint32_t)atoi(experts) : 9000u;
    }
    REQUIRE(text);
    REQUIRE(ds4_engine_open(&engine, &opt) == 0);
    ds4_tokenize_text(engine, text, &prompt);
    REQUIRE(prompt.len > 1200);
    REQUIRE(ds4_session_create(&session, engine, 4096) == 0);
    REQUIRE(ds41_spec_alloc(&spec, targets, 3));
    const int start = 1024;
    for (int i = 0; i < start; i++) ds4_tokens_push(&prefix, prompt.v[i]);
    REQUIRE(ds4_session_sync(session, &prefix, err, sizeof(err)) == 0);
    REQUIRE(ds4_session_save_snapshot(session, &base, err, sizeof(err)) == 0);

    const int REPS = 9, DROP = 3;
    double d_before = 0, d_after = 0;
    double disk_d[2] = {0, 0};
    for (int phase = 0; phase < 2; phase++) {
        for (int warm = 0; warm < 8; warm++)
            REQUIRE(ds4_session_eval(session, prompt.v[start + warm], err, sizeof(err)) == 0);
        const uint64_t disk0 = disk_read_bytes();
        const double t = now_sec();
        for (int i = 0; i < 32; i++)
            REQUIRE(ds4_session_eval(session, prompt.v[start + 8 + i], err, sizeof(err)) == 0);
        const double ms = (now_sec() - t) * 1000.0 / 32.0;
        disk_d[phase] = (double)(disk_read_bytes() - disk0) / 32.0;
        if (!phase) d_before = ms; else d_after = ms;
        REQUIRE(ds4_session_load_snapshot(session, &base, err, sizeof(err)) == 0);
    }
    const double d_ms = 0.5 * (d_before + d_after);
    fprintf(stderr, "MODE %s  one-token decode D = %.2f ms (before %.2f, after %.2f)\n",
            streaming ? "ssd-streaming" : "resident", d_ms, d_before, d_after);
    fprintf(stderr, "  disk read per decode step: %.1f MB (phase 1, cold) / %.1f MB "
            "(phase 2, steady) -- a token's weights are 10.50 GB (see "
            "--weight-inventory), so steady-state miss rate is %.2f%%\n",
            disk_d[0] / 1048576.0, disk_d[1] / 1048576.0,
            100.0 * disk_d[1] / 10.50e9);
    fprintf(stderr, "rows  verify_ms (min/med/max)   disk MB/verify   argmax+commit_ms  V/D    a_verify  a_full\n");

    /* Rows above DS4_TP_BATCH_MAX_ROWS are skipped, not clamped: the point of
     * listing them is to see the shape on the far side of the ceiling, and a
     * clamp would silently report the ceiling as if it were the measurement. */
    const uint32_t rows_list[] = {2, 3, 4, 6, 8, 12, 16, 24, 32};
    for (size_t r = 0; r < sizeof(rows_list) / sizeof(rows_list[0]); r++) {
        const uint32_t rows = rows_list[r];
        if (rows > (uint32_t)DS4_TP_BATCH_MAX_ROWS) {
            fprintf(stderr, "%4u  skipped: above DS4_TP_BATCH_MAX_ROWS (%d)\n",
                    rows, (int)DS4_TP_BATCH_MAX_ROWS);
            continue;
        }
        double v[16], h[16];
        int nv = 0, nh = 0;
        double disk_rep[16];
        for (int rep = 0; rep < REPS; rep++) {
            /* DS4_TEST_VERIFY_WINDOW=same verifies one token eight times, which
             * routes every row to the same experts: the floor of a verify's
             * cost, and the test of whether a is expert-union growth or plain
             * per-row traffic. */
            static int same_rows = -1;
            if (same_rows < 0) same_rows = getenv("DS4_TEST_VERIFY_WINDOW") != NULL;
            const int *suffix = same_rows ? prompt.v + start
                                          : prompt.v + start + (rep * 11) % 96;
            REQUIRE(ds4_session_load_snapshot(session, &base, err, sizeof(err)) == 0);
            const uint64_t dk = disk_read_bytes();
            double t = now_sec();
            REQUIRE(ds41_graph_verify(&session->ds41_graph, &engine->model, &engine->weights,
                                      suffix, rows, &spec));
            v[nv++] = (now_sec() - t) * 1000.0;
            disk_rep[rep] = (double)(disk_read_bytes() - dk) / 1048576.0;
            if (rep >= DROP) {
                const float *logits = ds4_gpu_tensor_contents(spec.logits);
                REQUIRE(logits);
                t = now_sec();
                volatile int sink = 0;
                for (uint32_t i = 0; i < rows; i++)
                    sink += sample_argmax(logits + (size_t)i * DS4_N_VOCAB, DS4_N_VOCAB);
                (void)sink;
                if (!ds4_gpu_begin_commands() ||
                    !ds41_spec_commit(&session->ds41_graph, &spec, rows) ||
                    !ds4_gpu_end_commands()) {
                    if (ds4_gpu_commands_active()) (void)ds4_gpu_end_commands();
                    REQUIRE(false);
                }
                h[nh++] = (now_sec() - t) * 1000.0;
            }
        }
        double vmin = 1e30, vmax = 0, dkmed = 0;
        for (int i = DROP; i < nv; i++) { vmin = fmin(vmin, v[i]); vmax = fmax(vmax, v[i]); }
        for (int i = DROP; i < REPS; i++) dkmed += disk_rep[i];
        dkmed /= (double)(REPS - DROP);
        qsort(v + DROP, (size_t)(nv - DROP), sizeof(double), cmp_double);
        qsort(h, (size_t)nh, sizeof(double), cmp_double);
        const double vmed = v[DROP + (nv - DROP) / 2];
        const double hmed = h[nh / 2];
        fprintf(stderr, "%4u  %8.2f (%.2f/%.2f/%.2f)  %13.1f  %14.2f  %5.3f  %8.4f  %7.4f\n",
                rows, vmed, vmin, vmed, vmax, dkmed, hmed, vmed / d_ms,
                (vmed / d_ms - 1.0) / (double)(rows - 1),
                ((vmed + hmed) / d_ms - 1.0) / (double)(rows - 1));
    }
    rc = 0;
done:
    if (err[0]) fprintf(stderr, "error: %s\n", err);
    if (ds4_gpu_commands_active()) (void)ds4_gpu_end_commands();
    ds41_spec_free(&spec);
    ds4_session_snapshot_free(&base);
    ds4_tokens_free(&prefix); ds4_tokens_free(&prompt);
    ds4_session_free(session);
    ds4_engine_close(engine);
    free(text);
    return rc;
}

/* Measurement, not a check: what one decoded token actually reads.
 *
 * The stage timings are only interpretable against the bytes each stage moves,
 * and this GGUF is not uniformly quantized -- routed experts, the shared expert,
 * the attention projections and the head each carry their own mix. Summing the
 * tensor bytes the layer stack and the head touch turns "ms per stage" into
 * "GB/s per stage", which is what separates "the machine is at its limit" from
 * "the kernel is". */
static int weight_inventory(const char *model) {
    ds4_engine *engine = NULL;
    char err[256] = {0};
    int rc = 1;
    ds4_engine_options opt = {.model_path = model, .backend = DS4_BACKEND_METAL,
        .context_size = 4096, .power_percent = 100};
    if (getenv("DS4_TEST_SSD")) {
        opt.ssd_streaming = true;
        opt.ssd_streaming_cache_bytes = ds41_test_streaming_cache_bytes();
        const char *experts = getenv("DS4_TEST_STREAMING_CACHE_EXPERTS");
        opt.ssd_streaming_cache_experts = experts ? (uint32_t)atoi(experts) : 9000u;
    }
    REQUIRE(ds4_engine_open(&engine, &opt) == 0);
    const ds4_weights *w = &engine->weights;

    /* Named explicitly, not by struct offset: the layer struct interleaves the
     * routed and shared expert tensors, so an offset range silently merges
     * them. */
#define SUM_LAYER(field) do { for (uint32_t il = 0; il < DS4_N_LAYER; il++) \
        if (w->layer[il].field) { bytes += w->layer[il].field->bytes; \
                                  elems += w->layer[il].field->elements; } } while (0)
    struct { const char *name; int routed; } groups[] = {
        {"attention projections (q_a, q_b, kv)", 0},
        {"attention output (o_a, o_b)", 0},
        {"indexer + compressor", 0},
        {"shared expert (gate, up, down)", 0},
        {"routed experts (gate, up, down)", 1},
        {"hyper-connection mixes", 0},
        {"norms and router", 0},
    };
    fprintf(stderr, "V4.1 weight inventory (n_layer=%u, n_expert=%u, used=%u)\n",
            (unsigned)DS4_N_LAYER, (unsigned)DS4_N_EXPERT, (unsigned)DS4_N_EXPERT_USED);
    uint64_t per_token = 0, per_token_w = 0;
    for (size_t g = 0; g < sizeof(groups) / sizeof(groups[0]); g++) {
        uint64_t bytes = 0, elems = 0;
        switch (g) {
        case 0:
            SUM_LAYER(attn_q_a); SUM_LAYER(attn_q_a_norm); SUM_LAYER(attn_q_b);
            SUM_LAYER(attn_kv); SUM_LAYER(attn_kv_a_mqa); SUM_LAYER(attn_kv_a_norm);
            SUM_LAYER(attn_k_b); SUM_LAYER(attn_v_b); SUM_LAYER(attn_sinks);
            break;
        case 1:
            SUM_LAYER(attn_output); SUM_LAYER(attn_output_a); SUM_LAYER(attn_output_b);
            break;
        case 2:
            SUM_LAYER(attn_compressor_ape); SUM_LAYER(attn_compressor_kv);
            SUM_LAYER(attn_compressor_gate); SUM_LAYER(attn_compressor_norm);
            SUM_LAYER(indexer_attn_q_b); SUM_LAYER(indexer_attn_k); SUM_LAYER(indexer_k_norm);
            SUM_LAYER(indexer_k_norm_b); SUM_LAYER(indexer_proj);
            SUM_LAYER(indexer_compressor_ape); SUM_LAYER(indexer_compressor_kv);
            SUM_LAYER(indexer_compressor_gate); SUM_LAYER(indexer_compressor_norm);
            break;
        case 3:
            SUM_LAYER(ffn_gate_shexp); SUM_LAYER(ffn_up_shexp); SUM_LAYER(ffn_down_shexp);
            break;
        case 4:
            SUM_LAYER(ffn_gate_exps); SUM_LAYER(ffn_up_exps); SUM_LAYER(ffn_down_exps);
            break;
        case 5:
            SUM_LAYER(hc_attn_fn); SUM_LAYER(hc_attn_scale); SUM_LAYER(hc_attn_base);
            SUM_LAYER(hc_ffn_fn); SUM_LAYER(hc_ffn_scale); SUM_LAYER(hc_ffn_base);
            break;
        default:
            SUM_LAYER(attn_norm); SUM_LAYER(ffn_norm); SUM_LAYER(ffn_gate_inp);
            SUM_LAYER(ffn_exp_probs_b); SUM_LAYER(ffn_exp_probs_vl);
            SUM_LAYER(ffn_gate_tid2eid);
            break;
        }
        fprintf(stderr, "  %-36s %8.3f GiB  %7.3f G weights", groups[g].name,
                (double)bytes / 1073741824.0, (double)elems / 1e9);
        if (groups[g].routed) {
            fprintf(stderr, "  (%.2f MiB/expert, %u of %u per token)",
                    (double)bytes / DS4_N_LAYER / DS4_N_EXPERT / 1048576.0,
                    (unsigned)DS4_N_EXPERT_USED, (unsigned)DS4_N_EXPERT);
            per_token += bytes / DS4_N_EXPERT * DS4_N_EXPERT_USED;
            per_token_w += elems / DS4_N_EXPERT * DS4_N_EXPERT_USED;
        } else {
            per_token += bytes;
            per_token_w += elems;
        }
        fprintf(stderr, "\n");
    }
#undef SUM_LAYER
    /* The embedding table is a row lookup, not a stream: only the head and its
     * norm are read whole per token. */
    uint64_t head = 0, embd = 0, head_w = 0;
    if (w->token_embd) embd += w->token_embd->bytes;
    if (w->output) { head += w->output->bytes; head_w += w->output->elements; }
    if (w->output_norm) { head += w->output_norm->bytes; head_w += w->output_norm->elements; }
    fprintf(stderr, "  %-36s %8.3f GiB  %7.3f G weights\n", "output head + norm",
            (double)head / 1073741824.0, (double)head_w / 1e9);
    fprintf(stderr, "  %-36s %8.3f GiB (row lookup, not streamed)\n",
            "embedding table", (double)embd / 1073741824.0);
    per_token += head;
    per_token_w += head_w;
    fprintf(stderr, "per decoded token: %.2f GiB (%.2f GB), %.3f G weight touches\n",
            (double)per_token / 1073741824.0, (double)per_token / 1e9,
            (double)per_token_w / 1e9);
    rc = 0;
done:
    if (err[0]) fprintf(stderr, "error: %s\n", err);
    ds4_engine_close(engine);
    return rc;
}

/* Measurement: where one decoded token's GPU time goes, from 40 encodes of
 * one layer's own kernels on its real weights. */
static int time_stages(const char *model, const char *prompt_path) {
    ds4_engine *engine = NULL;
    ds4_session *session = NULL;
    ds4_tokens prompt = {0}, prefix = {0};
    char err[256] = {0};
    char *text = read_text(prompt_path);
    int rc = 1;
    ds4_engine_options opt = {.model_path = model, .backend = DS4_BACKEND_METAL,
        .context_size = 4096, .power_percent = 100};
    /* The per-stage split is only worth reading in the regime the engine is
     * deployed in, so honour the same streaming knobs the scan uses. */
    if (getenv("DS4_TEST_SSD")) {
        opt.ssd_streaming = true;
        opt.ssd_streaming_cache_bytes = ds41_test_streaming_cache_bytes();
        const char *experts = getenv("DS4_TEST_STREAMING_CACHE_EXPERTS");
        opt.ssd_streaming_cache_experts = experts ? (uint32_t)atoi(experts) : 9000u;
    }
    REQUIRE(text && ds4_engine_open(&engine, &opt) == 0);
    ds4_tokenize_text(engine, text, &prompt);
    REQUIRE(ds4_session_create(&session, engine, 4096) == 0);
    for (int i = 0; i < 1024; i++) ds4_tokens_push(&prefix, prompt.v[i]);
    REQUIRE(ds4_session_sync(session, &prefix, err, sizeof(err)) == 0);
    for (int i = 0; i < 8; i++)
        REQUIRE(ds4_session_eval(session, prompt.v[1024 + i], err, sizeof(err)) == 0);
    ds41_gpu_graph *g = &session->ds41_graph;
    const ds4_model *m = &engine->model;
    const ds4_weights *w = &engine->weights;
    (void)ds41_publish_measured_config(g, w);
    const char *names[] = {"attention projections (q_a, q_b, kv)", "attention core, index and window",
        "attention output (o_a, o_b)", "shared expert", "router and routed experts",
        "hyper-connection mixes and norms", "output head (once per token)"};
    for (int stage = 0; stage < 7; stage++) {
        double best = 1e30;
        for (int rep = 0; rep < 4; rep++) {
            const uint32_t pos = g->pos;
            REQUIRE(ds4_gpu_begin_commands());
            bool ok = true;
            const int loops = stage == 6 ? 8 : 40;
            for (int i = 0; ok && i < loops; i++) {
                const uint32_t il = stage == 6 ? 0 : (uint32_t)i;
                const ds4_layer_weights *l = &w->layer[il];
                switch (stage) {
                case 0: ok = ds41_attention_project(g, m, l); break;
                case 1: ok = ds41_attention(g, m, l, il, true); break;
                case 2: ok = ds41_attention_output(g, m, l, true); break;
                case 3: ok = ds41_shared_mid(g, m, l) &&
                            ds41_matmul(g->shared, m, l->ffn_down_shexp, g->shared_mid, true); break;
                case 4: {
                    /* Last argument is force_resident: the real call passes
                     * !g->streaming (ds4.c, ds41_moe_partial). Hardcoding true
                     * here forced the whole-map bind that streaming does not
                     * map, which is why this stage used to abort. */
                    uint64_t gate_row = 0, down_row = 0;
                    ok = tensor_nbytes(l->ffn_gate_exps->type, DS4_N_EMBD, &gate_row) &&
                        tensor_nbytes(l->ffn_down_exps->type, DS4_N_FF_EXP, &down_row) &&
                        ds41_matmul(g->route_logits, m, l->ffn_gate_inp, g->norm, false) &&
                        ds4_gpu_router_select_tensor(g->selected, g->route_weights, g->route_probs,
                            m->map, m->size, l->ffn_exp_probs_b->abs_offset, 0, 0, 0,
                            DS4_N_EXPERT, DS4_N_EXPERT_USED, DS4_EXPERT_WEIGHT_SCALE, 0, 0, true, false,
                            g->route_logits) &&
                        ds4_gpu_routed_moe_one_tensor(g->routed, g->gate, g->up, g->mid, g->experts,
                            m->map, m->size, l->ffn_gate_exps->abs_offset, l->ffn_up_exps->abs_offset,
                            l->ffn_down_exps->abs_offset, l->ffn_gate_exps->type, l->ffn_down_exps->type,
                            gate_row * DS4_N_FF_EXP, gate_row, down_row * DS4_N_EMBD, down_row,
                            DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EMBD, g->selected, g->route_weights,
                            DS4_N_EXPERT, DS4_N_EXPERT_USED, DS4_SWIGLU_CLAMP_EXP, g->norm, NULL, il,
                            !g->streaming);
                    break;
                }
                case 5: ok = ds41_hc_mix(g, m, l, false) &&
                            ds41_hc_collapse_norm(g, m, l->attn_norm, g->residual, g->pre) &&
                            ds41_hc_mix(g, m, l, true) &&
                            ds41_hc_collapse_norm(g, m, l->ffn_norm, g->after_attn, g->attn_split); break;
                default: ok = ds41_graph_logits_encode(g, m, w); break;
                }
            }
            const double t = now_sec();
            REQUIRE(ok && ds4_gpu_end_commands());
            const double ms = (now_sec() - t) * 1000.0 / (stage == 6 ? 8.0 : 1.0);
            if (ms < best) best = ms;
            g->pos = pos;
        }
        fprintf(stderr, "%-44s %7.2f ms per token\n", names[stage], best);
    }
    /* How the Q8_0 row kernels scale: 40 layers' q_b projection per launch. */
    for (uint32_t rows = 1; rows <= 8; rows += rows < 2 ? 1 : 2) {
        double best = 1e30;
        for (int rep = 0; rep < 4; rep++) {
            REQUIRE(ds4_gpu_begin_commands());
            bool ok = true;
            for (uint32_t il = 0; ok && il < 40; il++)
                ok = ds41_matmul_batch(g->batch.q, m, w->layer[il].attn_q_b, g->batch.qr, rows, false);
            const double t = now_sec();
            REQUIRE(ok && ds4_gpu_end_commands());
            if ((now_sec() - t) * 1000.0 < best) best = (now_sec() - t) * 1000.0;
        }
        fprintf(stderr, "q_b projection, %u rows: %7.2f ms for 40 layers (%.2f ms per row)\n",
                rows, best, best / rows);
    }
    rc = 0;
done:
    if (err[0]) fprintf(stderr, "error: %s\n", err);
    if (ds4_gpu_commands_active()) ds4_gpu_end_commands();
    ds4_tokens_free(&prefix); ds4_tokens_free(&prompt);
    ds4_session_free(session);
    ds4_engine_close(engine);
    free(text);
    return rc;
}

/* HANDOFF-V41-STREAMING-SMALL-PREFILL 12.5. With one layer's routed experts
 * mapped into the streaming model view, run the batch MoE twice on identical
 * activations, selected ids and route weights: once bound through the whole
 * expert map (force_resident) and once through the streaming selected-address
 * gather. The legs differ only in where the expert bytes come from, so a
 * bit-exact verdict isolates the address-table bind from every shared batch
 * operator upstream. */
static uint32_t bind_rng_state = 0x9e3779b9u;
static uint32_t bind_rng(void) {
    bind_rng_state ^= bind_rng_state << 13;
    bind_rng_state ^= bind_rng_state >> 17;
    bind_rng_state ^= bind_rng_state << 5;
    return bind_rng_state;
}

static int check_moe_bind_parity(const char *model) {
    ds4_engine *engine = NULL;
    ds4_gpu_tensor *x = NULL, *selected = NULL, *weights = NULL;
    ds4_gpu_tensor *a_gate = NULL, *a_up = NULL, *a_mid = NULL, *a_experts = NULL, *a_out = NULL;
    ds4_gpu_tensor *b_gate = NULL, *b_up = NULL, *b_mid = NULL, *b_experts = NULL, *b_out = NULL;
    float *host_x = NULL, *host_w = NULL, *host_a = NULL, *host_b = NULL;
    int32_t *host_ids = NULL;
    char err[256] = {0};
    int rc = 1;
    ds4_engine_options opt = {.model_path = model, .backend = DS4_BACKEND_METAL,
        .context_size = 4096, .power_percent = 100, .ssd_streaming = true,
        .ssd_streaming_cache_bytes = ds41_test_streaming_cache_bytes()};
    REQUIRE(ds4_engine_open(&engine, &opt) == 0);
    const ds4_model *m = &engine->model;
    const ds4_weights *w = &engine->weights;
    const uint32_t max_rows = 1024u, max_pairs = max_rows * DS4_N_EXPERT_USED;
#define BIND_TENSOR(name, floats) \
    name = ds4_gpu_tensor_alloc((uint64_t)(floats) * sizeof(float)); \
    REQUIRE(name);
    BIND_TENSOR(x, (uint64_t)max_rows * DS4_N_EMBD)
    BIND_TENSOR(selected, max_pairs)
    BIND_TENSOR(weights, max_pairs)
    BIND_TENSOR(a_gate, (uint64_t)max_pairs * DS4_N_FF_EXP)
    BIND_TENSOR(a_up, (uint64_t)max_pairs * DS4_N_FF_EXP)
    BIND_TENSOR(a_mid, (uint64_t)max_pairs * DS4_N_FF_EXP)
    BIND_TENSOR(a_experts, (uint64_t)max_pairs * DS4_N_EMBD)
    BIND_TENSOR(a_out, (uint64_t)max_rows * DS4_N_EMBD)
    BIND_TENSOR(b_gate, (uint64_t)max_pairs * DS4_N_FF_EXP)
    BIND_TENSOR(b_up, (uint64_t)max_pairs * DS4_N_FF_EXP)
    BIND_TENSOR(b_mid, (uint64_t)max_pairs * DS4_N_FF_EXP)
    BIND_TENSOR(b_experts, (uint64_t)max_pairs * DS4_N_EMBD)
    BIND_TENSOR(b_out, (uint64_t)max_rows * DS4_N_EMBD)
#undef BIND_TENSOR
    host_x = malloc((size_t)max_rows * DS4_N_EMBD * sizeof(float));
    host_w = malloc(max_pairs * sizeof(float));
    host_ids = malloc(max_pairs * sizeof(int32_t));
    host_a = malloc((size_t)max_rows * DS4_N_EMBD * sizeof(float));
    host_b = malloc((size_t)max_rows * DS4_N_EMBD * sizeof(float));
    REQUIRE(host_x && host_w && host_ids && host_a && host_b);
    const uint32_t layers[] = {0, 20, 39};
    /* Sweep past the family thresholds: mul_mv tiny pair (2..8), plain
     * mm-id TensorOps (32..511), packed TensorOps mm-id (512..1024).
     * 511/512 straddle the packed switch; 831/1024 are the live sweep
     * append sizes and the selected-addr auto_max cap (HANDOFF-V41 13.4). */
    const uint32_t row_sets[] = {2, 5, 8, 9, 16, 31, 32, 256, 511, 512, 831, 1024};
    int drifts = 0;
    for (size_t li = 0; li < sizeof(layers) / sizeof(layers[0]); li++) {
        const uint32_t il = layers[li];
        REQUIRE(metal_graph_stream_map_layer(m, w, il));
        const ds4_layer_weights *l = &w->layer[il];
        REQUIRE(l->ffn_gate_exps->type == DS4_TENSOR_IQ2_XXS &&
                l->ffn_down_exps->type == DS4_TENSOR_Q2_K);
        const uint64_t gate_row = routed_expert_row_bytes(l->ffn_gate_exps);
        const uint64_t down_row = routed_expert_row_bytes(l->ffn_down_exps);
        REQUIRE(gate_row && down_row);
        for (size_t ri = 0; ri < sizeof(row_sets) / sizeof(row_sets[0]); ri++) {
            const uint32_t rows = row_sets[ri];
            const uint64_t pairs = (uint64_t)rows * DS4_N_EXPERT_USED;
            /* Distinct selected ids within each token (routing never repeats
             * an expert for one token); across tokens ids repeat freely once
             * the batch pairs exceed the expert count, so large sweeps give
             * every expert many rows exactly as production routing does. */
            for (uint32_t t = 0; t < rows; t++) {
                bool seen[DS4_N_EXPERT];
                memset(seen, 0, sizeof(seen));
                for (uint32_t e = 0; e < DS4_N_EXPERT_USED; e++) {
                    const uint64_t p = (uint64_t)t * DS4_N_EXPERT_USED + e;
                    uint32_t id;
                    do { id = bind_rng() % DS4_N_EXPERT; } while (seen[id]);
                    seen[id] = true;
                    host_ids[p] = (int32_t)id;
                    host_w[p] = 0.05f + (float)(bind_rng() % 1000u) / 1000.0f;
                }
            }
            for (uint64_t i = 0; i < (uint64_t)rows * DS4_N_EMBD; i++)
                host_x[i] = ((float)(int32_t)(bind_rng() >> 8) - 8388608.0f) / 8388608.0f;
            const uint64_t out_bytes = (uint64_t)rows * DS4_N_EMBD * sizeof(float);
            bool f16_a = false, f16_b = false, f16_ok = false;
#define BIND_ZERO(gate, up, mid, experts, out) \
            (ds4_gpu_tensor_fill_f32((gate), 0.f, (uint64_t)pairs * DS4_N_FF_EXP) && \
             ds4_gpu_tensor_fill_f32((up), 0.f, (uint64_t)pairs * DS4_N_FF_EXP) && \
             ds4_gpu_tensor_fill_f32((mid), 0.f, (uint64_t)pairs * DS4_N_FF_EXP) && \
             ds4_gpu_tensor_fill_f32((experts), 0.f, (uint64_t)pairs * DS4_N_EMBD) && \
             ds4_gpu_tensor_fill_f32((out), 0.f, (uint64_t)rows * DS4_N_EMBD))
#define BIND_REWRITE_INPUTS \
            (ds4_gpu_tensor_write(x, 0, host_x, \
                        (uint64_t)rows * DS4_N_EMBD * sizeof(float)) && \
             ds4_gpu_tensor_write(selected, 0, host_ids, pairs * sizeof(int32_t)) && \
             ds4_gpu_tensor_write(weights, 0, host_w, pairs * sizeof(float)))
            REQUIRE(BIND_ZERO(a_gate, a_up, a_mid, a_experts, a_out));
            REQUIRE(BIND_REWRITE_INPUTS);
            REQUIRE(ds4_gpu_begin_commands());
            const bool ok_a = ds4_gpu_routed_moe_batch_tensor(a_out, a_gate, a_up, a_mid, a_experts,
                m->map, m->size, l->ffn_gate_exps->abs_offset, l->ffn_up_exps->abs_offset,
                l->ffn_down_exps->abs_offset, l->ffn_gate_exps->type, l->ffn_down_exps->type,
                gate_row * DS4_N_FF_EXP, gate_row, down_row * DS4_N_EMBD, down_row,
                DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EMBD, selected, weights,
                DS4_N_EXPERT, DS4_N_EXPERT_USED, DS4_SWIGLU_CLAMP_EXP, x, il, rows,
                &f16_a, true);
            REQUIRE(ok_a && ds4_gpu_end_commands());
            REQUIRE(BIND_ZERO(b_gate, b_up, b_mid, b_experts, b_out));
            REQUIRE(BIND_REWRITE_INPUTS);
            REQUIRE(ds4_gpu_begin_commands());
            const bool ok_b = ds4_gpu_routed_moe_batch_tensor(b_out, b_gate, b_up, b_mid, b_experts,
                m->map, m->size, l->ffn_gate_exps->abs_offset, l->ffn_up_exps->abs_offset,
                l->ffn_down_exps->abs_offset, l->ffn_gate_exps->type, l->ffn_down_exps->type,
                gate_row * DS4_N_FF_EXP, gate_row, down_row * DS4_N_EMBD, down_row,
                DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EMBD, selected, weights,
                DS4_N_EXPERT, DS4_N_EXPERT_USED, DS4_SWIGLU_CLAMP_EXP, x, il, rows,
                &f16_b, false);
            REQUIRE(ok_b && ds4_gpu_end_commands());
            f16_ok = (f16_a == f16_b);
            REQUIRE(f16_ok);
            REQUIRE(ds4_gpu_tensor_read(a_out, 0, host_a, out_bytes) &&
                    ds4_gpu_tensor_read(b_out, 0, host_b, out_bytes));
            if (memcmp(host_a, host_b, out_bytes) == 0) {
                printf("layer %u rows %u: address-table bind bit-exact vs whole-map bind\n",
                       il, rows);
            } else {
                float worst = 0.0f;
                for (uint64_t i = 0; i < (uint64_t)rows * DS4_N_EMBD; i++) {
                    const float d = host_a[i] > host_b[i] ? host_a[i] - host_b[i]
                                                          : host_b[i] - host_a[i];
                    if (d > worst) worst = d;
                }
                printf("layer %u rows %u: address-table bind DRIFTS vs whole-map bind (worst %g)\n",
                       il, rows, worst);
                drifts++;
            }
#undef BIND_ZERO
        }
    }
    if (drifts) {
        fprintf(stderr, "V4.1 MoE bind parity: %d case(s) drift -- the gather kernel "
                        "itself changes the bits\n", drifts);
        goto done;
    }
    printf("V4.1 MoE bind parity: address-table gather bit-exact at every layer and size\n");
    rc = 0;
done:
    if (err[0]) fprintf(stderr, "error: %s\n", err);
    if (ds4_gpu_commands_active()) ds4_gpu_end_commands();
    free(host_x); free(host_w); free(host_ids); free(host_a); free(host_b);
    ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(selected); ds4_gpu_tensor_free(weights);
    ds4_gpu_tensor_free(a_gate); ds4_gpu_tensor_free(a_up); ds4_gpu_tensor_free(a_mid);
    ds4_gpu_tensor_free(a_experts); ds4_gpu_tensor_free(a_out);
    ds4_gpu_tensor_free(b_gate); ds4_gpu_tensor_free(b_up); ds4_gpu_tensor_free(b_mid);
    ds4_gpu_tensor_free(b_experts); ds4_gpu_tensor_free(b_out);
    ds4_engine_close(engine);
    return rc;
}

int main(int argc, char **argv) {
    if (argc == 4 && !strcmp(argv[2], "--stage-timing"))
        return time_stages(argv[1], argv[3]);
    if (argc == 4 && !strcmp(argv[2], "--verify-timing"))
        return time_verify(argv[1], argv[3]);
    if (argc == 4 && !strcmp(argv[2], "--verify-scan"))
        return time_verify_scan(argv[1], argv[3], false);
    if (argc == 4 && !strcmp(argv[2], "--verify-scan-ssd"))
        return time_verify_scan(argv[1], argv[3], true);
    if (argc == 3 && !strcmp(argv[2], "--weight-inventory"))
        return weight_inventory(argv[1]);
    if (argc == 4 && !strcmp(argv[2], "--short-prefill"))
        return check_short_prefill(argv[1], argv[3], false);
    if (argc == 4 && !strcmp(argv[2], "--short-prefill-ssd"))
        return check_short_prefill(argv[1], argv[3], true);
    if (argc == 4 && !strcmp(argv[2], "--short-prefill-ssd-rows"))
        return check_short_prefill_rows(argv[1], argv[3]);
    if (argc == 4 && !strcmp(argv[2], "--verify-parity"))
        return check_verify_parity(argv[1], argv[3], false);
    if (argc == 4 && !strcmp(argv[2], "--verify-parity-ssd"))
        return check_verify_parity(argv[1], argv[3], true);
    if (argc == 3 && !strcmp(argv[2], "--moe-bind-parity"))
        return check_moe_bind_parity(argv[1]);
    fprintf(stderr, "usage: %s MODEL --verify-parity|--verify-parity-ssd"
                    "|--short-prefill|--short-prefill-ssd|--short-prefill-ssd-rows"
                    "|--verify-timing|--verify-scan|--verify-scan-ssd PROMPT\n", argv[0]);
    return 2;
}
