/* DSpark verify/decode logit parity harness (HANDOFF-DSPARK-20261019 §7.1).
 *
 * Compares a multi-row V4.1 verification block against the equivalent
 * one-token-at-a-time decode from the same prompt frontier, on the exact same
 * token sequence.  The tokens are the reference argmax chain, so this isolates
 * verification arithmetic from drafter quality: if the drafter were perfect,
 * verify rows and decode rows would see byte-identical inputs.
 *
 *   tests/test_dspark_verify_parity --lengths 2048,4096,8192 [--rows 5]
 *                                   [--fresh] [--repeat]
 *
 * --lengths   prompt token counts to probe (ascending; each is a case)
 * --rows      verification block size, 2..DS41_VERIFY_ROWS (default 5)
 * --fresh     rebuild both sessions per length instead of rewinding between
 *             cases (slow, but free of any state-restore assumption)
 * --repeat    run each case twice and compare, proving rewind transparency
 *
 * Env: DS4_TEST_MODEL, DS4_TEST_DSPARK, DS4_TEST_SSD_STREAMING,
 *      DS4_TEST_SSD_STREAMING_CACHE_EXPERTS.
 */
#include "../ds4.c"

typedef struct {
    float max_abs;      /* max |a - b| over the vocabulary */
    int   max_at;       /* token holding that maximum */
    bool  bitwise;      /* every float identical, bit for bit */
    int   argmax_a, argmax_b;
    float top1_a, top2_a;
    float delta_at_b_argmax;
} parity_cmp;

static parity_cmp parity_compare(const float *a, const float *b, uint32_t n) {
    parity_cmp c;
    memset(&c, 0, sizeof(c));
    c.bitwise = true;
    c.max_at = -1;
    c.argmax_a = c.argmax_b = -1;
    c.top1_a = c.top2_a = -INFINITY;
    float best_b = -INFINITY;
    bool have1 = false, have2 = false;
    for (uint32_t i = 0; i < n; i++) {
        const float d = fabsf(a[i] - b[i]);
        if (c.max_at < 0 || d > c.max_abs) { c.max_abs = d; c.max_at = (int)i; }
        if (memcmp(&a[i], &b[i], sizeof(float)) != 0) c.bitwise = false;
        if (!have1 || a[i] > c.top1_a) {
            if (have1) { c.top2_a = c.top1_a; have2 = true; }
            c.top1_a = a[i];
            c.argmax_a = (int)i;
            have1 = true;
        } else if (!have2 || a[i] > c.top2_a) {
            c.top2_a = a[i];
            have2 = true;
        }
        if (c.argmax_b < 0 || b[i] > best_b) { best_b = b[i]; c.argmax_b = (int)i; }
    }
    c.delta_at_b_argmax = a[c.argmax_b] - b[c.argmax_b];
    return c;
}

static int parity_argmax(const float *v, uint32_t n) {
    int best = -1;
    float bv = -INFINITY;
    for (uint32_t i = 0; i < n; i++)
        if (best < 0 || v[i] > bv) { bv = v[i]; best = (int)i; }
    return best;
}

static ds4_engine *parity_open_engine(void) {
    const char *model = getenv("DS4_TEST_MODEL");
    const char *support = getenv("DS4_TEST_DSPARK");
    if (!model || !model[0] || !support || !support[0]) {
        fprintf(stderr, "dspark-parity: set DS4_TEST_MODEL and DS4_TEST_DSPARK\n");
        return NULL;
    }
    ds4_engine *engine = NULL;
    ds4_engine_options opt = {
        .model_path = model,
#ifdef __APPLE__
        .backend = DS4_BACKEND_METAL,
#else
        .backend = DS4_BACKEND_CUDA,
#endif
        .quality = false,
        .prefill_chunk = 512,
        .ssd_streaming = getenv("DS4_TEST_SSD_STREAMING") && getenv("DS4_TEST_SSD_STREAMING")[0] &&
                         strcmp(getenv("DS4_TEST_SSD_STREAMING"), "0") != 0,
        .ssd_streaming_cache_experts = 0,
        .mtp_path = support,
        .mtp_draft_tokens = 0,
        .dspark = true,
        .dspark_confidence_threshold = 0.9f,
        .dspark_confidence_threshold_set = true,
    };
    const char *cache = getenv("DS4_TEST_SSD_STREAMING_CACHE_EXPERTS");
    if (cache && cache[0]) opt.ssd_streaming_cache_experts = (uint32_t)strtoul(cache, NULL, 10);
    if (ds4_engine_open(&engine, &opt) != 0) {
        fprintf(stderr, "dspark-parity: engine open failed\n");
        return NULL;
    }
    return engine;
}

static bool parity_sync(ds4_session *s, const ds4_tokens *prompt, int len, const char *tag) {
    char err[256];
    ds4_tokens view = *prompt;
    view.len = len;
    if (ds4_session_sync(s, &view, err, sizeof(err)) != 0) {
        fprintf(stderr, "dspark-parity: %s sync(%d) failed: %s\n", tag, len, err);
        return false;
    }
    if (ds4_session_pos(s) != len) {
        fprintf(stderr, "dspark-parity: %s sync(%d) landed at pos=%d\n", tag, len, ds4_session_pos(s));
        return false;
    }
    return true;
}

/* One case: from a shared frontier at `len`, decode `rows` tokens one at a time
 * on `dec` and run the same `rows` tokens as one verification block on `ver`.
 *
 * Row convention (ds41_session_speculate): verified row i carries the logits
 * after token i, i.e. it predicts tokens[i+1].  So decode row i+1 must match
 * verified row i; the frontier distribution itself is what seeds tokens[0].
 *
 * Returns the number of verified rows whose logits are not bit-identical. */
static int parity_case(ds4_session *dec, ds4_session *ver, int len, int rows,
                       const char *label) {
    const uint32_t nv = DS4_N_VOCAB;
    float *ref[DS41_VERIFY_ROWS + 1];   /* rows + 1: one extra frontier step */
    float *got_block = NULL;            /* the hook writes `rows` rows contiguously */
    float *got[DS41_VERIFY_ROWS];
    int tokens[DS41_VERIFY_ROWS];
    char err[256];
    int mismatched = 0;

    memset(ref, 0, sizeof(ref));
    got_block = malloc((size_t)rows * nv * sizeof(float));
    if (!got_block) {
        fprintf(stderr, "dspark-parity: out of memory\n");
        return -1;
    }
    for (int i = 0; i <= rows; i++) {
        ref[i] = malloc((size_t)nv * sizeof(float));
        if (!ref[i]) {
            fprintf(stderr, "dspark-parity: out of memory\n");
            goto done;
        }
        if (i < rows) got[i] = got_block + (size_t)i * nv;
    }

    /* Oracle chain: the exact tokens a perfect drafter would have proposed. */
    for (int i = 0; i < rows; i++) {
        if (ds4_session_copy_logits(dec, ref[i], (int)nv) <= 0) {
            fprintf(stderr, "dspark-parity: copy_logits(%d) failed\n", i);
            goto done;
        }
        tokens[i] = parity_argmax(ref[i], nv);
        if (ds4_session_eval(dec, tokens[i], err, sizeof(err)) != 0) {
            fprintf(stderr, "dspark-parity: decode(%d) failed: %s\n", i, err);
            goto done;
        }
    }
    if (ds4_session_copy_logits(dec, ref[rows], (int)nv) <= 0) {
        fprintf(stderr, "dspark-parity: copy_logits(%d) failed\n", rows);
        goto done;
    }
    if (ds4_test_ds41_verify_tokens(ver, tokens, rows, got_block, err, sizeof(err)) != 0) {
        fprintf(stderr, "dspark-parity: verify(%d) failed: %s\n", rows, err);
        goto done;
    }

    for (int i = 0; i < rows; i++) {
        const parity_cmp c = parity_compare(ref[i + 1], got[i], nv);
        const float margin = c.top1_a - c.top2_a;
        if (!c.bitwise) mismatched++;
        printf("%s len=%d row=%d %s dmax=%.6g at=%d argmax=%d/%d%s margin=%.6g top1=%.6g "
               "d@argmax=%.6g consumed=%d\n",
               label, len, i, c.bitwise ? "EXACT" : "DIFF ",
               (double)c.max_abs, c.max_at, c.argmax_a, c.argmax_b,
               c.argmax_a == c.argmax_b ? "" : " (FLIP)",
               (double)margin, (double)c.top1_a, (double)c.delta_at_b_argmax, tokens[i]);
    }

done:
    for (int i = 0; i <= rows; i++) { free(ref[i]); ref[i] = NULL; }
    free(got_block);
    return mismatched;
}


/* ---- hidden-state diff ----------------------------------------------------
 * The frontier logits only exercise the last row.  Compare everything the next
 * step can read: the raw ring, the compressed/index caches, and the ratio-2
 * pair carry.  A difference here with identical frontier logits means the two
 * sessions diverged before any verification happened. */
static void state_cmp_tensor(const char *name, ds4_gpu_tensor *ta, ds4_gpu_tensor *tb,
                             uint64_t floats) {
    if (!ta || !tb) { printf("  %-28s missing\n", name); return; }
    float *x = malloc((size_t)floats * sizeof(float));
    float *y = malloc((size_t)floats * sizeof(float));
    if (!x || !y || !ds4_gpu_tensor_read(ta, 0, x, floats * sizeof(float)) ||
        !ds4_gpu_tensor_read(tb, 0, y, floats * sizeof(float))) {
        printf("  %-28s unreadable\n", name);
        free(x); free(y);
        return;
    }
    uint64_t bad = 0, first = 0;
    float dmax = 0.0f;
    for (uint64_t i = 0; i < floats; i++) {
        if (memcmp(&x[i], &y[i], sizeof(float)) != 0) {
            if (!bad) first = i;
            bad++;
            const float d = fabsf(x[i] - y[i]);
            if (d > dmax) dmax = d;
        }
    }
    if (bad) printf("  %-28s DIFF n=%llu first=%llu dmax=%g\n", name,
                    (unsigned long long)bad, (unsigned long long)first, (double)dmax);
    else printf("  %-28s ok (%llu)\n", name, (unsigned long long)floats);
    free(x); free(y);
}

static void parity_state_diff(ds4_session *a, ds4_session *b, int len) {
    ds41_gpu_graph *ga = &a->ds41_graph, *gb = &b->ds41_graph;
    printf("state len=%d pos=%u/%u valid=%d/%d raw_log_from=%u/%u hist_tail=%d,%d,%d/%d,%d,%d\n",
           len, ga->pos, gb->pos, ga->valid, gb->valid,
           ga->raw_log_from, gb->raw_log_from,
           ga->history.tail[0], ga->history.tail[1], ga->history.tail[2],
           gb->history.tail[0], gb->history.tail[1], gb->history.tail[2]);
    char name[48];
    for (unsigned owner = 0; owner < 4; owner++) {
        snprintf(name, sizeof(name), "previous_kv[%u]", owner);
        state_cmp_tensor(name, ga->previous_kv[owner], gb->previous_kv[owner], DS4_N_HEAD_DIM);
        snprintf(name, sizeof(name), "previous_score[%u]", owner);
        state_cmp_tensor(name, ga->previous_score[owner], gb->previous_score[owner], DS4_N_HEAD_DIM);
        const uint32_t ratio = owner == 3 ? 1u : 2u;
        const uint64_t n_comp = (uint64_t)(ga->pos + 1u) / ratio;
        snprintf(name, sizeof(name), "compressed[%u] n=%llu", owner, (unsigned long long)n_comp);
        state_cmp_tensor(name, ga->compressed[owner], gb->compressed[owner], n_comp * DS4_N_HEAD_DIM);
        snprintf(name, sizeof(name), "index_cache[%u] n=%llu", owner, (unsigned long long)n_comp);
        state_cmp_tensor(name, ga->index_cache[owner], gb->index_cache[owner],
                         n_comp * DS4_N_INDEXER_HEAD_DIM);
    }
    for (unsigned il = 0; il < DS4_N_LAYER; il++) {
        snprintf(name, sizeof(name), "window[%u]", il);
        state_cmp_tensor(name, ga->window[il], gb->window[il], 128u * DS4_N_HEAD_DIM);
    }
}


/* ---- activation trace ----------------------------------------------------
 * Decode one token and verify a two-row block from the same frontier, probing
 * every layer's pre-attention activations and output residual.  The first
 * layer whose row 0 differs localizes the divergence. */
static float *trace_read(ds4_gpu_tensor *t, uint64_t floats) {
    float *buf = malloc((size_t)floats * sizeof(float));
    if (!buf || !ds4_gpu_tensor_read(t, 0, buf, floats * sizeof(float))) {
        free(buf);
        return NULL;
    }
    return buf;
}

static int trace_compare(const char *name, const float *a, const float *b,
                         uint32_t width, float *first_dmax_out) {
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const uint64_t off = ((uint64_t)il * ds4_test_probe_rows + 0) * width;
        uint64_t bad = 0;
        float dmax = 0.0f;
        uint64_t at = 0;
        for (uint32_t k = 0; k < width; k++) {
            if (memcmp(&a[off + k], &b[off + k], sizeof(float)) != 0) {
                if (!bad) at = k;
                bad++;
                const float d = fabsf(a[off + k] - b[off + k]);
                if (d > dmax) dmax = d;
            }
        }
        if (bad) {
            printf("  %-6s layer=%2u DIFF n=%llu first=%llu dmax=%g\n", name, il,
                   (unsigned long long)bad, (unsigned long long)at, (double)dmax);
            if (first_dmax_out) *first_dmax_out = dmax;
            return 1;
        }
    }
    printf("  %-6s all %u layers exact\n", name, DS4_N_LAYER);
    return 0;
}

static void parity_usage(const char *prog) {
    fprintf(stderr, "usage: %s --lengths N[,N...] [--rows R] [--fresh] [--repeat]\n"
                    "          [--sweep N]\n\n"
                    "  compares a R-row V4.1 verification block against R one-token\n"
                    "  decodes of the same oracle chain, at each prompt length.\n"
                    "  --sweep N runs N consecutive blocks from the first length without\n"
                    "  rewinding, which walks every position in the range.\n",
            prog);
}

int main(int argc, char **argv) {
    const char *lengths_arg = NULL;
    const char *prompt_file = "speed-bench/promessi_sposi.txt";
    int rows = 5;
    int sweep = 0;
    bool state_only = false;
    bool trace_only = false;
    bool fresh = false, repeat = false;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--lengths") && i + 1 < argc) lengths_arg = argv[++i];
        else if (!strcmp(argv[i], "--prompt-file") && i + 1 < argc) prompt_file = argv[++i];
        else if (!strcmp(argv[i], "--rows") && i + 1 < argc) rows = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--fresh")) fresh = true;
        else if (!strcmp(argv[i], "--repeat")) repeat = true;
        else if (!strcmp(argv[i], "--sweep") && i + 1 < argc) sweep = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--state")) state_only = true;
        else if (!strcmp(argv[i], "--trace")) trace_only = true;
        else { parity_usage(argv[0]); return 2; }
    }
    if (!lengths_arg) { parity_usage(argv[0]); return 2; }
    if (rows < 2 || (uint32_t)rows > DS41_VERIFY_ROWS) {
        fprintf(stderr, "dspark-parity: --rows must be 2..%u\n", DS41_VERIFY_ROWS);
        return 2;
    }

    int lengths[64];
    int nlengths = 0;
    for (const char *p = lengths_arg; *p && nlengths < 64; ) {
        char *end = NULL;
        const long v = strtol(p, &end, 10);
        if (end == p || v < 8) { fprintf(stderr, "dspark-parity: bad --lengths\n"); return 2; }
        lengths[nlengths++] = (int)v;
        p = end;
        while (*p == ',' || *p == ' ') p++;
    }
    /* Ascending order keeps the incremental sync path in play. */
    for (int i = 0; i < nlengths; i++)
        for (int j = i + 1; j < nlengths; j++)
            if (lengths[j] < lengths[i]) { int t = lengths[i]; lengths[i] = lengths[j]; lengths[j] = t; }

    FILE *fp = fopen(prompt_file, "rb");
    if (!fp) { fprintf(stderr, "dspark-parity: cannot open %s\n", prompt_file); return 2; }
    fseek(fp, 0, SEEK_END);
    const long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char *text = malloc((size_t)size + 1);
    const bool read_ok = text && fread(text, 1, (size_t)size, fp) == (size_t)size;
    fclose(fp);
    if (!read_ok) { free(text); fprintf(stderr, "dspark-parity: cannot read %s\n", prompt_file); return 2; }
    text[size] = '\0';

    ds4_engine *engine = parity_open_engine();
    if (!engine) { free(text); return 1; }

    ds4_tokens prompt = {0};
    ds4_tokenize_text(engine, text, &prompt);
    free(text);
    if (prompt.len < lengths[nlengths - 1]) {
        fprintf(stderr, "dspark-parity: prompt tokenizes to %d, need %d\n",
                prompt.len, lengths[nlengths - 1]);
        ds4_tokens_free(&prompt);
        ds4_engine_close(engine);
        return 2;
    }

    const int ctx = lengths[nlengths - 1] + (sweep > 0 ? sweep * rows : 0) + rows + 8;
    ds4_session *dec = NULL, *ver = NULL;
    if (ds4_session_create(&dec, engine, ctx) != 0 || ds4_session_create(&ver, engine, ctx) != 0) {
        fprintf(stderr, "dspark-parity: session create failed\n");
        return 1;
    }
    printf("dspark-parity: rows=%d ctx=%d lengths=%d model=%s\n",
           rows, ctx, nlengths, getenv("DS4_TEST_MODEL"));

    if (trace_only) {
        const int len = lengths[0];
        const uint64_t qf = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
        const uint64_t slots = (uint64_t)DS4_N_LAYER * ds4_test_probe_rows;
        ds4_test_probe_norm = ds4_gpu_tensor_alloc(slots * DS4_N_EMBD * sizeof(float));
        ds4_test_probe_q = ds4_gpu_tensor_alloc(slots * qf * sizeof(float));
        ds4_test_probe_kv = ds4_gpu_tensor_alloc(slots * DS4_N_HEAD_DIM * sizeof(float));
        ds4_test_probe_res = ds4_gpu_tensor_alloc(slots * (uint64_t)DS4_N_HC * DS4_N_EMBD * sizeof(float));
        if (!ds4_test_probe_norm || !ds4_test_probe_q || !ds4_test_probe_kv || !ds4_test_probe_res) {
            fprintf(stderr, "dspark-parity: probe alloc failed\n");
            return 1;
        }
        if (!parity_sync(dec, &prompt, len, "dec") || !parity_sync(ver, &prompt, len, "ver")) return 1;

        float *ref = malloc((size_t)DS4_N_VOCAB * sizeof(float));
        if (!ref || ds4_session_copy_logits(dec, ref, (int)DS4_N_VOCAB) <= 0) return 1;
        const int t0 = parity_argmax(ref, DS4_N_VOCAB);
        free(ref);
        char err[256];

        ds4_test_probe_row = 0;
        if (ds4_session_eval(dec, t0, err, sizeof(err)) != 0) {
            fprintf(stderr, "dspark-parity: decode failed: %s\n", err);
            return 1;
        }
        (void)ds4_gpu_synchronize();
        float *dec_norm = trace_read(ds4_test_probe_norm, slots * DS4_N_EMBD);
        float *dec_q = trace_read(ds4_test_probe_q, slots * qf);
        float *dec_kv = trace_read(ds4_test_probe_kv, slots * DS4_N_HEAD_DIM);
        float *dec_res = trace_read(ds4_test_probe_res, slots * (uint64_t)DS4_N_HC * DS4_N_EMBD);
        if (!dec_norm || !dec_q || !dec_kv || !dec_res) {
            fprintf(stderr, "dspark-parity: probe read failed\n");
            return 1;
        }

        int tokens[2] = {t0, 0};
        float *got = malloc(2 * (size_t)DS4_N_VOCAB * sizeof(float));
        if (!got) return 1;
        /* next token from the decode row seeds the second verify row */
        {
            float *l1 = malloc((size_t)DS4_N_VOCAB * sizeof(float));
            if (!l1 || ds4_session_copy_logits(dec, l1, (int)DS4_N_VOCAB) <= 0) return 1;
            tokens[1] = parity_argmax(l1, DS4_N_VOCAB);
            free(l1);
        }
        if (ds4_test_ds41_verify_tokens(ver, tokens, 2, got, err, sizeof(err)) != 0) {
            fprintf(stderr, "dspark-parity: verify failed: %s\n", err);
            return 1;
        }
        (void)ds4_gpu_synchronize();
        float *ver_norm = trace_read(ds4_test_probe_norm, slots * DS4_N_EMBD);
        float *ver_q = trace_read(ds4_test_probe_q, slots * qf);
        float *ver_kv = trace_read(ds4_test_probe_kv, slots * DS4_N_HEAD_DIM);
        float *ver_res = trace_read(ds4_test_probe_res, slots * (uint64_t)DS4_N_HC * DS4_N_EMBD);
        if (!ver_norm || !ver_q || !ver_kv || !ver_res) {
            fprintf(stderr, "dspark-parity: probe read failed\n");
            return 1;
        }
        printf("trace len=%d t0=%d t1=%d (row 0: decode vs verify)\n", len, t0, tokens[1]);
        int bad = 0;
        bad += trace_compare("norm", dec_norm, ver_norm, DS4_N_EMBD, NULL);
        bad += trace_compare("q", dec_q, ver_q, (uint32_t)qf, NULL);
        bad += trace_compare("kv", dec_kv, ver_kv, DS4_N_HEAD_DIM, NULL);
        bad += trace_compare("res", dec_res, ver_res, DS4_N_HC * DS4_N_EMBD, NULL);
        printf("dspark-parity: %s\n", bad ? "PARITY BROKEN" : "PARITY OK");
        return bad ? 1 : 0;
    }
    if (state_only) {
        int bad = 0;
        for (int ci = 0; ci < nlengths; ci++) {
            const int len = lengths[ci];
            if (fresh) {
                ds4_session_free(dec); ds4_session_free(ver);
                dec = ver = NULL;
                if (ds4_session_create(&dec, engine, ctx) != 0 ||
                    ds4_session_create(&ver, engine, ctx) != 0) return 1;
            }
            if (!parity_sync(dec, &prompt, len, "dec") || !parity_sync(ver, &prompt, len, "ver")) {
                bad = -1; break;
            }
            (void)ds4_gpu_synchronize();
            parity_state_diff(dec, ver, len);
            if (!fresh) {
                ds4_session_rewind(dec, len);
                ds4_session_rewind(ver, len);
            }
        }
        ds4_session_free(dec); ds4_session_free(ver);
        ds4_tokens_free(&prompt);
        ds4_engine_close(engine);
        return bad == 0 ? 0 : 1;
    }
    int bad_total = 0;
    if (sweep > 0) {
        const int start = lengths[0];
        if (!parity_sync(dec, &prompt, start, "dec") || !parity_sync(ver, &prompt, start, "ver")) {
            bad_total = -1;
        }
        for (int k = 0; bad_total == 0 && k < sweep; k++) {
            char label[32];
            snprintf(label, sizeof(label), "sweep%d", k);
            if (parity_case(dec, ver, start + k * rows, rows, label) != 0) {
                bad_total = 1;
                printf("dspark-parity: first divergence in block starting at %d\n",
                       start + k * rows);
                break;
            }
        }
        printf("dspark-parity: %s (sweep of %d blocks from %d)\n",
               bad_total == 0 ? "PARITY OK" : "PARITY BROKEN", sweep, start);
        ds4_session_free(dec);
        ds4_session_free(ver);
        ds4_tokens_free(&prompt);
        ds4_engine_close(engine);
        return bad_total == 0 ? 0 : 1;
    }
    for (int ci = 0; ci < nlengths; ci++) {
        const int len = lengths[ci];
        char label[32];
        snprintf(label, sizeof(label), "case%d", ci);

        if (fresh) {
            ds4_session_free(dec); ds4_session_free(ver);
            dec = ver = NULL;
            if (ds4_session_create(&dec, engine, ctx) != 0 ||
                ds4_session_create(&ver, engine, ctx) != 0) {
                fprintf(stderr, "dspark-parity: session create failed\n");
                return 1;
            }
        }
        if (!parity_sync(dec, &prompt, len, "dec") || !parity_sync(ver, &prompt, len, "ver")) {
            bad_total = -1;
            break;
        }
        /* Both sessions must present the same frontier before anything else:
         * a differing prefill would make every downstream number moot. */
        float *pl = malloc((size_t)DS4_N_VOCAB * sizeof(float));
        float *pr = malloc((size_t)DS4_N_VOCAB * sizeof(float));
        if (!pl || !pr || ds4_session_copy_logits(dec, pl, (int)DS4_N_VOCAB) <= 0 ||
            ds4_session_copy_logits(ver, pr, (int)DS4_N_VOCAB) <= 0) {
            free(pl); free(pr);
            fprintf(stderr, "dspark-parity: frontier compare failed\n");
            bad_total = -1;
            break;
        }
        const parity_cmp fc = parity_compare(pl, pr, DS4_N_VOCAB);
        free(pl); free(pr);
        if (!fc.bitwise)
            printf("%s len=%d frontier DIFF dmax=%.6g argmax=%d/%d\n",
                   label, len, (double)fc.max_abs, fc.argmax_a, fc.argmax_b);

        const int bad = parity_case(dec, ver, len, rows, label);
        if (bad > 0) bad_total += bad;

        if (repeat) {
            ds4_session_rewind(dec, len);
            ds4_session_rewind(ver, len);
            if (!ds4_session_checkpoint_valid(dec) || !ds4_session_checkpoint_valid(ver)) {
                fprintf(stderr, "dspark-parity: repeat rewind invalidated a session\n");
                bad_total = -1; break;
            }
            printf("%s len=%d repeat:\n", label, len);
            const int bad2 = parity_case(dec, ver, len, rows, label);
            if (bad2 > 0) bad_total += bad2;
        }
        if (fresh) continue;
        ds4_session_rewind(dec, len);
        ds4_session_rewind(ver, len);
        if (!ds4_session_checkpoint_valid(dec) || !ds4_session_checkpoint_valid(ver)) {
            fprintf(stderr, "dspark-parity: rewind invalidated a session at len=%d\n", len);
            bad_total = -1;
            break;
        }
    }

    printf("dspark-parity: %s (rows not bit-identical: %d)\n",
           bad_total == 0 ? "PARITY OK" : (bad_total < 0 ? "ERROR" : "PARITY BROKEN"),
           bad_total < 0 ? -1 : bad_total);
    ds4_session_free(dec);
    ds4_session_free(ver);
    ds4_tokens_free(&prompt);
    ds4_engine_close(engine);
    return bad_total == 0 ? 0 : 1;
}
