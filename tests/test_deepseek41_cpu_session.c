/* Exercise real session admission and state transitions without model weights.
 * Each request either reuses an existing frontier or stops before forward. */
#include "../ds4.c"
#include <assert.h>

typedef struct {
    unsigned calls, stop_at, progress;
} session_control;

static bool cancel_at_poll(void *ud) {
    session_control *control = ud;
    return ++control->calls >= control->stop_at;
}

static void count_progress(void *ud, const char *event, int current, int total) {
    session_control *control = ud;
    (void)event; (void)current; (void)total;
    control->progress++;
}

static ds4_session *session_fixture(ds4_engine *engine) {
    ds4_session *s = calloc(1, sizeof(*s));
    assert(s);
    s->engine = engine;
    s->ctx_size = 8;
    s->prefill_cap = 1;
    s->ds41_cpu = calloc(1, sizeof(*s->ds41_cpu));
    assert(s->ds41_cpu);
    ds41_cpu_state *g = s->ds41_cpu;
    g->table[0].fd = g->table[1].fd = -1;
    g->ctx = 8;
    g->pos = 2;
    g->valid = true;
    g->attention = ds41_cpu_attention_create(g->ctx);
    assert(g->attention);
    g->history.tail[0] = 2;
    g->history.tail[1] = 1;
    g->history.tail[2] = DS4_ENGRAM_DEAD;
    g->attention->previous_kv[0][0] = 19.0f;
    g->attention->previous_score[1][0] = 23.0f;
    g->residual[0] = 17.0f;
    g->pre[0] = 1.0f;
    s->logits = calloc(DS4_N_VOCAB, sizeof(*s->logits));
    assert(s->logits);
    for (uint32_t i = 0; i < DS4_N_VOCAB; i++) s->logits[i] = (float)i;
    ds4_tokens_push(&s->checkpoint, 1);
    ds4_tokens_push(&s->checkpoint, 2);
    s->checkpoint_valid = true;
    return s;
}

static void check_frontier(const ds4_session *s) {
    assert(s->checkpoint_valid && s->checkpoint.len == 2);
    assert(s->checkpoint.v[0] == 1 && s->checkpoint.v[1] == 2);
    assert(s->ds41_cpu->valid && s->ds41_cpu->pos == 2);
    assert(s->ds41_cpu->history.tail[0] == 2);
    assert(s->ds41_cpu->attention->previous_kv[0][0] == 19.0f);
    assert(s->logits[7] == 7.0f);
}

static void check_reset(const ds4_session *s) {
    assert(!s->checkpoint_valid && s->checkpoint.len == 0);
    assert(s->ds41_cpu->valid && s->ds41_cpu->pos == 0);
    for (unsigned i = 0; i < DS4_ENGRAM_NGRAM - 1; i++)
        assert(s->ds41_cpu->history.tail[i] == DS4_ENGRAM_DEAD);
    assert(s->ds41_cpu->attention->previous_kv[0][0] == 0.0f);
    assert(s->ds41_cpu->attention->previous_score[1][0] == 0.0f);
    assert(s->ds41_cpu->residual[0] == 0.0f && s->ds41_cpu->pre[0] == 0.0f);
}

static void test_reuse_and_invalid_input(ds4_engine *engine) {
    ds4_session *s = session_fixture(engine);
    session_control control = {0};
    ds4_session_set_progress(s, count_progress, &control);
    char err[256] = {0};
    int ids[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    ds4_tokens prompt = {.v = ids, .len = 2, .cap = 8};
    assert(ds4_session_sync(s, &prompt, err, sizeof(err)) == 0);
    assert(control.progress == 0);
    check_frontier(s);
    float logits[16];
    assert(ds4_session_copy_logits(s, logits, 16) == 16 && logits[7] == 7.0f);
    assert(ds4_session_argmax(s) == 15);

    prompt.len = 0;
    assert(ds4_session_sync(s, &prompt, err, sizeof(err)) != 0);
    prompt.len = 8;
    assert(ds4_session_sync(s, &prompt, err, sizeof(err)) != 0);
    prompt.len = 3;
    const int invalid[] = {-1, 16};
    for (unsigned i = 0; i < 2; i++) {
        ids[2] = invalid[i];
        assert(ds4_session_sync(s, &prompt, err, sizeof(err)) != 0);
        assert(strstr(err, "vocabulary"));
        check_frontier(s);
        assert(ds4_session_eval(s, invalid[i], err, sizeof(err)) != 0);
        check_frontier(s);
    }
    assert(control.progress == 0);
    ds4_session_free(s);
}

static void test_cancel_and_divergence(ds4_engine *engine) {
    ds4_session *s = session_fixture(engine);
    int ids[] = {1, 2, 3};
    ds4_tokens prompt = {.v = ids, .len = 3, .cap = 3};
    char err[256] = {0};
    session_control control = {.stop_at = 1};
    ds4_session_set_cancel(s, cancel_at_poll, &control);
    assert(ds4_session_sync(s, &prompt, err, sizeof(err)) == DS4_SESSION_SYNC_INTERRUPTED);
    check_frontier(s);

    /* Stop an append at its first token boundary, retaining the live prefix. */
    control = (session_control){.stop_at = 2};
    assert(ds4_session_sync(s, &prompt, err, sizeof(err)) == DS4_SESSION_SYNC_INTERRUPTED);
    check_frontier(s);

    /* A divergent transcript must clear history and pending compressor pairs
     * before attempting its first token, even if that token is cancelled. */
    ids[0] = 4;
    control = (session_control){.stop_at = 2};
    assert(ds4_session_sync(s, &prompt, err, sizeof(err)) == DS4_SESSION_SYNC_INTERRUPTED);
    check_reset(s);
    assert(ds4_session_argmax(s) == -1);
    ds4_session_free(s);
}

static void test_rewind_and_invalidate(ds4_engine *engine) {
    ds4_session *s = session_fixture(engine);
    char err[256] = {0};
    ds4_session_rewind(s, 2);
    check_frontier(s);
    ds4_session_rewind(s, 1);
    assert(s->checkpoint.len == 1 && !s->checkpoint_valid);
    assert(ds4_session_argmax(s) == -1);
    assert(ds4_session_eval(s, 3, err, sizeof(err)) != 0);
    assert(strstr(err, "synchronized checkpoint"));
    ds4_session_invalidate(s);
    check_reset(s);
    ds4_session_free(s);
}

static void test_decode_limits_and_dispatch(ds4_engine *engine) {
    ds4_session *s = session_fixture(engine);
    char err[256] = {0};
    ds4_decode_item repeated[] = {{s, 3}, {s, 4}};
    assert(ds4_sessions_eval_batch(repeated, 2, err, sizeof(err)) != 0);
    assert(strstr(err, "repeats session"));
    check_frontier(s);

    int accepted[2] = {-9, -9};
    uint64_t rng = 1;
    err[0] = '\0';
    assert(ds4_session_eval_speculative_argmax(s, -1, 2, -1, accepted, 2,
                                              err, sizeof(err)) == -1);
    assert(err[0] && accepted[0] == -9);
    err[0] = '\0';
    assert(ds4_session_eval_speculative(s, -1, 2, -1, 1.0f, 0, 1.0f, 0.0f,
                                      &rng, accepted, 2, err, sizeof(err)) == -1);
    assert(err[0] && accepted[0] == -9);
    check_frontier(s);

    for (int i = 2; i < s->ctx_size; i++) ds4_tokens_push(&s->checkpoint, i);
    s->ds41_cpu->pos = (uint32_t)s->checkpoint.len;
    assert(ds4_session_eval(s, 3, err, sizeof(err)) != 0);
    assert(s->checkpoint.len == 8 && s->ds41_cpu->pos == 8);
    s->checkpoint.len = 2;
    s->ds41_cpu->pos = 1;
    assert(ds4_session_eval(s, 3, err, sizeof(err)) != 0);
    assert(!s->checkpoint_valid && s->checkpoint.len == 2);
    ds4_session_free(s);
}

int main(void) {
    const ds4_shape saved = g_ds4_shape;
    g_ds4_shape = DS4_SHAPE_FLASH41;
    g_ds4_shape.n_vocab = 16;
    ds4_engine engine = {.backend = DS4_BACKEND_CPU};
    test_reuse_and_invalid_input(&engine);
    test_cancel_and_divergence(&engine);
    test_rewind_and_invalidate(&engine);
    test_decode_limits_and_dispatch(&engine);
    g_ds4_shape = saved;
    puts("V4.1 CPU session reuse, cancellation, divergence, rewind and bounds: PASS");
    return 0;
}
