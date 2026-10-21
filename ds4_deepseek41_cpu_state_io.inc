/* V4.1 CPU snapshots use the GPU payload layout. Raw windows retain physical
 * ring order; compressed caches contain only live rows. The three unfinished
 * ratio-two pairs are durable only at odd token frontiers. */
typedef struct {
    float *data;
    uint64_t bytes;
} ds41_cpu_payload_span;

static uint32_t ds41_cpu_payload_spans(ds41_cpu_state *g, uint32_t pos,
                                      ds41_cpu_payload_span spans[54]) {
    uint32_t n = 0;
    const uint32_t raw = pos < 128u ? pos : 128u;
    for (uint32_t il = 0; il < 40; il++)
        spans[n++] = (ds41_cpu_payload_span){g->attention->window[il],
                                             (uint64_t)raw * 512u * sizeof(float)};
    for (uint32_t i = 0; i < 4; i++) {
        const uint32_t live = pos / (i < 3 ? 2u : 1u);
        spans[n++] = (ds41_cpu_payload_span){g->attention->compressed[i],
                                             (uint64_t)live * 512u * sizeof(float)};
        spans[n++] = (ds41_cpu_payload_span){g->attention->index_cache[i],
                                             (uint64_t)live * 128u * sizeof(float)};
        if (i < 3 && (pos & 1u)) {
            spans[n++] = (ds41_cpu_payload_span){g->attention->previous_kv[i],
                                                 512u * sizeof(float)};
            spans[n++] = (ds41_cpu_payload_span){g->attention->previous_score[i],
                                                 512u * sizeof(float)};
        }
    }
    return n;
}

static uint64_t ds41_cpu_payload_body_bytes(ds41_cpu_state *g, uint32_t pos) {
    if (!g || !g->attention || pos > g->ctx) return 0;
    ds41_cpu_payload_span spans[54];
    const uint32_t n = ds41_cpu_payload_spans(g, pos, spans);
    uint64_t bytes = (uint64_t)pos * sizeof(uint32_t) +
                     (uint64_t)DS4_N_VOCAB * sizeof(float);
    for (uint32_t i = 0; i < n; i++) bytes += spans[i].bytes;
    return bytes;
}

static bool ds41_cpu_payload_storage_valid(ds41_cpu_state *g, uint32_t pos) {
    if (!g || !g->attention || !g->ctx || g->ctx > 1048576u ||
        pos >= g->ctx || g->attention->ctx != g->ctx) return false;
    for (uint32_t i = 0; i < 4; i++)
        if (pos / (i < 3 ? 2u : 1u) > g->attention->comp_cap[i]) return false;
    ds41_cpu_payload_span spans[54];
    const uint32_t n = ds41_cpu_payload_spans(g, pos, spans);
    for (uint32_t i = 0; i < n; i++)
        if (spans[i].bytes && !spans[i].data) return false;
    return true;
}

static int ds41_cpu_save_payload(ds4_session *s, FILE *fp, char *err, size_t errlen) {
    ds41_cpu_state *g = s ? s->ds41_cpu : NULL;
    if (!s || !fp || !s->logits || !s->checkpoint_valid || !g || !g->valid ||
        !g->pos || g->pos != (uint32_t)s->checkpoint.len ||
        !ds41_cpu_payload_storage_valid(g, g->pos)) {
        payload_set_err(err, errlen, "V4.1 CPU session has no complete frontier to save");
        return 1;
    }
    const uint32_t h[DS4_SESSION_PAYLOAD_U32_FIELDS] = {
        DS4_SESSION_PAYLOAD_MAGIC, DS4_SESSION_PAYLOAD_VERSION,
        g->ctx, 1, 128, 128, g->ctx + 1u, g->pos, 40, 512, 128, DS4_N_VOCAB, 0x413431u
    };
    for (uint32_t i = 0; i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++)
        if (payload_write_u32(fp, h[i], err, errlen)) return 1;
    for (int i = 0; i < s->checkpoint.len; i++)
        if (payload_write_u32(fp, (uint32_t)s->checkpoint.v[i], err, errlen)) return 1;
    if (payload_write_bytes(fp, s->logits,
                            (uint64_t)DS4_N_VOCAB * sizeof(float), err, errlen)) return 1;
    ds41_cpu_payload_span spans[54];
    const uint32_t n = ds41_cpu_payload_spans(g, g->pos, spans);
    for (uint32_t i = 0; i < n; i++)
        if (payload_write_bytes(fp, spans[i].data, spans[i].bytes, err, errlen)) return 1;
    return 0;
}

static int ds41_cpu_load_payload(ds4_session *s, FILE *fp, const uint32_t *h,
                                 uint64_t remaining, char *err, size_t errlen) {
    ds41_cpu_state *g = s ? s->ds41_cpu : NULL;
    const uint32_t pos = h ? h[7] : 0;
    if (!s || !fp || !h || !s->logits || !g || !g->token_map || !pos ||
        h[0] != DS4_SESSION_PAYLOAD_MAGIC || h[1] != DS4_SESSION_PAYLOAD_VERSION ||
        pos >= h[2] || h[2] > 1048576u || h[3] != 1 || h[4] != 128 || h[5] != 128 ||
        h[6] != h[2] + 1u || h[8] != 40 || h[9] != 512 || h[10] != 128 ||
        h[11] != DS4_N_VOCAB || h[12] != 0x413431u ||
        !ds41_cpu_payload_storage_valid(g, pos) ||
        remaining != ds41_cpu_payload_body_bytes(g, pos)) {
        payload_set_err(err, errlen, "invalid V4.1 CPU snapshot dimensions or size");
        return 1;
    }

    ds4_tokens tokens = {0};
    int rc = 0;
    for (uint32_t i = 0; i < pos; i++) {
        uint32_t token;
        if (payload_read_u32(fp, &token, &remaining, err, errlen) || token >= DS4_N_VOCAB) {
            payload_set_err(err, errlen, "invalid V4.1 CPU snapshot token");
            rc = 1;
            break;
        }
        token_vec_push(&tokens, (int)token);
    }
    if (!rc) {
        /* Once the first live buffer changes, failure must never leave the
         * former checkpoint available for decode or another snapshot. */
        s->checkpoint_valid = false;
        s->mtp_draft_valid = false;
        g->valid = false;
        rc = payload_read_bytes(fp, s->logits,
                                (uint64_t)DS4_N_VOCAB * sizeof(float),
                                &remaining, err, errlen);
        ds41_cpu_payload_span spans[54];
        const uint32_t n = ds41_cpu_payload_spans(g, pos, spans);
        for (uint32_t i = 0; i < n && !rc; i++)
            rc = payload_read_bytes(fp, spans[i].data, spans[i].bytes,
                                    &remaining, err, errlen);
        if (!rc) {
            /* Reset only hash history: resetting the attention state here
             * could discard the just-restored rings and unfinished pairs. */
            ds4_engram_history_reset(&g->history);
            for (uint32_t i = 0; i < DS4_ENGRAM_NGRAM - 1u && i < pos; i++)
                g->history.tail[i] = (int32_t)g->token_map[tokens.v[pos - 1u - i]];
            g->pos = pos;
            g->valid = true;
            ds4_tokens_copy(&s->checkpoint, &tokens);
            s->checkpoint_valid = true;
        }
    }
    ds4_tokens_free(&tokens);
    return rc;
}
