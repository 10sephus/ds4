/* Native V4.1 DSpark graph oracle. No target experts are made resident. */
#include "../ds4.c"

#define REQUIRE(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); goto done; } } while (0)

static bool dump_tensor(const char *dir, const char *name, ds4_gpu_tensor *x, size_t floats) {
    char path[4096];
    if (snprintf(path, sizeof(path), "%s/%s.f32", dir, name) >= (int)sizeof(path)) return false;
    float *data = malloc(floats * sizeof(float));
    if (!data) return false;
    bool ok = ds4_gpu_tensor_read(x, 0, data, floats * sizeof(float));
    for (size_t i = 0; ok && i < floats; i++) ok = isfinite(data[i]);
    FILE *fp = ok ? fopen(path, "wb") : NULL;
    ok = fp && fwrite(data, sizeof(float), floats, fp) == floats;
    if (fp && fclose(fp)) ok = false;
    free(data);
    return ok;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s TARGET_GGUF DSPARK_GGUF OUTPUT_DIRECTORY\n", argv[0]);
        return 2;
    }
    int rc = 1;
    ds4_model target = {.fd = -1}, draft = {.fd = -1};
    ds4_weights tw = {0};
    ds4_dspark_weights dw = {0};
    ds41_dspark *d = NULL;
    ds4_gpu_tensor *hc = NULL;
    float *source = NULL;
    model_open(&target, argv[1], true, false);
    config_validate_model(&target);
    REQUIRE(DS4_MODEL_FAMILY == DS4_MODEL_FAMILY_DEEPSEEK41);
    weights_bind(&tw, &target, false, 0, UINT32_MAX, true, false);
    model_open(&draft, argv[2], true, false);
    REQUIRE(support_model_checkpoint_compatible(&draft));
    ds4_dspark_summary summary = model_dspark_summary(&draft);
    dspark_weights_bind_optional(&dw, &draft, &summary);
    REQUIRE(dw.v41 && !dw.invalid_tensors && !dw.missing_tensors && !dw.metadata_errors);
    REQUIRE(ds4_gpu_init());
    const uint64_t offsets[] = {tw.token_embd->abs_offset, tw.output->abs_offset};
    const uint64_t sizes[] = {tw.token_embd->bytes, tw.output->bytes};
    REQUIRE(ds4_gpu_set_model_map_spans(target.map, target.size, offsets, sizes, 2, tw.token_embd->bytes));
    REQUIRE(ds4_gpu_set_model_map_range(draft.map, draft.size, draft.tensor_data_pos,
        draft.size - draft.tensor_data_pos, draft.max_tensor_bytes));
    d = ds41_dspark_alloc(&dw);
    REQUIRE(d);
    enum { ROWS = 137 };
    const size_t elements = (size_t)ROWS * DS4_N_HC * DS4_N_EMBD;
    source = malloc(elements * sizeof(float));
    hc = ds4_gpu_tensor_alloc(elements * sizeof(float));
    REQUIRE(source && hc);
    for (unsigned stage = 0; stage < 3; stage++) {
        for (size_t i = 0; i < elements; i++)
            source[i] = ((int)((i * 17u + stage * 29u) % 257u) - 128) / 64.0f;
        REQUIRE(ds4_gpu_tensor_write(hc, 0, source, elements * sizeof(float)));
        REQUIRE(ds4_gpu_begin_commands());
        REQUIRE(ds41_dspark_capture(d, 37 + stage, 0, hc, ROWS));
        REQUIRE(ds4_gpu_end_commands());
    }
    REQUIRE(ds41_dspark_forward(d, &target, &tw, &draft, ROWS, 100));
    REQUIRE(d->kv_valid && d->kv_end == ROWS);
    REQUIRE(dump_tensor(argv[3], "main_x", d->main_x, DS41_DRAFT_WINDOW * DS4_N_EMBD));
    for (unsigned stage = 0; stage < 3; stage++) {
        char name[32];
        snprintf(name, sizeof(name), "window%u", stage);
        REQUIRE(dump_tensor(argv[3], name, d->window[stage], (DS41_DRAFT_WINDOW + DS41_DRAFT_ROWS) * DS4_N_HEAD_DIM));
    }
    REQUIRE(dump_tensor(argv[3], "hidden", d->batch.x, DS41_DRAFT_ROWS * DS4_N_EMBD));
    REQUIRE(dump_tensor(argv[3], "norm", d->batch.norm, DS41_DRAFT_ROWS * DS4_N_EMBD));
    REQUIRE(dump_tensor(argv[3], "logits", d->logits, DS41_DRAFT_ROWS * DS4_N_VOCAB));
    REQUIRE(ds4_gpu_tensor_read(d->logits, 0, d->host_logits, DS41_DRAFT_ROWS * DS4_N_VOCAB * sizeof(float)));
    REQUIRE(ds41_dspark_forward(d, &target, &tw, &draft, ROWS, 100));
    float *second = malloc(DS41_DRAFT_ROWS * DS4_N_VOCAB * sizeof(float));
    REQUIRE(second);
    bool repeated = ds4_gpu_tensor_read(d->logits, 0, second, DS41_DRAFT_ROWS * DS4_N_VOCAB * sizeof(float)) &&
        !memcmp(second, d->host_logits, DS41_DRAFT_ROWS * DS4_N_VOCAB * sizeof(float));
    free(second);
    REQUIRE(repeated);
    int previous = 100;
    for (unsigned row = 0; row < DS41_DRAFT_ROWS; row++) {
        float confidence;
        REQUIRE(ds41_dspark_markov(d, &draft, row, previous, &confidence));
        char name[32];
        snprintf(name, sizeof(name), "markov%u", row);
        REQUIRE(dump_tensor(argv[3], name, d->markov_logits, DS4_N_VOCAB));
        snprintf(name, sizeof(name), "confidence%u", row);
        REQUIRE(dump_tensor(argv[3], name, d->confidence, 1));
        previous = (int)dspark_argmax_f32(d->host_logits + (uint64_t)row * DS4_N_VOCAB, DS4_N_VOCAB);
        fprintf(stderr, "draft[%u]=%d confidence=%.7g\n", row, previous, confidence);
    }
    REQUIRE(dump_tensor(argv[3], "biased_logits", d->logits, DS41_DRAFT_ROWS * DS4_N_VOCAB));
    REQUIRE(ds4_gpu_begin_commands());
    const float pre[5][4] = {{1}, {1}, {1}, {1}, {1}};
    REQUIRE(ds4_gpu_tensor_write(d->batch.pre, 0, pre, sizeof(pre)));
    for (uint32_t row = 0; row < DS41_DRAFT_ROWS; row++) {
        const uint64_t bytes = (uint64_t)DS4_N_EMBD * DS4_N_HC * sizeof(float);
        ds4_gpu_tensor *out = ds4_gpu_tensor_view(d->batch.residual, row * bytes, bytes);
        bool ok = out && ds4_gpu_embed_token_hc_tensor(out, target.map, target.size,
            tw.token_embd->abs_offset, DS4_N_VOCAB, row ? dw.noise_token_id : 100,
            DS4_N_EMBD, DS4_N_HC);
        ds4_gpu_tensor_free(out);
        REQUIRE(ok);
    }
    REQUIRE(ds4_gpu_end_commands());
    REQUIRE(dump_tensor(argv[3], "embedded", d->batch.residual, 5u * 4u * DS4_N_EMBD));
    for (unsigned stage = 0; stage < 3; stage++) {
        REQUIRE(ds4_gpu_begin_commands());
        REQUIRE(ds41_dspark_stage(d, &draft, stage, ROWS));
        REQUIRE(ds4_gpu_end_commands());
#define STAGE_DUMP(field, count) do { \
    char name[64]; snprintf(name, sizeof(name), "stage%u_" #field, stage); \
    REQUIRE(dump_tensor(argv[3], name, d->batch.field, (count))); \
} while (0)
        STAGE_DUMP(q, 5u * DS4_N_HEAD * DS4_N_HEAD_DIM);
        STAGE_DUMP(kv, 5u * DS4_N_HEAD_DIM);
        STAGE_DUMP(heads, 5u * DS4_N_HEAD * DS4_N_HEAD_DIM);
        STAGE_DUMP(after_attn, 5u * DS4_N_HC * DS4_N_EMBD);
        STAGE_DUMP(norm, 5u * DS4_N_EMBD);
        STAGE_DUMP(route_logits, 5u * dw.n_experts);
        STAGE_DUMP(routed, 5u * DS4_N_EMBD);
        STAGE_DUMP(shared, 5u * DS4_N_EMBD);
        STAGE_DUMP(ffn_split, 5u * 24u);
        STAGE_DUMP(residual, 5u * DS4_N_HC * DS4_N_EMBD);
#undef STAGE_DUMP
    }
    puts("V4.1 draft capture, ring seed and forward: finite and repeatable");
    rc = 0;
done:
    if (ds4_gpu_commands_active()) (void)ds4_gpu_end_commands();
    ds4_gpu_tensor_free(hc);
    ds41_dspark_free(d);
    ds4_gpu_cleanup();
    model_close(&draft); model_close(&target);
    free(source);
    return rc;
}
