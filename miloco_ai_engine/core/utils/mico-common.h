/**
 * Copyright (C) 2025 Xiaomi Corporation
 * This software may be used and distributed according to the terms of the Xiaomi Miloco License Agreement.
 */
#ifndef MICO_COMMON_H
#define MICO_COMMON_H
#include <limits.h>

#include <map>
#include <mutex>

#include "common/sampling.h"
#include "mutil-modal/mtmd-helper.h"
#include "mutil-modal/mtmd.h"
#include "utils/llama-memory-scheduling.h"

struct LlamaSeqState {
    std::atomic<llama_token> last_token{-1};
    std::atomic<size_t> n_past{0};
    // int n_max_genarate{INT_MAX};
    std::atomic<bool> is_infering{false};  // true if this sequence is already inferred
    std::string respone{""};               // last text generated for this sequence
    mtmd::bitmaps bitmaps;
};

struct LlamaMicoContext {
    mtmd::context_ptr ctx_vision;   // for modal
    common_init_result llama_init;  // initialize/release llama_context manually

    llama_model* model;
    llama_context* lctx;
    const llama_vocab* vocab;
    common_sampler* smpl;
    int32_t n_batch;
    int32_t n_seq_max;
    int32_t n_usage_context;

    // cache
    int32_t kv_cache_seq;

    void* batch_scheduler{nullptr};   // batch scheduler
    void* memory_scheduler{nullptr};  // batch scheduler

    // state for sequences
    std::map<size_t, LlamaSeqState> process_seqs;
    mutable std::mutex process_seqs_mutex;
    LlamaSeqState& get_seq_state(size_t id) {
        std::lock_guard<std::mutex> lock(process_seqs_mutex);
        return process_seqs[id];
    }

    std::map<size_t, int32_t> cmpl_to_seq;
    mutable std::mutex cmpl_to_seq_mutex;
    int32_t set_seq_id(size_t cmpl_id) {
        std::lock_guard<std::mutex> lock(cmpl_to_seq_mutex);
        int32_t seq_id = -1;
        for (int i = 0; i < n_seq_max; i++)
            if (!get_seq_state(i).is_infering.load()) {
                seq_id = i;
                break;
            }
        if (seq_id != -1) cmpl_to_seq[cmpl_id] = seq_id;
        return seq_id;
    }
    int32_t get_seq_id(size_t cmpl_id) {
        std::lock_guard<std::mutex> lock(cmpl_to_seq_mutex);
        int32_t has = (cmpl_to_seq.count(cmpl_id) > 0 ? cmpl_to_seq[cmpl_id] : -1);
        return has;
    }
    bool erase_seq(int32_t seq_id) {
        std::lock_guard<std::mutex> lock(cmpl_to_seq_mutex);
        size_t to_erase = -1;
        for (auto& it : cmpl_to_seq) {
            if (it.second == seq_id) {
                to_erase = it.first;
                break;
            }
        }
        if (to_erase >= 0) cmpl_to_seq.erase(to_erase);
        return to_erase >= 0;
    }

    std::string media_marker = MICO_DEFAULT_IMAGE_MARKER;
    common_chat_templates_ptr tmpls;
    llama_tokens antiprompt_tokens;
    int n_threads = 1;

    LlamaMicoContext(common_params& params) : llama_init(common_init_from_params(params)) {
        model = llama_init.model.get();
        lctx = llama_init.context.get();
        vocab = llama_model_get_vocab(model);
        smpl = common_sampler_init(model, params.sampling);
        n_threads = params.cpuparams.n_threads;
        n_batch = params.n_batch;
        n_usage_context = params.n_usage_context;

        n_seq_max = params.n_seq_max;
        n_seq_max -= params.cache_seq;  // reserved space for cache

        kv_cache_seq = params.cache_seq;

        // memory_scheduler
        memory_scheduler = new LlamaMemoryScheduler(lctx);

        if (!model || !lctx) {
            exit(1);
        }

        if (!llama_model_chat_template(model, nullptr) && params.chat_template.empty()) {
            LOG_ERR("Model does not have chat template.\n");
            LOG_ERR("  For old llava models, you may need to use '--chat-template vicuna'\n");
            LOG_ERR("  For MobileVLM models, use '--chat-template deepseek'\n");
            LOG_ERR("  For Mistral Small 3.1, use '--chat-template mistral-v7'\n");
            exit(1);
        }

        tmpls = common_chat_templates_init(model, params.chat_template);
        LOG_INF("%s: chat template example:\n%s\n", __func__,
                common_chat_format_example(tmpls.get(), params.use_jinja).c_str());

        init_vision_context(params);

        // load antiprompt tokens for legacy templates
        if (params.chat_template == "vicuna") {
            antiprompt_tokens = common_tokenize(lctx, "ASSISTANT:", false, true);
        } else if (params.chat_template == "deepseek") {
            antiprompt_tokens = common_tokenize(lctx, "###", false, true);
        }
    }

    ~LlamaMicoContext() { common_sampler_free(smpl); }

    void init_vision_context(common_params& params) {
        const char* clip_path = params.mmproj.path.c_str();
        mtmd_context_params mparams = mtmd_context_params_default();
        mparams.use_gpu = params.mmproj_use_gpu;
        mparams.print_timings = true;
        mparams.n_threads = params.cpuparams.n_threads;
        mparams.verbosity = params.verbosity > 0 ? GGML_LOG_LEVEL_DEBUG : GGML_LOG_LEVEL_INFO;
        ctx_vision.reset(mtmd_init_from_file(clip_path, model, mparams));
        if (!ctx_vision.get()) {
            LOG_ERR("Failed to load vision model from %s\n", clip_path);
            exit(1);
        }
    }

    bool check_antiprompt(const llama_tokens& generated_tokens) {
        if (antiprompt_tokens.empty() || generated_tokens.size() < antiprompt_tokens.size()) {
            return false;
        }
        return std::equal(generated_tokens.end() - antiprompt_tokens.size(), generated_tokens.end(),
                          antiprompt_tokens.begin());
    }
};

#endif  // MICO_COMMON_H