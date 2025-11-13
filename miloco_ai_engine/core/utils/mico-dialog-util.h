/**
 * Copyright (C) 2025 Xiaomi Corporation
 * This software may be used and distributed according to the terms of the Xiaomi Miloco License Agreement.
 */
#pragma once
#include "common/json-partial.h"
#include "utils/mico-common.h"

#define CHAT_CMP_ID_PREFIX "local-chatcmpl-"
#define PROMPT_PROPORTION_LIMIT 0.8

using json = nlohmann::ordered_json;

struct MicoRequest {
    int32_t id{0};
    int32_t priority{0};
    int32_t max_tokens{1024};
    json messages;
    json tools;
    std::vector<std::map<const unsigned char*, int32_t>> modal_prts;
    bool stop = false;
};

inline bool from_json_to_request(const json& j, MicoRequest& r) {
    std::string chat_cmpl_id = j.value("id", "local-chatcmpl-0");
    std::string prefix = CHAT_CMP_ID_PREFIX;
    if (chat_cmpl_id.substr(0, prefix.size()) == prefix) {
        r.id = std::stoi(chat_cmpl_id.substr(prefix.size()));
    }

    r.max_tokens = j.value("max_tokens", r.max_tokens);
    r.priority = j.value("priority", r.priority);
    if (j.contains("messages")) r.messages = j.at("messages");
    if (j.contains("tools")) r.tools = j.at("tools");
    if (j.contains("modal_prts")) {
        for (const auto& modal : j.at("modal_prts")) {
            std::map<const unsigned char*, int32_t> modal_map;
            for (const auto& [key, value] : modal.items()) {
                std::uintptr_t addr_value = 0;
                try {
                    addr_value = static_cast<std::uintptr_t>(std::stoull(key, nullptr, 10));
                } catch (const std::exception&) {
                    LOG_ERR("ERR: invalid address in modal_prts: %s\n", key.c_str());
                    return false;
                }
                modal_map[reinterpret_cast<const unsigned char*>(addr_value)] = value;
            }
            r.modal_prts.push_back(modal_map);
        }
    }
    r.stop = j.value("stop", false);
    return true;
}

inline int32_t stop_process(bool sucess, std::string& respone, const char** content, int32_t& is_finished,
                            LlamaSeqState& state, LlamaMicoContext* context, int32_t seq_id,
                            bool stop_infer = true) {  // End seq_id
    state.respone = respone;
    *content = state.respone.c_str();

    if (stop_infer) {
        is_finished = 1;
        state.is_infering.store(false);
        state.n_past.store(0);

        LlamaMemoryScheduler* ms = static_cast<LlamaMemoryScheduler*>(context->memory_scheduler);
        ms->submit_clear_mem(seq_id, -1, -1);

        context->erase_seq(seq_id);
    } else {
        is_finished = 0;
    }

    if (!sucess) {
        LOG_ERR("ERR: %s", respone.c_str());
        return -1;
    }
    return 0;
}

inline void apply_chat_templates(common_chat_params& formatted_chat, common_chat_templates_inputs& tmpl_inputs,
                                 LlamaMicoContext* context, json messages, json tools) {
    tmpl_inputs.messages = common_chat_msgs_parse_oaicompat(messages);
    if (!tools.is_null() && !tools.empty()) {
        tmpl_inputs.tools = common_chat_tools_parse_oaicompat(tools);
    }
    tmpl_inputs.add_generation_prompt = true;
    tmpl_inputs.use_jinja = true;  // jinja not support yet
    tmpl_inputs.enable_thinking = false;
    formatted_chat = common_chat_templates_apply(context->tmpls.get(), tmpl_inputs);
}

inline bool ready_modal_bitmaps(std::vector<std::map<const unsigned char*, int32_t>>& modal_prts,
                                common_chat_templates_inputs& tmpl_inputs, LlamaMicoContext* context,
                                LlamaSeqState& state) {
    if (!modal_prts.empty()) {
        for (const auto& modal : modal_prts) {
            for (const auto& [p, len] : modal) {
                auto bitmap_ptr = mtmd_helper_bitmap_init_from_buf(context->ctx_vision.get(), p, len, 0, 0);
                if (!bitmap_ptr) {
                    return false;
                }
                state.bitmaps.entries.emplace_back(bitmap_ptr);
            }
        }
    } else {
        // Images converted from base64
        for (const auto& m : tmpl_inputs.messages) {
            for (const auto& p : m.content_parts) {
                for (const auto& img : p.images) {
                    auto bitmap_ptr = mtmd_helper_bitmap_init_from_buf(
                        context->ctx_vision.get(), reinterpret_cast<const unsigned char*>(img.c_str()), img.size(), 0,
                        0);
                    if (!bitmap_ptr) {
                        return false;
                    }
                    state.bitmaps.entries.emplace_back(bitmap_ptr);
                }
            }
        }
    }
    return true;
}

inline bool from_input_to_token_chunks(common_chat_params& formatted_chat, std::shared_ptr<mtmd::input_chunks> chunks,
                                       LlamaMicoContext* context, LlamaSeqState& state) {
    mtmd_input_text text;
    text.text = formatted_chat.prompt.c_str();
    text.add_special = true;
    text.parse_special = true;
    auto bitmaps_c_ptr = state.bitmaps.c_ptr();
    int32_t ret =
        mtmd_tokenize(context->ctx_vision.get(), chunks->ptr.get(), &text, bitmaps_c_ptr.data(), bitmaps_c_ptr.size());
    state.bitmaps.entries.clear();
    return ret == 0;
}

inline void limit_prompt_tokens(std::shared_ptr<mtmd::input_chunks> chunks, int32_t n_usage_context,
                                LlamaSeqState& state) {
    float prompt_proportion = PROMPT_PROPORTION_LIMIT;
    int32_t prompt_limit = n_usage_context * prompt_proportion;
    int32_t current_tokens = 0;
    int32_t chunk_size = mtmd_input_chunks_size(chunks->ptr.get());
    for (int i = 0; i < chunk_size; ++i) {
        auto chunk = mtmd_input_chunks_get(chunks->ptr.get(), i);
        current_tokens += mtmd_input_chunk_get_n_tokens(chunk);
    }

    if (current_tokens > prompt_limit) {
        LOG_INF("prompt_tokens %d > usage_context %d * %f, need to crop\n", current_tokens, n_usage_context,
                prompt_proportion);

        mtmd_input_chunks* new_chunks = mtmd_input_chunks_init();
        int32_t remaining_tokens = prompt_limit;

        for (int i = chunk_size - 1; i >= 0 && remaining_tokens > 0; --i) {
            auto chunk = mtmd_input_chunks_get(chunks->ptr.get(), i);
            size_t n_tokens_chunk = mtmd_input_chunk_get_n_tokens(chunk);
            if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                const llama_token* tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_tokens_chunk);
                int32_t tokens_to_keep = std::min((int32_t)n_tokens_chunk, remaining_tokens);

                if (tokens_to_keep > 0) {
                    std::vector<llama_token> new_tokens(tokens + n_tokens_chunk - tokens_to_keep,
                                                        tokens + n_tokens_chunk);
                    mtmd_input_chunk* text_chunk = mtmd_create_text_chunk(std::move(new_tokens));
                    mtmd_input_chunks_insert_chunk_front(new_chunks, text_chunk);
                    mtmd_input_chunk_free(text_chunk);
                    remaining_tokens -= tokens_to_keep;
                }
            } else {
                if (n_tokens_chunk <= remaining_tokens) {
                    mtmd_input_chunk* copied_chunk = mtmd_input_chunk_copy(chunk);
                    mtmd_input_chunks_insert_chunk_front(new_chunks, copied_chunk);
                    mtmd_input_chunk_free(copied_chunk);
                    remaining_tokens -= n_tokens_chunk;
                } else {
                    // Discard modal tokens
                    break;
                }
            }
        }
        // Replace original chunks
        chunks->ptr.reset(new_chunks);
    }
}