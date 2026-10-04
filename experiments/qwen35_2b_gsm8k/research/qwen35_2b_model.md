# Qwen/Qwen3.5-2B — RL Training Compatibility (READ-ONLY investigation)

_Date: 2026-09-23. Metadata only; no safetensors downloaded._

## Verdict (TL;DR)

| Question | Answer |
|---|---|
| (a) Model exists & accessible | ✅ **YES** — public, ungated, non-private. 4.78M downloads. |
| (b) Architecture class | ⚠️ **`Qwen3_5ForConditionalGeneration`** — VL-style, carries a `vision_config` (vision tower) even for text-only use. Hybrid **Gated DeltaNet (linear attention)** + full attention, 3:1 ratio. |
| (c) Chat-template hash match w/ gateway | ✅ **MATCH** — `273d8e0e…` = **`qwen3_5` (nothink)** schema. Byte-exact. |
| (d) Engine support / risks | ✅ vLLM natively supports GDN (`qwen_gdn_linear_attn`, ≥0.23); verl has FSDP GRPO support (PR #5682, merged) + Megatron examples. **Risks:** needs `flash-linear-attention` (fla) + `causal-conv1d` CUDA kernels; VL wrapper needs `--language-model-only` for text; MTP layer present. |
| (e) Alternatives if 2B problematic | **Qwen3-4B-Instruct-2507** (repo already validated, plain `qwen3` schema, no GDN/vision), or Qwen3.5-4B. |

## 1. Repo existence & access

`Qwen/Qwen3.5-2B` — `gated=False`, `private=False`, `library_name=transformers`,
`pipeline_tag=image-text-to-text`, tags include `qwen3_5`, `conversational`,
`base_model:Qwen/Qwen3.5-2B-Base` (so this is the Instruct/post-trained variant).

Siblings (`?blobs=true`): single-shard `model.safetensors-00001-of-00001.safetensors`
= **4,548,221,488 bytes (~4.55 GB)** → ~2.27B params in bf16 (matches "2B"). Also
present: `chat_template.jinja` (7755 B), `config.json`, `tokenizer.json`,
`merges.txt`/`vocab.json` (Qwen2Tokenizer/BPE), **`preprocessor_config.json` +
`video_preprocessor_config.json`** (vision/video processors — confirms VL). No
`generation_config.json` (HTTP 404).

## 2. Metadata files persisted

Downloaded to `…/experiments/qwen35_2b_gsm8k/research/qwen35_2b_meta/`:
`config.json`, `tokenizer_config.json`, `chat_template.jinja`,
`preprocessor_config.json`, `video_preprocessor_config.json`.
`generation_config.json` **does not exist** on the repo (404) — no vendor-recommended
sampling params shipped; use Qwen defaults (see §6).

## 3. Architecture (config.json)

- `architectures`: **`Qwen3_5ForConditionalGeneration`**; `model_type`: `qwen3_5`.
- **`vision_config` present** (depth 24, hidden 1024, patch 16, `Qwen2VLImageProcessorFast`)
  + `text_config` (`model_type: qwen3_5_text`) → early-fusion VLM. `image_token_id`,
  `video_token_id`, `vision_start/end_token_id` all defined.
- Text tower: `hidden_size` **2048**, `num_hidden_layers` **24**,
  `num_attention_heads` **8**, `num_key_value_heads` **2** (GQA 4:1),
  `head_dim` 256, `intermediate_size` 6144, `vocab_size` **248320**,
  `max_position_embeddings` **262144** (256K), `tie_word_embeddings` **true**,
  `torch_dtype`/`dtype` **bfloat16**, `hidden_act` silu (SwiGLU), `rms_norm_eps` 1e-6.
- **Hybrid attention** — `layer_types` = 3× `linear_attention` then 1× `full_attention`,
  repeated 6× over 24 layers (`full_attention_interval: 4`). Linear layers are
  **Gated DeltaNet**: `linear_conv_kernel_dim` 4, `linear_key/value_head_dim` 128,
  `linear_num_key/value_heads` 16, `mamba_ssm_dtype` float32, `attn_output_gate` true.
- **MTP**: `mtp_num_hidden_layers: 1` (multi-token-prediction head → optional vLLM
  speculative decoding).
- **RoPE**: `rope_type` default, `rope_theta` 1e7, `partial_rotary_factor` 0.25,
  **mRoPE** (`mrope_interleaved: true`, `mrope_section: [11,11,10]`) — multimodal
  positional encoding.

**Param / memory estimate:** ~2.27B params (from 4.55 GB bf16 shard incl. vision
tower). bf16 **weights ≈ 4.55 GB**. RL training footprint is much larger: Adam
optimizer (fp32 m+v) + grads roughly ×(1 + 2×4/2 + 2) ≈ **~27–35 GB** per replica
before activations/KV — fits a single 40–80 GB GPU for the actor; the Mamba/GDN
state and 256K context KV add on top.

**Implications:**
- YES, it is a **VL-style `…ForConditionalGeneration`** with a vision tower even for
  text-only GSM8K. For training/serving text-only, use `--language-model-only`
  (vLLM) / mm wrappers must be handled or the vision encoder wastes memory. The repo
  already notes this class carries a `vision_config` (see `backends/verl/README.md`
  re: `context_parallel_size > 1`).
- YES, it uses **hybrid Gated DeltaNet linear attention** → the fast path needs
  `flash-linear-attention` (fla) + `causal-conv1d` CUDA kernels at import time
  (silent slow fallback otherwise; no MPS/CPU fast path).

## 4. Chat-template hash vs. gateway

**How the gateway hashes** (`rollout_gateway/response_schemas.py::resolve_schema_name`):
```python
digest = hashlib.sha256(template.encode("utf-8")).hexdigest()
```
The **raw** template string (tokenizer's `chat_template`), UTF-8, **no strip / no
normalization**. Looked up in `_TEMPLATE_HASHES`.

**Computed** on the downloaded `chat_template.jinja` (7755 B, raw and stripped both):
`273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80`

→ **MATCH** = `_TEMPLATE_HASHES["273d8e0e…"] = "qwen3_5"`, the **nothink** variant.
(The other expected qwen3_5 hash `a4aee8af…` is the "think" variant — not this repo.)

**Tool-call format:** XML-style, confirmed in the template:
`<tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n</function>\n</tool_call>`
— exactly what `QWEN3_5_SCHEMA` parses (`x-regex` on `<function=…>` /
`<parameter=…>`). So `tokenizer.parse_response(…, schema=qwen3_5)` in
`HfTemplateRenderer._parse_with_schema` will correctly extract reasoning/text/tool
calls. **No gateway change required** — this model drops straight into the existing
`qwen3_5` schema, same as Qwen3.5-0.8B/27B/Qwen3-Coder already in the live test.

## 5. Engine support (verl 0.9.0 / vLLM 0.24.0 / transformers 5.12.1)

_(.venv probe was not run; findings from official docs/PRs.)_
- **transformers**: `qwen3_5` is a first-class model (`docs/model_doc/qwen3_5.md`).
  Config here declares `transformers_version 4.57.0.dev0`; a 5.12.1 floor is well
  above that, so the class is present. Fast GDN path checks for `causal_conv1d` +
  `flash-linear-attention` at import.
- **vLLM**: native GDN support — `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn`
  exists across **v0.23–v0.27** (`QwenGatedDeltaNetAttention`, `chunk_gated_delta_rule`).
  v0.24 should support it. Official recipe flags:
  - `--reasoning-parser qwen3`
  - disable thinking via CLI: `--default-chat-template-kwargs '{"enable_thinking": false}'`
  - tool calling: `--enable-auto-tool-choice --tool-call-parser qwen3_coder`
  - text-only: `--language-model-only` (skips vision encoder)
  - optional MTP speculative decoding: `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'`
  - `--enable-prefix-caching` (Mamba cache "align" mode experimental)
- **verl**: **Qwen3.5 FSDP GRPO support merged** (verl PR #5682) + Megatron GRPO
  example `run_qwen3_5-35b-megatron.sh`. So RL (GRPO) on Qwen3.5 is supported;
  validate that verl 0.9.0 actually contains #5682 (check its changelog/commit).

**Risks / requirements checklist:**
1. Install `flash-linear-attention` + `causal-conv1d` (CUDA) or the GDN path is slow.
2. VL wrapper: ensure text-only path / `--language-model-only`; multimodal processor
   configs present but unused for GSM8K.
3. Confirm verl 0.9.0 ≥ PR #5682; if not, bump verl.
4. mRoPE + partial rotary + MTP are non-standard vs. plain Qwen3 — engine version
   sensitivity is higher; pin exact vLLM that has the GDN autotuner.

## 6. Thinking mode & sampling

**Thinking is OFF by default.** Template tail:
```jinja
{%- if enable_thinking is defined and enable_thinking is true %}
    {{- '<think>\n' }}
{%- else %}
    {{- '<think>\n\n</think>\n\n' }}   # empty think block → no reasoning
{%- endif %}
```
Default render injects an **empty `<think></think>`** → good for GSM8K token budget.
To force-disable at serve time: `--default-chat-template-kwargs '{"enable_thinking": false}'`.
To enable: pass `enable_thinking=true` in chat-template kwargs. The **nothink** hash
match (`273d8e0e…`) is consistent with this default.

**Sampling:** `generation_config.json` is **absent** (404) — no vendor defaults
shipped. Use Qwen3-family non-thinking defaults: `temperature≈0.7, top_p≈0.8,
top_k≈20, repetition_penalty≈1.0` (or greedy/`temperature 0` for GSM8K
deterministic eval). `eos_token = <|im_end|>`, `pad_token = <|endoftext|>`,
`model_max_length 262144`.

## 7. Existing toolkit mentions of Qwen3.5

Repo already treats `qwen3_5` as a known family:
- `rollout_gateway/response_schemas.py`: `QWEN3_5_SCHEMA` + hashes for
  Qwen3.5 (think/nothink), Qwen3.6, Nemotron-3, Qwen3-Coder.
- `tests/rollout_gateway/test_template_hashes_live.py`: live-tests
  `Qwen/Qwen3.5-0.8B` (nothink), `Qwen/Qwen3.5-27B` (think), `Qwen/Qwen3.6-27B`,
  `Qwen/Qwen3-Coder-30B-A3B-Instruct` → all `qwen3_5`. **2B is NOT yet listed** —
  consider adding `("Qwen/Qwen3.5-2B", "qwen3_5")` to that test.
- `backends/verl/README.md` + migration_agent examples reference
  `Qwen3_5ForConditionalGeneration` / the `qwen3_5` XML schema and the vision_config
  caveat for context-parallel.

## Recommendation

Qwen3.5-2B is **usable for RL/GRPO** and needs **no rollout-gateway change** (schema
already matches). The friction is purely engine/deps: the GDN kernels
(`flash-linear-attention` + `causal-conv1d`), the VL wrapper (`--language-model-only`),
and confirming verl 0.9.0 carries the Qwen3.5 FSDP-GRPO support. If any of those block
you, fall back to **Qwen3-4B-Instruct-2507** — already validated in this repo, plain
`qwen3` schema, dense full-attention, no vision tower, no GDN kernels — or Qwen3.5-4B
for the same family at slightly larger scale.
