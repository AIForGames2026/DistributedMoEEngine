# Chat templates read from the GGUF (2026-11-18)

TASK-38. Each model's chat format was written by hand in `src/chat.zig` (tulu, qwen3, qwen3_release, qwen38 and glm), so a new model could not chat until someone wrote its format. Now the engine renders the GGUF's own Jinja template with an interpreter written in Zig. The hand-written formats stay as a fallback for a template the interpreter refuses.

## What was built

- **`src/jinja.zig`**, a Jinja interpreter in Zig, written for this task. Nothing is borrowed, so `THIRD_PARTY.md` is unchanged. It matches transformers' environment: Jinja2 with `trim_blocks` and `lstrip_blocks`, the loop controls, `raise_exception`, `strftime_now` (local time, through libc), transformers' own `tojson`, and the `generation` block.
  - **Statements:** if/elif/else; for (with tuple targets, an `if` filter, else, break and continue, and the loop variables including `previtem` and `nextitem`); set (of names and namespace attributes); macros with default arguments; comments.
  - **Expressions:** Python's values and operators, slices with steps, list, tuple and dict literals, and ternaries.
  - **Filters:** 28 names, aliases included, such as `tojson`, `trim`, `items`, `sort` and `unique`.
  - **Tests:** 21 of them, such as `defined`, `none`, `string`, `mapping` and `iterable`.
  - **String and dict methods:** such as `strip`, `split`, `startswith`, `get` and `items`.
  - **Semantics checked against Jinja2 3.1.6** in transformers' environment before writing:
    - Loop bodies get fresh variables on each pass.
    - Macros see top-level names but not their caller's.
    - Printed values look as Python prints them, floats included.
    - Whitespace follows the `lstrip_blocks` line-start rule.
    - A missing attribute of a defined value is undefined, while any attribute of an undefined value is an error.
- **Refusal:** a statement, filter or test outside the subset fails when the template is parsed, with its name and line, for example `the template uses the filter 'wordwrap', which is not supported`. A method outside it fails when called. A template's own `raise_exception` reaches the client as a 400 with the template's message.
- **`chat.Chat`** holds the GGUF's template, the hand-written format it matches (if any) and the special tokens. It renders messages, picks stop ids and says whether replies may hold reasoning. The template gets what transformers passes:
  - `messages`, with `tools` and `documents` as None;
  - `add_generation_prompt`, plus `enable_thinking` and `reasoning_effort` only when the client set them;
  - `bos_token`, `eos_token`, `unk_token` and `pad_token` from the GGUF.
  - Whether the reply opens inside `<think>` is now read from the rendered prompt, so it holds for any template.
  - A model with no hand-written format stops at EOS or the GGUF's end-of-turn token.
- **Server:** it renders through `Chat` and prints where its prompts come from ("the GGUF's chat template"). A template that cannot be rendered is reported at startup, and the built-in format is used if there is one.
- **`tftok`:**
  - `--render` prints a conversation's prompt text.
  - `--template FILE --config tokenizer_config.json` renders a template without a tokenizer, for models the engine cannot load yet.
  - `--chat` passes every field of each message to the template. gpt-oss, for example, reads `thinking`.
- **`tools/ref/check_chat.py`:**
  - It now compares prompt text, rendered by transformers' own `render_jinja_template` with the special tokens from the model's `tokenizer_config.json`.
  - When a `tokenizer.json` is present, it also compares token ids from Hugging Face `tokenizers`, as `apply_chat_template` tokenizes. Loading the tokenizer through transformers is no longer needed; it failed for GLM-5.3-Flash.
  - It has two new conversations with gpt-oss's `thinking` field: 20 conversations, 7 settings each.

## Checks on `gx10-02`

The settings cover system prompt or not, several turns, reasoning in and out of earlier turns, `enable_thinking` on, off and unset, `reasoning_effort` low, medium and high, and `add_generation_prompt` on and off. Logs are in `2026-11-18-chat-templates/logs/`, from `check.sh`.

| Template | Engine reads it from | Prompts identical | Refused by both | Token ids identical |
|---|---|---|---|---|
| OLMoE-1B-7B (tulu) | its GGUF | 140 of 140 | 0 | 140 of 140 |
| Qwen3-30B-A3B, the GGUF's older template | its GGUF | 140 of 140 | 0 | 140 of 140 |
| Qwen3-30B-A3B, Hugging Face's current template | `tokenizer_config.json` | 140 of 140 | 0 | 140 of 140 |
| Qwen3.6-35B-A3B (had no built-in format) | its GGUF | 140 of 140 | 14 | 126 of 126 |
| Qwen3.8-Flash-Next | vocab-only GGUF | 140 of 140 | 32 | 108 of 108 |
| GLM-5.3-Flash | its GGUF | 140 of 140 | 0 | 140 of 140 |
| Mixtral-8x7B-Instruct | vocab-only GGUF | 140 of 140 | 7 | 133 of 133 |
| DeepSeek V3 | vocab-only GGUF | 140 of 140 | 0 | 140 of 140 |
| gpt-oss-20b | vocab-only GGUF | 140 of 140 | 0 | 140 of 140 |
| Llama 4 Scout | template file | 140 of 140 | 0 | (tokenizer not supported yet) |
| Kimi K2 | template file | 140 of 140 | 0 | (tokenizer needs Moonshot's code) |

- **Refusals:** each was a `raise_exception` in the template, and the engine refused the same conversations: a system turn after the first (Qwen3.6, Qwen3.8, Mixtral), no user query (Qwen3.8), or a reasoning effort the template does not take.
- **Jinja on every run:** no run fell back to a built-in format. Qwen3.6, Mixtral, DeepSeek V3, gpt-oss, Llama 4 and Kimi K2 have no built-in format at all.
- **gpt-oss's template prints today's date:** both sides ran on the same machine and day.
- **Templates of models the engine cannot load:** Llama 4's comes from the ungated unsloth copy of Meta's repository. Kimi K2's comes from Moonshot's. Their prompts are compared as text.

**Server, end to end:** Qwen3-30B-A3B served with `--gpu --serve` from its GGUF's template ("prompts from the GGUF's chat template"). On three conversations, with thinking default, off and on and with an earlier turn's reasoning, the server's prompt ids equal `tftok`'s (`serve_check.py`, `logs/serve_check.log`), so they also equal transformers'.

**Unit tests:** 128 of 128 on `gx10-02`, 110 plus 1 skipped on Windows. The interpreter's tests pin the semantics found with Jinja2 above, plus the refusals. A `Chat` test covers special tokens and a refused template.

## Changed behavior

- Prompts now come from the GGUF's template for every model that has one. For OLMoE, Qwen3, Qwen3.8 and GLM-5.3-Flash the text is unchanged: both paths match transformers on all 140 prompts.
- `reasoning_effort` is now passed to the template only when the client sends it. Qwen3.8's built-in format no longer refuses a high effort when thinking is off, as its template does not.

## Not done

- Tool calls stay out, as planned: `tools` is always None.
- The interpreter does not support `include`, `import`, `extends`, `call` blocks, block `set`, `recursive` loops or `*args`. A template using any of them falls back to the built-in format or is refused by name.
- Llama 4 and Kimi K2 are checked as templates only. Their tokenizers (`llama4` and `kimi-k2` pre-tokenizers) are not supported, and neither are the models.
