# SentencePiece and more pre-tokenizers (2026-11-18)

TASK-37. The tokenizer read byte-level BPE with 4 pre-tokenizer patterns (`olmo`, `qwen2`, `qwen35`, `glm4`), each added with its model. It now also reads SentencePiece vocabularies (Mixtral) and the patterns of DeepSeek V3 (`deepseek-v3`) and gpt-oss (`gpt-4o`), the families TASK-39 adds, all from the GGUF alone. Anything else is refused by name.

## What was built (`src/tokenizer.zig`)

- **SentencePiece** (`tokenizer.ggml.model` "llama"): spaces become U+2581, and one goes in front of the text as `tokenizer.ggml.metaspace.prepend_scheme` says. Mixtral's says "first": only the very start, not after an added token. BPE merges characters, and a character that is not in the vocabulary becomes its UTF-8 bytes' `<0xXX>` tokens. Decoding turns U+2581 back into spaces and drops one space at the start, as Hugging Face's decoder does.
- **Merges from scores.** A SentencePiece GGUF stores scores, not merges. Every split of a vocabulary token into two vocabulary tokens is a merge, ranked by the token's score, then by its id, then by the left and right ids. For Mixtral this gives exactly Hugging Face's 58,980 merges in the same order, which was checked against its `tokenizer.json` before writing the code.
- **`deepseek-v3`:** three splits in turn, as Hugging Face's `Sequence` of `Split`s runs them: numbers in runs of up to three, then runs of CJK ideographs and kana, then the main pattern. That pattern keeps ASCII punctuation with the letters that follow and uses the punctuation and symbol categories. Text that no alternative matches becomes a piece of its own, as in Hugging Face.
- **`gpt-4o`:** words split by case. Optional capitals then lower-case letters, or capitals then optional lower-case letters, each with an optional contraction. Numbers come in runs of up to three, and punctuation keeps a trailing `/` or line break. As with `glm4`, merges are ignored: a piece that is a whole token is that token.
- **Unicode tables** (`tools/ref/gen_unicode.py`, Unicode 15.0 as before) gain upper (Lu, Lt), lower (Ll), other letters (Lm, Lo), punctuation (P) and symbols (S). The existing tables are unchanged.
- **BOS:** `tokenizer.ggml.add_bos_token` is read. `tftok` starts with BOS when it is set, as Hugging Face's `encode` does (Mixtral). The chat path still takes BOS from the template, which is TASK-38's.
- **Normalization from the pattern:** a GGUF does not record the normalizer. OLMoE's and Qwen's are NFC; GLM, DeepSeek V3, gpt-oss and SentencePiece have none.
- **Refusal:** an unknown `tokenizer.ggml.pre` or tokenizer model fails with its name and the supported list, for example `pre-tokenizer "llama-bpe" is not supported (olmo, qwen2, qwen35, glm4, deepseek-v3 and gpt-4o are)`.

No llama.cpp code was borrowed. The patterns and the SentencePiece path follow Hugging Face's `tokenizers`, which is the reference, so `THIRD_PARTY.md` is unchanged.

## Test vocabularies

The new families' GGUFs are hundreds of GB, and the check needs only the tokenizer. `vocab.sh` fetches each model's Hugging Face tokenizer files (Mixtral-8x7B-Instruct-v0.1, DeepSeek-V3, gpt-oss-20b; 37 MB together), and llama.cpp's own converter (`convert_hf_to_gguf.py --vocab-only`, commit `ffc87c3a` as in TASK-36) turns them into vocab-only GGUFs on `gx10-02`. The converter chose `default` with model "llama" for Mixtral, `deepseek-v3` and `gpt-4o`, as published GGUFs have.

For the models already supported, the check reads their own GGUFs. Qwen3.8-Flash-Next's GGUF was deleted after TASK-31. Its converter needs the checkpoint even for the vocabulary, so its vocab-only GGUF was made from the checkpoint already in `gx10-02`'s Hugging Face cache. OLMoE's `tokenizer.json` was on none of the machines and was fetched for this check.

## Checks on `gx10-02`

`tools/ref/check_tokenizer.py` compares `tftok`'s ids with Hugging Face's `encode` on 5,038 texts per model: the 3 golden prompts, 32 edge cases, 5,000 seeded random strings, and three long files (`src/tokenizer.zig`, `Plan.md`, `README.md`). It also checks that decoding gives the text back. 9 of the edge cases are new for this task: case splits, paths, digit runs, CJK runs, ASCII punctuation before letters, literal U+2581, `<s>` and `</s>` in text, and control characters. Logs are in `2026-11-18-tokenizers/logs/`, from `check.sh`.

| Model | From | Texts identical | Tokens |
|---|---|---|---|
| Mixtral-8x7B (SentencePiece) | vocab-only GGUF | 5,038 of 5,038 | 186,981 |
| DeepSeek V3 (`deepseek-v3`) | vocab-only GGUF | 5,038 of 5,038 | 160,573 |
| gpt-oss-20b (`gpt-4o`) | vocab-only GGUF | 5,038 of 5,038 | 153,347 |
| OLMoE-1B-7B (`olmo`) | its Q8_0 GGUF | 5,038 of 5,038 | 169,431 |
| Qwen3-30B-A3B (`qwen2`) | its Q8_0, Q4_K_M and IQ4_XS GGUFs | 5,038 of 5,038 each | 144,621 |
| Qwen3.6-35B-A3B (`qwen35`) | its Q8_0 GGUF | 5,038 of 5,038 | 153,567 |
| Qwen3.8-Flash-Next (`qwen35`) | vocab-only GGUF | 5,038 of 5,038 | 153,567 |
| GLM-5.3-Flash (`glm4`) | its GGUF | 5,038 of 5,038 | 152,144 |

Every decode round trip passed. Loaded from `tokenizer.json` instead, OLMoE, Qwen3, Qwen3.6 and GLM-5.3-Flash also match on all 5,038.

**Unit tests:** 122 of 122 on `gx10-02`, 104 plus 1 skipped on Windows. The new tests cover the two patterns' splits on strings whose pieces come from Hugging Face's `pre_tokenize_str`, the refusal, and SentencePiece on a GGUF built in the test (merge order, byte fallback, a space in front of the first piece only, BOS and decoding).

**Chat prompts** (`tools/ref/check_chat.py`), to see that nothing changed:
- OLMoE: 126 of 126.
- Qwen3: 126 of 126 with the GGUF's own template. Against the newer template on Hugging Face it gives 119 of 126, the same as the build before this task.
- Qwen3.6: its template is not recognized, the same as before.
- GLM-5.3-Flash: the installed transformers cannot load its tokenizer class. Chat templates are TASK-38's.

## Not done

- Mixtral's chat template is not recognized yet; that is TASK-38's.
- `fromJson` (loading a `tokenizer.json` directly) still supports only the older four setups. The new ones load from the GGUF, which is what the engine runs.
