#!/usr/bin/env python3
"""Holds a short scripted conversation with `tfswarm ... --serve --record FILE`
(TASK-16), so the recording has a chat to replay. Standard library only.

Usage:
  python3 tools/ref/record_chat.py --base http://127.0.0.1:8092 [--max-tokens 160] [--thinking] [--effort low]
"""
import argparse
import json
import urllib.request

TURNS = [
    "In two or three sentences, what is a mixture-of-experts language model?",
    "Why does it matter which machine each expert lives on when the model is split across computers?",
    "Give me a one-line summary of what we discussed.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--thinking", action="store_true", help="let Qwen3 reason first (off by default to keep replies short)")
    ap.add_argument("--effort", help="reasoning_effort for models that always reason, such as GLM-5.3-Flash")
    args = ap.parse_args()
    msgs = []
    for turn in TURNS:
        msgs.append({"role": "user", "content": turn})
        body = {"messages": msgs, "max_tokens": args.max_tokens, "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": args.thinking}}
        if args.effort:
            body["chat_template_kwargs"]["reasoning_effort"] = args.effort
        req = urllib.request.Request(args.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            out = json.load(r)
        m = out["choices"][0]["message"]
        t = out["tfswarm"]
        print(f"> {turn}\n{m['content'].strip()[:300]}\n  ({out['usage']['completion_tokens']} tokens, first token {t['ttft_ms']:.0f} ms, "
              f"{t['decode_tokens_per_s']:.1f} tokens/s, finish {out['choices'][0]['finish_reason']})\n")
        msgs.append({"role": "assistant", "content": m["content"]})


if __name__ == "__main__":
    main()
