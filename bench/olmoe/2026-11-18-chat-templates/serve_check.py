#!/usr/bin/env python3
"""TASK-38: the server's prompt token ids against tftok's for the same conversations.

tools/ref/check_chat.py shows tftok renders and tokenizes as transformers does; this
shows the server builds the same prompt. Standard library only.

Usage: serve_check.py BASE_URL TFTOK MODEL.gguf
"""
import json
import os
import subprocess
import sys
import tempfile
import urllib.request

CASES = [
    ({"messages": [{"role": "user", "content": "What is the capital of France?"}]}, {}),
    ({"messages": [{"role": "system", "content": "Answer in one sentence."}, {"role": "user", "content": "Why is the sky blue?"}]},
     {"enable_thinking": False}),
    ({"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!", "reasoning_content": "greet"},
                   {"role": "user", "content": "Count to three."}]}, {"enable_thinking": True}),
]


def main():
    base, tftok, model = sys.argv[1:4]
    bad = 0
    for conv, kwargs in CASES:
        body = {"model": "x", "messages": conv["messages"], "max_tokens": 24, "temperature": 0, "return_tokens": True}
        if kwargs:
            body["chat_template_kwargs"] = kwargs
        req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        reply = json.load(urllib.request.urlopen(req, timeout=600))
        server_ids = reply["tensorfold"]["prompt_ids"]
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "c.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump({**conv, "add_generation_prompt": True, **kwargs}, f)
            out = subprocess.run([tftok, model, "--chat", p], capture_output=True, text=True, check=True).stdout
        want = [int(x) for x in out.split()]
        same = server_ids == want
        bad += not same
        text = reply["choices"][0]["message"].get("content") or ""
        print(f"{'same' if same else 'DIFFERENT'}: {len(want)} prompt ids; reply {text[:60]!r}")
    print("OK: the server's prompts match tftok's" if not bad else f"FAIL: {bad} prompts differ")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
