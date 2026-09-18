#!/usr/bin/env python3
"""Regression for long image conversations against a live vision server.

The server counts only the images that are *not* already covered by the live KV
prefix against the per-request image limit.  A client that replays a growing
transcript (the normal agent loop) keeps every historical image in the request,
so the old total-count limit rejected the first turn past 16 images.

Run against an otherwise idle vision server:

    ./ds4-server -m gguf/Qwen3.8-Flash-Next-Q4.gguf \
        --ctx 250000 --mtp --host 0.0.0.0 \
        --vision gguf/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf
    python3 tests/test_server_vision_growth.py --url http://127.0.0.1:8000
"""

import argparse
import base64
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen3.8-flash-next")
    parser.add_argument("--fixtures", default="qwen38")
    parser.add_argument("--images", type=int, default=20,
                        help="total images to accumulate in the transcript")
    args = parser.parse_args()

    fixtures = Path(__file__).resolve().parent / "vision-fixtures" / args.fixtures
    names = sorted(p.name for p in fixtures.iterdir()
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    assert names, "no image fixtures in %s" % fixtures

    def image(name):
        data = base64.b64encode((fixtures / name).read_bytes()).decode()
        return {"type": "image_url",
                "image_url": {"url": "data:image/png;base64," + data}}

    def ask(label, history):
        body = {"model": args.model, "messages": history, "temperature": 0,
                "max_tokens": 24, "stream": False}
        request = urllib.request.Request(
            args.url.rstrip("/") + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                reply = json.load(response)
        except urllib.error.HTTPError as exc:
            raise SystemExit("%s: HTTP %d: %s" %
                             (label, exc.code, exc.read().decode())) from exc
        usage = reply["usage"]
        cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
        print("%-12s total=%d images=%d cached=%d %.2fs" %
              (label, usage["total_tokens"], history_images(history), cached,
               time.monotonic() - started), flush=True)
        return reply["choices"][0]["message"]

    def history_images(history):
        return sum(1 for message in history
                   if isinstance(message.get("content"), list)
                   for part in message["content"]
                   if part.get("type") == "image_url")

    history = [{"role": "system", "content": "Answer briefly and accurately."},
               {"role": "user", "content": "Reply with exactly READY."}]
    history.append(ask("text", history))

    for i in range(args.images):
        history.append({"role": "assistant", "content": "OK."})
        history.append({"role": "user", "content": [
            {"type": "text", "text": "Acknowledge image %d with a single word." % (i + 1)},
            image(names[i % len(names)])]})
        ask("image-%02d" % (i + 1), history)

    total = history_images(history)
    assert total > 16, "fixture did not exceed the old 16-image limit (%d)" % total

    # A final turn that adds no image must still see the whole transcript.
    history.append({"role": "assistant", "content": "OK."})
    history.append({"role": "user", "content": "How many images did I send in total? Reply with just the number."})
    ask("final", history)

    # The same transcript must also work from a cold slot: nothing cached for
    # this new conversation, so every image has to be encoded from scratch.
    cold = [{"role": "system", "content": "A second, unrelated session."}]
    for i in range(args.images):
        cold.append({"role": "user", "content": [
            {"type": "text", "text": "Image %d." % (i + 1)},
            image(names[i % len(names)])]})
    cold.append({"role": "user", "content": "Acknowledge the last image with one word."})
    ask("cold", cold)

    print("PASS: %d images accepted both as a growing replay and as a cold prompt" % total)


if __name__ == "__main__":
    main()
