#!/usr/bin/env python3
"""Concurrent greedy chat parity and aggregate throughput probe (stdlib only)."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import time
import urllib.request


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--url', default='http://127.0.0.1:8080/v1/chat/completions')
    ap.add_argument('--reference-url', help='optional B=1 server endpoint for the greedy reference')
    ap.add_argument('--reference-file', help='saved greedy replies from a previous B=1 run')
    ap.add_argument('--save-references', help='write the sequential greedy references as JSON')
    ap.add_argument('--model', default='vegajet')
    ap.add_argument('-n', '--requests', type=int, default=4)
    ap.add_argument('--max-tokens', type=int, default=128)
    ap.add_argument('--prompt', action='append', help='repeat to mix prompts')
    ap.add_argument('--stream', action='store_true')
    ap.add_argument('--timeout', type=float, default=600)
    args = ap.parse_args()
    if args.requests < 1:
        ap.error('requests must be positive')
    prompts = args.prompt or ['Write a short Python function to compute Fibonacci numbers.']

    def request(prompt, url=None):
        body = dict(model=args.model, messages=[dict(role='user', content=prompt)],
                    temperature=0, max_tokens=args.max_tokens, stream=args.stream)
        if args.stream:
            body['stream_options'] = {'include_usage': True}
        req = urllib.request.Request(url or args.url, json.dumps(body).encode(),
                                     {'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=args.timeout) as response:
            if not args.stream:
                data = json.load(response)
                msg = data['choices'][0]['message']
                return (msg.get('reasoning_content', '') or '') + (msg.get('content', '') or ''), data['usage']['completion_tokens']
            pieces, tokens = [], 0
            for line in response:
                if not line.startswith(b'data: '):
                    continue
                payload = line[6:].strip()
                if payload == b'[DONE]':
                    break
                data = json.loads(payload)
                if 'error' in data:
                    raise RuntimeError(data['error'])
                tokens = (data.get('usage') or {}).get('completion_tokens', tokens)
                for choice in data.get('choices', []):
                    delta = choice.get('delta', {})
                    pieces.append((delta.get('reasoning_content', '') or '') + (delta.get('content', '') or ''))
            return ''.join(pieces), tokens

    # Establish each greedy reference with exactly one outstanding request.
    if args.reference_file:
        refs = json.loads(Path(args.reference_file).read_text())
        if any(p not in refs for p in prompts):
            ap.error('reference file does not cover every prompt')
    else:
        refs = {p: request(p, args.reference_url)[0] for p in prompts}
    if args.save_references:
        Path(args.save_references).write_text(json.dumps(refs, ensure_ascii=False, indent=2) + '\n')
    work = [prompts[i % len(prompts)] for i in range(args.requests)]
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.requests) as pool:
        replies = list(pool.map(request, work))
    elapsed = time.perf_counter() - start
    agree = sum(reply[0] == refs[prompt] for reply, prompt in zip(replies, work))
    total = sum(reply[1] for reply in replies)
    print(f'{agree}/{args.requests} greedy replies match; {total} tokens / {elapsed:.3f}s = {total / elapsed:.2f} tok/s aggregate')
    if not total:
        print('Server omitted SSE token usage; use non-streaming mode for throughput.')
    raise SystemExit(0 if agree == args.requests else 1)


if __name__ == '__main__':
    main()
