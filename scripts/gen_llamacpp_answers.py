"""Answers for the blind-judged test from GGUF files served by stock llama.cpp (llama-server).

    python scripts/gen_llamacpp_answers.py --name lcpp_12b_q4km --gguf exports/gemma4-12b-Q4_K_M.gguf
    python scripts/gen_llamacpp_answers.py --name lcpp_pack64_q3km --gguf "exports/gemma4-26b-{domain}-64e-Q3_K_M.gguf"

A --gguf containing {domain} is a per-domain pack: each domain's questions go to its own file
(the "own" configuration). Same 300 questions as scripts/gen_judge_answers.py, greedy decoding,
thinking off, at most 320 new tokens. Writes results/judge/answers_<name>.jsonl (resumable) and
records each file's size in results/judge/gguf_sizes.json.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
J = ROOT / "results" / "judge"
SERVER = Path(os.environ["USERPROFILE"]) / "tools" / "llama.cpp-b12549" / "cuda" / "llama-server.exe"
PORT = 8091


def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})  # fmt: skip
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def serve(gguf: Path, parallel: int) -> subprocess.Popen:
    p = subprocess.Popen([str(SERVER), "-m", str(gguf), "-ngl", "99", "-np", str(parallel), "-c", str(2048 * parallel),
                          "--jinja", "-rea", "off", "--port", str(PORT), "--host", "127.0.0.1"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # fmt: skip
    for _ in range(600):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                if r.status == 200:
                    return p
        except Exception:
            pass
        if p.poll() is not None:
            sys.exit(f"llama-server exited while loading {gguf}")
        time.sleep(1)
    p.kill()
    sys.exit(f"llama-server did not come up for {gguf}")


def ask(q: dict, max_new: int) -> dict:
    r = post("/v1/chat/completions", {"messages": [{"role": "user", "content": q["question"]}],
                                       "temperature": 0, "max_tokens": max_new, "seed": 0})  # fmt: skip
    text = (r["choices"][0]["message"].get("content") or "").strip()
    return {"id": q["id"], "answer": text, "tokens": int(r["usage"]["completion_tokens"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=320)
    args = ap.parse_args()

    qs = [json.loads(l) for l in (J / "questions.jsonl").read_text(encoding="utf-8").splitlines()]
    out = J / f"answers_{args.name}.jsonl"
    done = {json.loads(l)["id"] for l in out.read_text(encoding="utf-8").splitlines()} if out.exists() else set()
    todo = [q for q in qs if q["id"] not in done]
    per_domain = "{domain}" in args.gguf
    groups = {}
    for q in todo:
        groups.setdefault(q["domain"] if per_domain else "all", []).append(q)

    sizes_path = J / "gguf_sizes.json"
    sizes = json.loads(sizes_path.read_text(encoding="utf-8")) if sizes_path.exists() else {}
    for key, group in groups.items():
        gguf = ROOT / args.gguf.format(domain=key)
        sizes.setdefault(args.name, {})[key] = round(gguf.stat().st_size / 1e9, 3)
        sizes_path.write_text(json.dumps(sizes, indent=1), encoding="utf-8")
        print(f"=== {args.name} [{key}] {len(group)} questions  {gguf.name}", flush=True)
        srv = serve(gguf, args.parallel)
        try:
            t0 = time.time()
            with ThreadPoolExecutor(args.parallel) as ex:
                res = list(ex.map(lambda q: ask(q, args.max_new), group))
            with out.open("a", encoding="utf-8") as f:
                for r in res:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  {len(res)} answers in {time.time() - t0:.0f}s", flush=True)
        finally:
            srv.kill()
            srv.wait()


if __name__ == "__main__":
    main()
