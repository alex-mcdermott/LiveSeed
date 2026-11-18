#!/usr/bin/env python
"""abss_core_tokens_gpt.py: the core words of every prompt for the ABSS baseline, asked from GPT-5.1 as ABSS does.

ABSS (Zhang, Liu & Hong, NeurIPS 2026, Appendix A) extracts, once per prompt, "up to three core words that denote the main
objects or subjects that would occupy most of the image" with GPT-5.1, using the instruction and the two worked examples
quoted verbatim below (from their paper's TeX source). This script sends each prompt to GPT-5.1 with that template, keeps
the raw answer, and writes the core words in the file format of the authors' released code, which abss_statistic.py reads.

Input   --prompts  a prompt list with the columns ID and prompt.
Output  --out      a json {"<id>": {"adjective": [...], "verb": [...], "entity": [...], "other": [...]}}. The lists hold
                   1-based positions among the whitespace-split words of the prompt (the format of the authors'
                   initno_core_token_index_dict.json). "entity" holds the core words and is what the statistic uses.
                   Each word GPT returns is matched to the prompt's words: first unused occurrence, case-insensitive,
                   punctuation stripped as the authors' split_prompt_words does.
        <out>.raw.jsonl   the raw answers, one line per prompt (id, prompt, answer json, model, seconds), so that the
                   matching can be redone without asking again.

Usage (the key is read from the environment or from a file you name; it is never printed or written anywhere):
  OPENAI_API_KEY=... python abss_core_tokens_gpt.py --prompts ../../prompts/ABSS_DrawBench.csv --out runs/tokens/drawbench_gpt.json
  python abss_core_tokens_gpt.py --prompts ... --out ... --key-file /path/to/keyfile   (first line that looks like a key)
  python abss_core_tokens_gpt.py --compare runs/tokens/initno_gpt.json <ABSS code>/initno_core_token_index_dict.json
Options: --model gpt-5.1 (default), --limit N (the first N prompts only, for a quick test of access), --resume (reuse the
answers already in <out>.raw.jsonl instead of asking again). --compare OURS THEIRS asks nothing: it prints how far two
positions files agree on "entity" and writes the figures to OURS with the suffix .agreement.json.
"""
from __future__ import annotations
import argparse, csv, json, os, re, sys, time
from pathlib import Path

# The instruction and the two worked examples of the ABSS paper, verbatim.
INSTRUCTION = ("Assuming you are an expert in text-to-image prompting and visual semantics, given a text-to-image prompt, identify up to three core words "
               "that denote the main objects or subjects that would occupy most of the image. Output these as core_tokens. Then label the remaining content "
               "words into adjectives, verbs, and prepositions (ignore articles and stopwords).")
EXAMPLES = [
    ("Kneeling cat knight portrait finely detailed armor intricate design silver silk cinematic lighting",
     {"core_tokens": ["cat", "knight", "armor"], "adjectives": ["kneeling", "portrait", "finely", "detailed", "intricate", "silver", "silk", "cinematic"], "verbs": [], "prepositions": []}),
    ("A galactic eldritch squid towering over the planet Earth stars galaxies and nebulas in the background photorealistic 8k",
     {"core_tokens": ["squid", "Earth", "nebulas"], "adjectives": ["galactic", "eldritch", "towering", "planet", "photorealistic", "8k"], "verbs": [], "prepositions": ["over", "in"]}),
]


# ----------------------------------------------------------------------------- words and positions
def norm(w: str) -> str:
    """Lower-cases a word and strips the punctuation around it (the authors' normalize_word)."""
    return re.sub(r"^[^\w]+|[^\w]+$", "", w.strip().lower())


def split_words(prompt: str):
    """The prompt's words in order: split on whitespace, normalised, empty strings dropped (the authors' split_prompt_words)."""
    return [w for w in (norm(x) for x in prompt.strip().split()) if w]


def positions(words, targets):
    """Returns (sorted 1-based positions in `words` of the target words, the targets that were not found).

    Each target takes the first occurrence that is not yet used. A target that is not a whole word is tried as a
    prefix/suffix match (plural, hyphenation), then as the closest unused word; if all three fail it is reported as missing.
    """
    used, out, missing = set(), [], []
    for t in targets:
        t = norm(str(t))
        if not t:
            continue
        hit = next((i for i, w in enumerate(words) if w == t and i not in used), None)
        if hit is None:
            hit = next((i for i, w in enumerate(words) if (w.startswith(t) or t.startswith(w)) and min(len(w), len(t)) >= 3 and i not in used), None)
        if hit is None:      # DrawBench's misspelt prompts ("Bzaseball galove"): GPT answers the intended word; take the closest unused word
            import difflib
            cands = [w for i, w in enumerate(words) if i not in used]
            close = difflib.get_close_matches(t, cands, n=1, cutoff=0.6)
            if close:
                hit = next(i for i, w in enumerate(words) if w == close[0] and i not in used)
        if hit is None:
            missing.append(t)
            continue
        used.add(hit)
        out.append(hit + 1)
    return sorted(out), missing


# ----------------------------------------------------------------------------- the GPT query
def read_key(key_file):
    """The API key: the environment variable OPENAI_API_KEY if it is set, else the first key-like string in `key_file`.

    Exits with a message when neither gives a key.
    """
    k = os.environ.get("OPENAI_API_KEY", "").strip()
    if k:
        return k
    if key_file and Path(key_file).exists():
        for line in Path(key_file).read_text(encoding="utf-8").splitlines():
            m = re.search(r"(sk-[A-Za-z0-9_\-]{20,})", line)
            if m:
                return m.group(1)
    raise SystemExit("no API key: set OPENAI_API_KEY or pass --key-file")


def ask(client, model, prompt, retries=5):
    """Asks for the core words of one prompt. Returns (the answer as a dict, seconds taken, the model name the API reports).

    The instruction is the system message and the two worked examples are given as earlier turns of the conversation.
    The call is retried, with a growing pause, up to `retries` times.
    """
    msgs = [{"role": "system", "content": INSTRUCTION + " Answer with one JSON object with keys core_tokens, adjectives, verbs, prepositions and nothing else."}]
    for p, a in EXAMPLES:
        msgs.append({"role": "user", "content": p})
        msgs.append({"role": "assistant", "content": json.dumps(a)})
    msgs.append({"role": "user", "content": prompt})
    last = None
    for attempt in range(retries):
        try:
            t0 = time.time()
            r = client.chat.completions.create(model=model, messages=msgs, response_format={"type": "json_object"})
            txt = r.choices[0].message.content
            return json.loads(txt), time.time() - t0, getattr(r, "model", model)
        except Exception as e:           # rate limits, transient errors, a malformed answer
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"GPT call failed after {retries} tries: {type(last).__name__}: {last}")


# ----------------------------------------------------------------------------- agreement between two positions files
def compare(ours, theirs):
    """Prints and returns how far two positions files agree on "entity", over the prompt ids they share.

    Returned: n (shared prompts), identical (prompts with the same entity set), any_overlap (prompts sharing at least one
    entity), jaccard (shared entities / all entities, pooled over prompts) and diffs (the prompts that differ).
    """
    a = json.load(open(ours))
    b = json.load(open(theirs))
    ids = sorted(set(a) & set(b), key=int)
    same = sum(sorted(a[i]["entity"]) == sorted(b[i]["entity"]) for i in ids)
    inter = sum(len(set(a[i]["entity"]) & set(b[i]["entity"])) for i in ids)
    union = sum(len(set(a[i]["entity"]) | set(b[i]["entity"])) for i in ids)
    anyov = sum(bool(set(a[i]["entity"]) & set(b[i]["entity"])) for i in ids)
    print(f"{len(ids)} prompts in common: identical entity sets on {same} ({100 * same / len(ids):.1f}%); at least one shared entity on {anyov} ({100 * anyov / len(ids):.1f}%); "
          f"Jaccard {inter / max(union, 1):.3f}; mean |ours| {sum(len(a[i]['entity']) for i in ids) / len(ids):.2f}, |theirs| {sum(len(b[i]['entity']) for i in ids) / len(ids):.2f}")
    diffs = [(i, a[i]["entity"], b[i]["entity"]) for i in ids if sorted(a[i]["entity"]) != sorted(b[i]["entity"])]
    for d in diffs[:15]:
        print("  differs:", d)
    return dict(n=len(ids), identical=same, any_overlap=anyov, jaccard=inter / max(union, 1), diffs=diffs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts")
    ap.add_argument("--out")
    ap.add_argument("--model", default="gpt-5.1")
    ap.add_argument("--key-file", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--compare", nargs=2, metavar=("OURS", "THEIRS"))
    a = ap.parse_args()

    # ---- comparison mode: no query
    if a.compare:
        res = compare(*a.compare)
        Path(a.compare[0]).with_suffix(".agreement.json").write_text(json.dumps(res, indent=1))
        return

    from openai import OpenAI
    client = OpenAI(api_key=read_key(a.key_file))
    with open(a.prompts, encoding="utf-8-sig", newline="") as f:
        rows = [(str(r["ID"]).strip(), str(r["prompt"]).strip()) for r in csv.DictReader(f)]
    if a.limit:
        rows = rows[: a.limit]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw_path = out.with_suffix(".raw.jsonl")

    # ---- with --resume, the answers already in the raw file are reused, so those prompts are not asked (or paid for) twice
    done = {}
    if a.resume and raw_path.exists():
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
                done[d["id"]] = d
            except Exception:
                pass

    # ---- ask, then turn the answered words into positions among the prompt's words
    result, n_missing, t_all = {}, 0, time.time()
    with open(raw_path, "a", encoding="utf-8") as raw:
        for k, (pid, prompt) in enumerate(rows):
            if pid in done:
                ans = done[pid]["answer"]
            else:
                ans, dt, used_model = ask(client, a.model, prompt)
                # Written and flushed at once, so that an interrupted run keeps every answer it received.
                raw.write(json.dumps(dict(id=pid, prompt=prompt, answer=ans, model=used_model, seconds=round(dt, 2))) + "\n")
                raw.flush()
            words = split_words(prompt)
            ent, miss = positions(words, (ans.get("core_tokens") or [])[:3])      # at most three core words, as ABSS specifies
            adj, m2 = positions(words, ans.get("adjectives") or [])
            verb, m3 = positions(words, ans.get("verbs") or [])
            # Every word position belongs to exactly one list, with the precedence entity, adjective, verb, other.
            adj = [i for i in adj if i not in ent]
            verb = [i for i in verb if i not in ent and i not in adj]
            other = [i for i in range(1, len(words) + 1) if i not in ent and i not in adj and i not in verb]
            result[pid] = {"adjective": adj, "verb": verb, "entity": ent, "other": other}
            if miss:
                n_missing += 1
                print(f"[gpt] {pid}: core words not found in the prompt: {miss} (prompt: {prompt[:80]})", flush=True)
            if (k + 1) % 25 == 0:
                print(f"[gpt] {k + 1}/{len(rows)} ({time.time() - t_all:.0f}s)", flush=True)
    json.dump(result, open(out, "w", encoding="utf-8"), indent=1)
    empty = [k for k, v in result.items() if not v["entity"]]
    print(f"[gpt] wrote {out}: {len(result)} prompts, mean {sum(len(v['entity']) for v in result.values()) / max(len(result), 1):.2f} entity words, "
          f"{len(empty)} without any {empty[:10]}, {n_missing} prompts with an unmatched core word; model {a.model}; {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
