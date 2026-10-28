#!/usr/bin/env python3
"""Score a release checkpoint on the manual-seat validation corpus (offline).

`release_train.py` only scores `in_distribution` (a random 10% holdout of the
TRAINING period) plus two general targets. Model selection needs a future
period, and nothing in the repo evaluates one. This does: it loads a release
checkpoint with the same loader the server uses, runs it over a Case JSONL, and
writes both a summary and one row per case so two arms can be paired later.

    python scripts/eval_val.py --ckpt CKPT --val val.jsonl --out EVAL.json

`--ckpt` is a release directory (tower/scorer/meta.json). A checkpoint trained
here was saved WITHOUT embeddings (save_release), so it is not self-contained:
`meta["base_model"]` must be reachable. If `Qwen/Qwen3.5-4B-Base` is not on the
Hub cache, point `--base-model` at a self-contained HF directory that carries
the same frozen embeddings (e.g. the converted v6.1 tower) and only the tower's
embedding lookup is taken from there -- every other weight comes from CKPT.

Validation size: `--subset N` takes a deterministic, per-(day,label) stratified
subset. 2,000-3,000 rows is enough to pick an epoch; write the ids once with
`--subset-file` and reuse the same list for every arm.

Comparison: `--compare OTHER.rows.jsonl` pairs two runs on (case_id, key) and
reports McNemar (exact) plus a cluster bootstrap over `uid` -- the same uid
appears on several days, so per-row bootstrap would understate the interval.

    python scripts/eval_val.py --ckpt arm-a --val val.jsonl --out a.json
    python scripts/eval_val.py --ckpt arm-b --val val.jsonl --out b.json \\
        --compare a.json.rows.jsonl
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import random
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rsijev.contract import gold_label, load_cases            # noqa: E402
from rsijev.evaluate import predict, score_predictions        # noqa: E402
from serve.release import load_release, resolve_ckpt          # noqa: E402


def split_id(case_id: str) -> tuple[str, str]:
    """"2026-10-16:<uid>:0" -> ("2026-10-16", uid); anything else -> ("", "")."""
    parts = case_id.split(":")
    if len(parts) >= 3:
        return parts[0], parts[1]
    return "", ""


def pick_subset(cases, n: int | None, seed: int) -> list[int]:
    """Deterministic proportional sample per (day, gold label) stratum."""
    if not n or n >= len(cases):
        return list(range(len(cases)))
    strata: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, c in enumerate(cases):
        q = c.questions[0]
        strata[(split_id(c.case_id)[0], gold_label(q, c.gold[q.key]))].append(i)
    if n < len(strata):                      # keep at least one per stratum
        n = len(strata)
    total = len(cases)
    quota, frac = {}, {}
    for k, idx in strata.items():
        exact = n * len(idx) / total
        quota[k] = min(len(idx), max(1, int(exact)))
        frac[k] = exact - int(exact)
    # hand out / take back the rounding remainder
    while sum(quota.values()) > n:
        k = max(quota, key=lambda k: (quota[k] > 1, quota[k]))
        quota[k] -= 1
    for k in sorted(frac, key=lambda k: -frac[k]):
        if sum(quota.values()) >= n:
            break
        if quota[k] < len(strata[k]):
            quota[k] += 1
    rng = random.Random(seed)
    picked: list[int] = []
    for k in sorted(strata):
        picked += rng.sample(sorted(strata[k]), quota[k])
    return sorted(picked)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p from the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def cluster_bootstrap(rows_a: dict, rows_b: dict, n_boot: int = 2000,
                      seed: int = 17) -> tuple[float, float]:
    """95% CI of acc(A) - acc(B), resampling uids (not rows)."""
    by_uid: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in rows_a:
        by_uid[rows_a[key]["uid"]].append(key)
    uids = sorted(by_uid)
    if len(uids) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    deltas = []
    for _ in range(n_boot):
        keys: list[tuple[str, str]] = []
        for _ in uids:
            keys += by_uid[uids[rng.randrange(len(uids))]]
        da = sum(rows_a[k]["correct"] for k in keys) / len(keys)
        db = sum(rows_b[k]["correct"] for k in keys) / len(keys)
        deltas.append(da - db)
    deltas.sort()
    lo = deltas[int(0.025 * n_boot)]
    hi = deltas[min(n_boot - 1, int(0.975 * n_boot))]
    return (lo, hi)


def load_with_base(ckpt: Path, base: str | None, device: str, dtype):
    """load_release, but with meta["base_model"] overridden for checkpoints that
    were saved without embeddings and whose base is not on the Hub."""
    if base is None:
        return load_release(resolve_ckpt(str(ckpt)), device, infer_dtype=dtype)
    with tempfile.TemporaryDirectory(prefix="eval-val-") as tmp:
        t = Path(tmp)
        for name in ("tower.safetensors", "scorer.safetensors", "aux_scorers.safetensors",
                     "calibration.json", "calibration.safetensors"):
            if (ckpt / name).is_file():
                (t / name).symlink_to((ckpt / name).resolve())
        meta = json.loads((ckpt / "meta.json").read_text())
        meta["base_model"] = base
        (t / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        return load_release(t, device, infer_dtype=dtype)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="release checkpoint directory")
    ap.add_argument("--val", required=True, help="Case JSONL (one case per line)")
    ap.add_argument("--out", default=None, help="summary JSON (default: <ckpt>/val_eval.json)")
    ap.add_argument("--rows-out", default=None, help="per-case JSONL (default: <out>.rows.jsonl)")
    ap.add_argument("--orders", default="canonical", help="comma list: canonical,reversed")
    ap.add_argument("--subset", type=int, default=None,
                    help="stratified subset size; omit for the full corpus")
    ap.add_argument("--subset-file", default=None,
                    help="read case ids from here (or write the chosen ids here on first run)")
    ap.add_argument("--seed", type=int, default=17, help="subset sampling seed")
    ap.add_argument("--limit", type=int, default=None, help="cap rows after subsetting (debug)")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--base-model", default=None,
                    help="override meta['base_model'] (a local HF directory)")
    ap.add_argument("--compare", default=None,
                    help="another run's rows JSONL; pairs on (case_id, key)")
    args = ap.parse_args()

    ckpt = Path(args.ckpt).expanduser().resolve()
    out = Path(args.out) if args.out else ckpt / "val_eval.json"
    rows_out = Path(args.rows_out) if args.rows_out else Path(str(out) + ".rows.jsonl")

    cases = load_cases(str(Path(args.val).expanduser()))
    if args.subset_file and Path(args.subset_file).is_file():
        keep = {line.strip() for line in Path(args.subset_file).read_text().splitlines() if line.strip()}
        idx = [i for i, c in enumerate(cases) if c.case_id in keep]
        print(f"[subset] {len(idx)} cases read from {args.subset_file}", flush=True)
    else:
        idx = pick_subset(cases, args.subset, args.seed)
        if args.subset_file:
            Path(args.subset_file).write_text(
                "\n".join(cases[i].case_id for i in idx) + "\n", encoding="utf-8")
            print(f"[subset] {len(idx)} ids written to {args.subset_file}", flush=True)
    if args.limit:
        idx = idx[: args.limit]
    cases = [cases[i] for i in idx]
    print(f"[data] {len(cases)} cases from {args.val}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    model, tok, enc, meta = load_with_base(ckpt, args.base_model, device, dtype)
    spec = meta["spec"]
    print(f"[model] base={meta.get('base_model')} steps={spec.get('steps')} "
          f"own_tokens={enc.option_pool_own_tokens} max_length={enc.max_length} "
          f"tower={args.dtype}", flush=True)

    batch_size = args.batch_size or int(spec.get("eval_batch_size") or 32)
    orders = [o.strip() for o in args.orders.split(",") if o.strip()]
    rows: dict[tuple[str, str], dict] = {}
    summary = {"ckpt": str(ckpt), "val": args.val, "n": len(cases),
               "dtype": args.dtype, "orders": orders, "per_order": {}}

    for order in orders:
        e = dataclasses.replace(enc, option_order=order)
        # The tower precision is already set by load_release(infer_dtype=...);
        # passing eval_dtype here would try to cast the fp32 scorer too.
        preds = predict(model, tok, cases, e, max_options=int(spec["max_options"]),
                        device=device, batch_size=batch_size, eval_dtype=None)
        rep = score_predictions(preds, target="manual_seat_v02", split="val")
        for mode, mr in rep.per_mode.items():
            summary["per_order"].setdefault(order, {})[mode] = {
                "n": mr.n, "accuracy": round(mr.accuracy, 4),
                "base_rate": round(mr.base_rate, 4), "ece": round(mr.ece, 4)}
        correct = 0
        per_day: dict[str, list[int]] = defaultdict(list)
        recall: dict[str, list[int]] = defaultdict(list)
        for c, q, p in preds:
            gold = gold_label(q, c.gold[q.key])
            pred = q.options[max(range(len(p.probs)), key=p.probs.__getitem__)]
            day, uid = split_id(c.case_id)
            ok = int(pred == gold)
            correct += ok
            per_day[day].append(ok)
            recall[gold].append(ok)
            rows[(c.case_id, q.key)] = {
                "case_id": c.case_id, "key": q.key, "order": order, "day": day,
                "uid": uid, "gold": gold, "pred": pred, "correct": ok,
                "confidence": round(p.confidence(), 6)}
        lo, hi = wilson(correct, len(preds))
        summary["per_order"][order]["pooled"] = {
            "n": len(preds), "accuracy": round(correct / len(preds), 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "per_day": {d: round(sum(v) / len(v), 4) for d, v in sorted(per_day.items())},
            "recall": {g: round(sum(v) / len(v), 4) for g, v in sorted(recall.items())},
        }
        print(f"[{order}] acc {correct / len(preds):.4f} "
              f"[{lo:.4f}, {hi:.4f}] n={len(preds)}", flush=True)

    cards = defaultdict(list)
    for r in rows.values():
        cards[r["gold"]].append(r["correct"])
    summary["confusion"] = {g: round(sum(v) / len(v), 4) for g, v in sorted(cards.items())}

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    rows_out.parent.mkdir(parents=True, exist_ok=True)
    with open(rows_out, "w", encoding="utf-8") as fh:
        for k in sorted(rows):
            fh.write(json.dumps(rows[k], ensure_ascii=False) + "\n")
    print(f"[done] {out}\n       rows {rows_out}", flush=True)

    if args.compare:
        other_raw = [json.loads(line) for line in
                     Path(args.compare).read_text(encoding="utf-8").splitlines() if line.strip()]
        other = {(r["case_id"], r["key"]): r for r in other_raw}
        mine = {k: v for k, v in rows.items() if k in other}
        if not mine:
            print("[compare] no shared (case_id, key) rows", flush=True)
            return 1
        b = sum(1 for k in mine if mine[k]["correct"] and not other[k]["correct"])
        c = sum(1 for k in mine if not mine[k]["correct"] and other[k]["correct"])
        acc_a = sum(mine[k]["correct"] for k in mine) / len(mine)
        acc_b = sum(other[k]["correct"] for k in mine) / len(mine)
        lo, hi = cluster_bootstrap(mine, other)
        res = {"n_paired": len(mine), "acc_a": round(acc_a, 4), "acc_b": round(acc_b, 4),
               "delta": round(acc_a - acc_b, 4), "delta_ci95_cluster_uid": [round(lo, 4), round(hi, 4)],
               "mcnemar": {"a_only": b, "b_only": c, "p_exact": round(mcnemar_exact(b, c), 6)}}
        summary["compare"] = res
        print(f"[compare] acc {acc_a:.4f} vs {acc_b:.4f}  delta {acc_a - acc_b:+.4f} "
              f"[{lo:+.4f}, {hi:+.4f}] (uid cluster)  McNemar p={res['mcnemar']['p_exact']:.4g} "
              f"(b={b}, c={c})", flush=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
