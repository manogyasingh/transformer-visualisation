"""Train the reward model used by the post-training (RLHF with PPO) walkthrough.

Follows the reward-model recipe of Stiennon et al. (2020) and InstructGPT (Ouyang et al., 2022):

1. For a prompt x, sample K responses from the pretrained model, the policy PPO will start from.
2. A labeler ranks the K responses; every pair (y_w better than y_l) from a ranking is one comparison.
3. The reward model is the pretrained transformer with the unembedding replaced by a scalar head that
   reads the final hidden state of the last token, trained with the Bradley-Terry loss
   -log sigmoid(r(x, y_w) - r(x, y_l)).
4. The loss only sees reward differences, so afterwards the head's bias is shifted so that the
   pretrained model's responses score 0 on average.

The labeler is simulated: it ranks responses by sampling from a Plackett-Luce model (Bradley-Terry
generalised to rankings) over a hidden utility. It dislikes chasing and strongly dislikes nonsense
(ungrammatical sentences, or an object that does not fit the verb). The reward model only ever sees the
rankings, never the utility.

The reward model reads whole sentences (prompt + response, up to 7 tokens) while the policy only ever
reads 6, so it gets one extra position embedding, initialised as a copy of the last one.

Usage:
    python3 training/train-reward-model.py

Reads src/model/tiny-gpt-weights.json and writes src/model/reward-model-weights.json.
"""

import importlib.util
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BASE_PATH = ROOT / "src" / "model" / "tiny-gpt-weights.json"
OUT_PATH = ROOT / "src" / "model" / "reward-model-weights.json"

_spec = importlib.util.spec_from_file_location("tiny_gpt", Path(__file__).with_name("train-tiny-gpt.py"))
tiny = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tiny)

VOCAB = tiny.VOCAB
EOS = VOCAB.index(".")
PROMPTS = ["the", "the cat", "the dog", "the bird"]
MAX_LEN = 7  # prompt + response; the policy can read 6 tokens, so it can write at most a 7th
K = 4
N_RANKINGS = 1500
VAL_FRACTION = 0.2

SUBJECTS = {"cat", "dog", "bird"}
# What each verb can sensibly take: you sit on furniture or branches, eat food, chase animals or balls.
OBJECTS = {
    "sat": {"mat", "rug", "branch"},
    "ate": {"fish", "bone", "seed", "mouse", "bug"},
    "chased": {"cat", "dog", "bird", "mouse", "ball", "bug"},
}
UTILITY = {"fine": 0.0, "chasing": -2.0, "nonsense": -4.0}


def sensible(words):
    """"the S sat on the O ." or "the S ate|chased the O ." with an object that fits the verb."""
    if len(words) == 7:
        return (words[0] == "the" and words[1] in SUBJECTS and words[2:5] == ["sat", "on", "the"]
                and words[5] in OBJECTS["sat"] and words[6] == ".")
    if len(words) == 6:
        return (words[0] == "the" and words[1] in SUBJECTS and words[2] in ("ate", "chased")
                and words[3] == "the" and words[4] in OBJECTS[words[2]] and words[5] == ".")
    return False


def label(words):
    if not sensible(words):
        return "nonsense"
    return "chasing" if "chased" in words else "fine"


# ---------------------------------------------------------------------------
# Sampling responses from the pretrained model
# ---------------------------------------------------------------------------

def next_probs(p, ids, cfg, temperature=1.0):
    logits, _ = tiny.forward(p, np.array([ids]), cfg)
    z = logits[0, -1] / temperature
    e = np.exp(z - z.max())
    return e / e.sum()


def sample_response(rng, p, prompt_ids, cfg, temperature):
    """Samples until "." or the length limit; returns None if the response never ended."""
    ids = list(prompt_ids)
    while len(ids) < MAX_LEN:
        tok = int(rng.choice(len(VOCAB), p=next_probs(p, ids, cfg, temperature)))
        ids.append(tok)
        if tok == EOS:
            return ids
    return None


def enumerate_responses(p, prompt_ids, cfg, min_prob=1e-7):
    """Every finished response with probability above min_prob, with its probability."""
    out = []
    stack = [(list(prompt_ids), 1.0)]
    while stack:
        ids, prob = stack.pop()
        if ids[-1] == EOS and len(ids) > len(prompt_ids):
            out.append((ids, prob))
            continue
        if len(ids) >= MAX_LEN:
            continue
        probs = next_probs(p, ids, cfg)
        for tok in np.nonzero(probs * prob > min_prob)[0]:
            stack.append((ids + [int(tok)], prob * float(probs[tok])))
    return out


# ---------------------------------------------------------------------------
# Labeler: Plackett-Luce rankings over the hidden utility
# ---------------------------------------------------------------------------

def rank(rng, responses):
    """Returns the responses best-first, picking each next one with probability ∝ exp(utility)."""
    left = list(responses)
    order = []
    while left:
        u = np.array([UTILITY[label([VOCAB[i] for i in r])] for r in left])
        w = np.exp(u - u.max())
        k = int(rng.choice(len(left), p=w / w.sum()))
        order.append(left.pop(k))
    return order


def build_comparisons(rng, base, cfg):
    rankings = []
    for _ in range(N_RANKINGS):
        prompt = PROMPTS[int(rng.integers(len(PROMPTS)))]
        prompt_ids = tiny.encode(prompt)
        responses = []
        while len(responses) < K:
            # Like InstructGPT, which ranked samples from several models, a quarter of the samples
            # come from a much hotter sampler so that the reward model also sees nonsense
            # (at temperature 1 the pretrained model almost never writes it).
            temperature = 1.0 if rng.random() < 0.75 else 4.0
            r = sample_response(rng, base, prompt_ids, cfg, temperature)
            if r is not None:
                responses.append(tuple(r))
        rankings.append(rank(rng, responses))
    return rankings


def pairs_of(rankings):
    """Every (better, worse) pair from each ranking; pairs of identical sentences carry no information."""
    out = []
    for order in rankings:
        for i in range(len(order)):
            for j in range(i + 1, len(order)):
                if order[i] != order[j]:
                    out.append((order[i], order[j]))
    return out


# ---------------------------------------------------------------------------
# Reward model
# ---------------------------------------------------------------------------

def scores(p, seqs, cfg):
    """Scalar reward of each sequence, read at its last token, plus what backprop needs."""
    lengths = np.array([len(s) for s in seqs])
    ids = np.full((len(seqs), MAX_LEN), EOS, dtype=np.int64)
    for b, s in enumerate(seqs):
        ids[b, : len(s)] = s
    _, cache = tiny.forward(p, ids, cfg)
    h = cache["f"][np.arange(len(seqs)), lengths - 1]
    return h @ p["head.w"] + p["head.b"][0], (ids, cache, h, lengths)


def bt_loss_and_grads(p, pairs, cfg):
    seqs = sorted({s for pair in pairs for s in pair})
    index = {s: i for i, s in enumerate(seqs)}
    r, (ids, cache, h, lengths) = scores(p, seqs, cfg)
    w = np.array([index[a] for a, _ in pairs])
    l = np.array([index[b] for _, b in pairs])
    margin = r[w] - r[l]
    loss = np.mean(np.logaddexp(0.0, -margin))
    accuracy = np.mean(margin > 0)

    dmargin = -1.0 / (1.0 + np.exp(margin)) / len(pairs)
    dr = np.zeros(len(seqs))
    np.add.at(dr, w, dmargin)
    np.add.at(dr, l, -dmargin)

    grads = {k: np.zeros_like(v) for k, v in p.items()}
    grads["head.w"] = h.T @ dr
    grads["head.b"] = np.array([dr.sum()])
    df = np.zeros_like(cache["f"])
    df[np.arange(len(seqs)), lengths - 1] = dr[:, None] * p["head.w"]
    tiny.backward_from_final(p, ids, cache, df, cfg, grads)
    return loss, accuracy, grads


def init_reward_model(base, rng, cfg):
    p = {k: v.copy() for k, v in base.items()}
    p["wpe"] = np.vstack([base["wpe"], base["wpe"][-1:]])
    d = cfg["d_model"]
    p["head.w"] = rng.normal(0, 1 / math.sqrt(d + 1), d)
    p["head.b"] = np.zeros(1)
    return p


def gradient_check(p, pairs, cfg, rng):
    _, _, grads = bt_loss_and_grads(p, pairs, cfg)
    worst, h = 0.0, 1e-6
    for name, value in p.items():
        if name == "head.b":
            continue  # the loss is invariant to the bias, so its gradient is exactly 0
        for _ in range(3):
            idx = tuple(rng.integers(0, s) for s in value.shape)
            old = value[idx]
            value[idx] = old + h
            lp, _, _ = bt_loss_and_grads(p, pairs, cfg)
            value[idx] = old - h
            lm, _, _ = bt_loss_and_grads(p, pairs, cfg)
            value[idx] = old
            numeric = (lp - lm) / (2 * h)
            if abs(numeric) + abs(grads[name][idx]) < 1e-7:
                # Exactly zero: attention ignores key biases, and the final LayerNorm bias shifts
                # every reward equally, which the loss cannot see.
                continue
            rel = abs(numeric - grads[name][idx]) / (abs(numeric) + abs(grads[name][idx]))
            worst = max(worst, rel)
    print(f"gradient check: worst relative error = {worst:.2e}")
    assert worst < 1e-5, "reward model backward pass disagrees with finite differences"


def train(p, train_pairs, val_pairs, cfg, steps=300, lr=3e-3):
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(v) for k, v in p.items()}
    b1, b2 = 0.9, 0.99
    for step in range(1, steps + 1):
        loss, acc, grads = bt_loss_and_grads(p, train_pairs, cfg)
        lr_t = lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        for k in p:
            m[k] = b1 * m[k] + (1 - b1) * grads[k]
            v[k] = b2 * v[k] + (1 - b2) * grads[k] ** 2
            p[k] -= lr_t * (m[k] / (1 - b1 ** step)) / (np.sqrt(v[k] / (1 - b2 ** step)) + 1e-8)
        if step % 50 == 0 or step == 1:
            vl, va, _ = bt_loss_and_grads(p, val_pairs, cfg)
            print(f"step {step:4d}  train loss {loss:.4f} acc {acc:.3f}   val loss {vl:.4f} acc {va:.3f}")
    return p


def expected_reward(p, base, base_cfg, cfg):
    """Mean reward of the pretrained model's responses, averaged over the prompts (exact enumeration)."""
    total = 0.0
    for prompt in PROMPTS:
        resp = enumerate_responses(base, tiny.encode(prompt), base_cfg)
        r, _ = scores(p, [tuple(s) for s, _ in resp], cfg)
        probs = np.array([q for _, q in resp])
        total += float((r * probs).sum() / probs.sum())
    return total / len(PROMPTS)


def text(seq):
    return " ".join(VOCAB[i] for i in seq)


def main():
    rng = np.random.default_rng(0)
    base_json = json.loads(BASE_PATH.read_text())
    base = tiny.unnest(base_json["params"])
    base_cfg = tiny.CONFIG
    cfg = dict(base_cfg, n_ctx=MAX_LEN)

    rankings = build_comparisons(rng, base, base_cfg)
    n_val = int(len(rankings) * VAL_FRACTION)
    train_pairs, val_pairs = pairs_of(rankings[n_val:]), pairs_of(rankings[:n_val])
    print(f"{len(rankings)} rankings of K={K} -> {len(train_pairs)} train / {len(val_pairs)} val comparisons")
    labels = [label([VOCAB[i] for i in s]) for order in rankings for s in order]
    print("  labels of ranked responses:", {k: labels.count(k) for k in UTILITY})

    p = init_reward_model(base, rng, cfg)
    gradient_check(p, train_pairs[::10], cfg, rng)
    p = train(p, train_pairs, val_pairs, cfg)

    p = {k: np.round(v, 4) for k, v in p.items()}
    offset = round(expected_reward(p, base, base_cfg, cfg), 4)
    p["head.b"] = p["head.b"] - offset
    print(f"shifted the bias by {-offset:+.4f}; pretrained model's mean reward is now "
          f"{expected_reward(p, base, base_cfg, cfg):+.5f}")
    train_loss, train_acc, _ = bt_loss_and_grads(p, train_pairs, cfg)
    val_loss, val_acc, _ = bt_loss_and_grads(p, val_pairs, cfg)
    print(f"final: train loss {train_loss:.4f} acc {train_acc:.3f}   val loss {val_loss:.4f} acc {val_acc:.3f}")

    reference = [tiny.encode(t) for t, _ in tiny.CORPUS] + [
        tiny.encode(t) for t in ["the bird sat on the mat .", "the cat ate the ball .", "the cat sat the mouse .",
                                 "the dog dog .", "the ."]
    ]
    ref_scores, _ = scores(p, [tuple(s) for s in reference], cfg)
    for s, r in sorted(zip(reference, ref_scores), key=lambda x: -x[1]):
        print(f"  {r:+.3f}  {text(s)}")

    examples = []
    for order in rankings[:n_val]:
        if len(examples) >= 12:
            break
        best, worst = order[0], order[-1]
        if best != worst:
            examples.append({"chosen": text(best), "rejected": text(worst)})

    nested = tiny.nest(p, cfg)
    out = {
        "config": dict(base_json["config"], nCtx=MAX_LEN),
        "params": {
            "trunk": nested,
            "head": {"w": tiny.to_list(p["head.w"]), "b": tiny.to_list(p["head.b"])},
        },
        "prompts": PROMPTS,
        "labeler": {"k": K, "utility": UTILITY},
        "data": {
            "rankings": len(rankings),
            "trainComparisons": len(train_pairs),
            "valComparisons": len(val_pairs),
            "trainLoss": float(train_loss),
            "trainAccuracy": float(train_acc),
            "valLoss": float(val_loss),
            "valAccuracy": float(val_acc),
            "examples": examples,
        },
        "offset": offset,
        "reference": [{"text": text(s), "score": float(r)} for s, r in zip(reference, ref_scores)],
    }
    OUT_PATH.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
