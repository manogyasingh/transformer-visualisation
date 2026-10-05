"""Train a tiny GPT-2-style transformer on a toy corpus with plain NumPy.

The architecture matches GPT-2 exactly (pre-LayerNorm blocks, learned absolute
position embeddings, tanh-GELU MLP, biases everywhere, tied input/output
embeddings) but is small enough that every number can be shown on screen.

Forward and backward passes are written out by hand; the backward pass is
verified against finite differences before training starts.

Usage:
    python3 training/train-tiny-gpt.py                   # train from scratch
    python3 training/train-tiny-gpt.py --reference-only  # only recompute test gradients

Writes src/model/tiny-gpt-weights.json, which the visualiser loads.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "src" / "model" / "tiny-gpt-weights.json"

VOCAB = [
    "the", "cat", "dog", "bird", "sat", "on", "ate", "chased",
    "mat", "rug", "branch", "fish", "bone", "seed", "mouse", "ball", "bug", ".",
]

# (sentence, number of copies in the training set). The counts set the
# conditional distributions the model should learn, e.g. after
# "the cat sat on the" the next token is "mat" 3/4 of the time and "rug" 1/4.
CORPUS = [
    ("the cat sat on the mat .", 3),
    ("the cat sat on the rug .", 1),
    ("the cat ate the fish .", 3),
    ("the cat ate the mouse .", 1),
    ("the cat chased the mouse .", 3),
    ("the cat chased the bird .", 1),
    ("the dog sat on the rug .", 3),
    ("the dog sat on the mat .", 1),
    ("the dog ate the bone .", 3),
    ("the dog ate the fish .", 1),
    ("the dog chased the ball .", 2),
    ("the dog chased the cat .", 2),
    ("the bird sat on the branch .", 4),
    ("the bird ate the seed .", 2),
    ("the bird ate the bug .", 2),
    ("the bird chased the bug .", 4),
]

PRESET_PROMPTS = [
    "the cat sat on the",
    "the dog sat on the",
    "the bird sat on the",
    "the dog chased the",
    "the cat ate the",
    "the bird",
    "the",
    "the cat sat on the mat",
]

CONFIG = {
    "n_ctx": 6,
    "d_model": 8,
    "n_heads": 2,
    "d_ff": 32,
    "n_layers": 2,
    "ln_eps": 1e-5,
}

GELU_C = math.sqrt(2.0 / math.pi)


def encode(text):
    return [VOCAB.index(w) for w in text.split()]


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

def init_params(rng, cfg):
    d, f, v = cfg["d_model"], cfg["d_ff"], len(VOCAB)
    p = {
        "wte": rng.normal(0, 0.3, (v, d)),
        "wpe": rng.normal(0, 0.3, (cfg["n_ctx"], d)),
        "lnf.g": np.ones(d),
        "lnf.b": np.zeros(d),
    }
    for l in range(cfg["n_layers"]):
        pre = f"h{l}."
        p[pre + "ln1.g"] = np.ones(d)
        p[pre + "ln1.b"] = np.zeros(d)
        for name in ("wq", "wk", "wv", "wo"):
            p[pre + "attn." + name] = rng.normal(0, 1 / math.sqrt(d), (d, d))
            p[pre + "attn.b" + name[1]] = np.zeros(d)
        p[pre + "ln2.g"] = np.ones(d)
        p[pre + "ln2.b"] = np.zeros(d)
        p[pre + "mlp.w1"] = rng.normal(0, 1 / math.sqrt(d), (d, f))
        p[pre + "mlp.b1"] = np.zeros(f)
        p[pre + "mlp.w2"] = rng.normal(0, 1 / math.sqrt(f), (f, d))
        p[pre + "mlp.b2"] = np.zeros(d)
    return p


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def layer_norm(x, g, b, eps):
    mu = x.mean(-1, keepdims=True)
    xc = x - mu
    var = (xc ** 2).mean(-1, keepdims=True)  # biased variance, as in PyTorch
    rstd = 1.0 / np.sqrt(var + eps)
    xhat = xc * rstd
    return g * xhat + b, (xhat, rstd, g)


def layer_norm_backward(dy, cache):
    xhat, rstd, g = cache
    d = xhat.shape[-1]
    dg = (dy * xhat).reshape(-1, d).sum(0)
    db = dy.reshape(-1, d).sum(0)
    dxhat = dy * g
    dx = rstd * (
        dxhat
        - dxhat.mean(-1, keepdims=True)
        - xhat * (dxhat * xhat).mean(-1, keepdims=True)
    )
    return dx, dg, db


def gelu(u):
    t = np.tanh(GELU_C * (u + 0.044715 * u ** 3))
    return 0.5 * u * (1.0 + t), t


def gelu_backward(dout, u, t):
    dt = (1.0 - t ** 2) * GELU_C * (1.0 + 3 * 0.044715 * u ** 2)
    return dout * (0.5 * (1.0 + t) + 0.5 * u * dt)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def forward(p, ids, cfg):
    B, T = ids.shape
    d, H = cfg["d_model"], cfg["n_heads"]
    dh = d // H
    eps = cfg["ln_eps"]
    causal = np.tril(np.ones((T, T), dtype=bool))

    def split(m):
        return m.reshape(B, T, H, dh).transpose(0, 2, 1, 3)

    x = p["wte"][ids] + p["wpe"][:T]
    caches = []
    for l in range(cfg["n_layers"]):
        pre = f"h{l}."
        a, ln1c = layer_norm(x, p[pre + "ln1.g"], p[pre + "ln1.b"], eps)
        q = a @ p[pre + "attn.wq"] + p[pre + "attn.bq"]
        k = a @ p[pre + "attn.wk"] + p[pre + "attn.bk"]
        v = a @ p[pre + "attn.wv"] + p[pre + "attn.bv"]
        qh, kh, vh = split(q), split(k), split(v)
        s = qh @ kh.transpose(0, 1, 3, 2) / math.sqrt(dh)
        s = np.where(causal, s, -np.inf)
        e = np.exp(s - s.max(-1, keepdims=True))
        att = e / e.sum(-1, keepdims=True)
        z = (att @ vh).transpose(0, 2, 1, 3).reshape(B, T, d)
        o = z @ p[pre + "attn.wo"] + p[pre + "attn.bo"]
        x_mid = x + o
        c, ln2c = layer_norm(x_mid, p[pre + "ln2.g"], p[pre + "ln2.b"], eps)
        u = c @ p[pre + "mlp.w1"] + p[pre + "mlp.b1"]
        g, t = gelu(u)
        m = g @ p[pre + "mlp.w2"] + p[pre + "mlp.b2"]
        caches.append(dict(a=a, ln1c=ln1c, qh=qh, kh=kh, vh=vh, att=att, z=z,
                           c=c, ln2c=ln2c, u=u, t=t, g=g))
        x = x_mid + m
    f, lnfc = layer_norm(x, p["lnf.g"], p["lnf.b"], eps)
    logits = f @ p["wte"].T
    return logits, dict(caches=caches, f=f, lnfc=lnfc)


def loss_and_grads(p, ids, targets, cfg):
    logits, cache = forward(p, ids, cfg)
    mask = targets >= 0
    n = mask.sum()
    shifted = logits - logits.max(-1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))
    safe_t = np.where(mask, targets, 0)
    nll = -np.take_along_axis(logp, safe_t[..., None], -1)[..., 0]
    loss = (nll * mask).sum() / n

    grads = {k: np.zeros_like(v) for k, v in p.items()}
    dlogits = np.exp(logp)
    np.put_along_axis(
        dlogits, safe_t[..., None],
        np.take_along_axis(dlogits, safe_t[..., None], -1) - 1.0, -1,
    )
    dlogits *= mask[..., None] / n

    grads["wte"] += np.einsum("btv,btd->vd", dlogits, cache["f"])
    backward_from_final(p, ids, cache, dlogits @ p["wte"], cfg, grads)
    return loss, grads


def backward_from_final(p, ids, cache, df, cfg, grads):
    """Adds to `grads` the gradients of every weight, given df = dLoss/d(final LayerNorm output)."""
    B, T = ids.shape
    d, H = cfg["d_model"], cfg["n_heads"]
    dh = d // H
    dx, grads["lnf.g"], grads["lnf.b"] = layer_norm_backward(df, cache["lnfc"])

    def merge(m):
        return m.transpose(0, 2, 1, 3).reshape(B, T, d)

    for l in reversed(range(cfg["n_layers"])):
        pre = f"h{l}."
        c_ = cache["caches"][l]

        # x_out = x_mid + W2 · gelu(W1 · LN2(x_mid))
        dm = dx
        grads[pre + "mlp.w2"] = np.einsum("btf,btd->fd", c_["g"], dm)
        grads[pre + "mlp.b2"] = dm.sum((0, 1))
        du = gelu_backward(dm @ p[pre + "mlp.w2"].T, c_["u"], c_["t"])
        grads[pre + "mlp.w1"] = np.einsum("btd,btf->df", c_["c"], du)
        grads[pre + "mlp.b1"] = du.sum((0, 1))
        dc = du @ p[pre + "mlp.w1"].T
        dx_ln2, grads[pre + "ln2.g"], grads[pre + "ln2.b"] = layer_norm_backward(dc, c_["ln2c"])
        dx_mid = dx + dx_ln2

        # x_mid = x + Wo · attention(LN1(x))
        do = dx_mid
        grads[pre + "attn.wo"] = np.einsum("btk,btd->kd", c_["z"], do)
        grads[pre + "attn.bo"] = do.sum((0, 1))
        dzh = (do @ p[pre + "attn.wo"].T).reshape(B, T, H, dh).transpose(0, 2, 1, 3)
        att = c_["att"]
        datt = dzh @ c_["vh"].transpose(0, 1, 3, 2)
        dvh = att.transpose(0, 1, 3, 2) @ dzh
        ds = att * (datt - (datt * att).sum(-1, keepdims=True)) / math.sqrt(dh)
        dqh = ds @ c_["kh"]
        dkh = ds.transpose(0, 1, 3, 2) @ c_["qh"]
        da = np.zeros_like(c_["a"])
        for name, dproj in (("q", merge(dqh)), ("k", merge(dkh)), ("v", merge(dvh))):
            grads[pre + f"attn.w{name}"] = np.einsum("btd,bte->de", c_["a"], dproj)
            grads[pre + f"attn.b{name}"] = dproj.sum((0, 1))
            da += dproj @ p[pre + f"attn.w{name}"].T
        dx_ln1, grads[pre + "ln1.g"], grads[pre + "ln1.b"] = layer_norm_backward(da, c_["ln1c"])
        dx = dx_mid + dx_ln1

    np.add.at(grads["wte"], ids, dx)
    grads["wpe"][:T] += dx.sum(0)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_dataset(cfg):
    T = cfg["n_ctx"]
    rows_x, rows_y = [], []
    for text, count in CORPUS:
        ids = encode(text)
        assert len(ids) - 1 <= T, text
        x = np.zeros(T, dtype=np.int64)
        y = np.full(T, -1, dtype=np.int64)
        x[: len(ids) - 1] = ids[:-1]
        y[: len(ids) - 1] = ids[1:]
        for _ in range(count):
            rows_x.append(x)
            rows_y.append(y)
    return np.stack(rows_x), np.stack(rows_y)


def optimal_loss():
    """Cross-entropy of the true conditional distributions (the best any model can do)."""
    counts = {}
    for text, c in CORPUS:
        ids = encode(text)
        for i in range(1, len(ids)):
            prefix = tuple(ids[:i])
            counts.setdefault(prefix, {})
            counts[prefix][ids[i]] = counts[prefix].get(ids[i], 0) + c
    total, n = 0.0, 0
    for nxt in counts.values():
        z = sum(nxt.values())
        for c in nxt.values():
            total += -c * math.log(c / z)
            n += c
    return total / n


# ---------------------------------------------------------------------------
# Checks, training, export
# ---------------------------------------------------------------------------

def gradient_check(cfg):
    rng = np.random.default_rng(123)
    p = init_params(rng, cfg)
    x, y = build_dataset(cfg)
    x, y = x[::5], y[::5]
    _, grads = loss_and_grads(p, x, y, cfg)
    worst = 0.0
    h = 1e-6
    for name, value in p.items():
        for _ in range(4):
            idx = tuple(rng.integers(0, s) for s in value.shape)
            old = value[idx]
            value[idx] = old + h
            lp, _ = loss_and_grads(p, x, y, cfg)
            value[idx] = old - h
            lm, _ = loss_and_grads(p, x, y, cfg)
            value[idx] = old
            numeric = (lp - lm) / (2 * h)
            analytic = grads[name][idx]
            rel = abs(numeric - analytic) / max(1e-8, abs(numeric) + abs(analytic))
            worst = max(worst, rel)
    print(f"gradient check: worst relative error = {worst:.2e}")
    assert worst < 1e-5, "backward pass disagrees with finite differences"


def train(cfg, steps=6000, lr=1e-2, seed=0):
    rng = np.random.default_rng(seed)
    p = init_params(rng, cfg)
    x, y = build_dataset(cfg)
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(v) for k, v in p.items()}
    b1, b2, wd = 0.9, 0.99, 1e-3
    loss = float("nan")
    for step in range(1, steps + 1):
        loss, grads = loss_and_grads(p, x, y, cfg)
        lr_t = lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        for k in p:
            m[k] = b1 * m[k] + (1 - b1) * grads[k]
            v[k] = b2 * v[k] + (1 - b2) * grads[k] ** 2
            mh = m[k] / (1 - b1 ** step)
            vh = v[k] / (1 - b2 ** step)
            decay = wd * p[k] if p[k].ndim == 2 else 0.0
            p[k] -= lr_t * (mh / (np.sqrt(vh) + 1e-8) + decay)
        if step % 1000 == 0 or step == 1:
            print(f"step {step:5d}  loss {loss:.4f}")
    return p, loss


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def to_list(a):
    return np.asarray(a).tolist()


NESTED_KEYS = [
    ("ln1", "gamma", "ln1.g"), ("ln1", "beta", "ln1.b"),
    ("attn", "wq", "attn.wq"), ("attn", "bq", "attn.bq"),
    ("attn", "wk", "attn.wk"), ("attn", "bk", "attn.bk"),
    ("attn", "wv", "attn.wv"), ("attn", "bv", "attn.bv"),
    ("attn", "wo", "attn.wo"), ("attn", "bo", "attn.bo"),
    ("ln2", "gamma", "ln2.g"), ("ln2", "beta", "ln2.b"),
    ("mlp", "w1", "mlp.w1"), ("mlp", "b1", "mlp.b1"),
    ("mlp", "w2", "mlp.w2"), ("mlp", "b2", "mlp.b2"),
]


def nest(p, cfg):
    """Flat parameter dict -> the nested JSON layout used by the visualiser."""
    layers = []
    for l in range(cfg["n_layers"]):
        layer = {}
        for group, key, flat in NESTED_KEYS:
            layer.setdefault(group, {})[key] = to_list(p[f"h{l}.{flat}"])
        layers.append(layer)
    return {
        "wte": to_list(p["wte"]),
        "wpe": to_list(p["wpe"]),
        "layers": layers,
        "lnf": {"gamma": to_list(p["lnf.g"]), "beta": to_list(p["lnf.b"])},
    }


def unnest(n):
    p = {
        "wte": np.array(n["wte"]),
        "wpe": np.array(n["wpe"]),
        "lnf.g": np.array(n["lnf"]["gamma"]),
        "lnf.b": np.array(n["lnf"]["beta"]),
    }
    for l, layer in enumerate(n["layers"]):
        for group, key, flat in NESTED_KEYS:
            p[f"h{l}.{flat}"] = np.array(layer[group][key])
    return p


REFERENCE_SENTENCE = "the cat sat on the mat ."


def reference_gradients(p, cfg):
    """Loss and gradients for one sentence (mean over its positions), for testing the TS backward pass."""
    ids = np.array([encode(REFERENCE_SENTENCE)])
    loss, grads = loss_and_grads(p, ids[:, :-1], ids[:, 1:], cfg)
    return {"sentence": REFERENCE_SENTENCE, "loss": float(loss), "grads": nest(grads, cfg)}


def export(p, cfg, final_loss, rounded_loss, best_loss, steps):
    reference = []
    for prompt in PRESET_PROMPTS:
        ids = np.array([encode(prompt)])
        logits, _ = forward(p, ids, cfg)
        reference.append({"prompt": prompt, "ids": encode(prompt), "logits": to_list(logits[0, -1])})

    out = {
        "config": {
            "vocab": VOCAB,
            "nCtx": cfg["n_ctx"],
            "dModel": cfg["d_model"],
            "nHeads": cfg["n_heads"],
            "dHead": cfg["d_model"] // cfg["n_heads"],
            "dFF": cfg["d_ff"],
            "nLayers": cfg["n_layers"],
            "lnEps": cfg["ln_eps"],
        },
        "params": nest(p, cfg),
        "training": {
            "corpus": [{"text": t, "count": c} for t, c in CORPUS],
            "presetPrompts": PRESET_PROMPTS,
            "steps": steps,
            "finalLoss": final_loss,
            "roundedLoss": rounded_loss,
            "optimalLoss": best_loss,
        },
        "reference": reference,
        "referenceGrads": reference_gradients(p, cfg),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {OUT_PATH.relative_to(ROOT)}")


def main():
    cfg = CONFIG
    if "--reference-only" in sys.argv:
        data = json.loads(OUT_PATH.read_text())
        data["referenceGrads"] = reference_gradients(unnest(data["params"]), cfg)
        OUT_PATH.write_text(json.dumps(data, separators=(",", ":")))
        print(f"updated referenceGrads in {OUT_PATH.relative_to(ROOT)}")
        return
    gradient_check(cfg)
    steps = 6000
    p, final_loss = train(cfg, steps=steps)

    # Round every weight to 4 decimals so the numbers shown in the visualiser
    # are the exact weights the model uses.
    p = {k: np.round(v, 4) for k, v in p.items()}
    x, y = build_dataset(cfg)
    rounded_loss, _ = loss_and_grads(p, x, y, cfg)
    best = optimal_loss()
    print(f"loss after rounding weights: {rounded_loss:.4f} (optimal: {best:.4f})")

    for prompt in PRESET_PROMPTS:
        logits, _ = forward(p, np.array([encode(prompt)]), cfg)
        probs = softmax(logits[0, -1])
        top = np.argsort(-probs)[:3]
        desc = ", ".join(f"{VOCAB[i]} {probs[i]:.2f}" for i in top)
        print(f"  {prompt!r:28} -> {desc}")

    export(p, cfg, float(final_loss), float(rounded_loss), best, steps)


if __name__ == "__main__":
    main()
