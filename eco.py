"""Toy ecosystem experiment: two small models with different data histories that
interact, versus controls. Pure NumPy (no torch needed)

Task: modular addition y = (a + b) mod P on all PxP pairs.
See PREREG.md for hypotheses and decision rules.
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # scripts read results/ relative to the repo root, wherever they are launched from
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np

P = 53
HALF = 27  # region split for mode="region": region 0 is a < HALF


# ----------------------------------------------------------------- data
def all_pairs(p=P):
    a, b = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
    a, b = a.ravel(), b.ravel()
    return a, b, (a + b) % p


def onehot(a, b, p=P):
    X = np.zeros((len(a), 2 * p), dtype=np.float32)
    X[np.arange(len(a)), a] = 1.0
    X[np.arange(len(a)), p + b] = 1.0
    return X


def make_split(seed, mode, fa, fb, fu, fv, p=P):
    """Disjoint index pools into the list of all p*p pairs.
    Returns dict with S_A, S_B (labeled), U (unlabeled), Val, Test."""
    rng = np.random.RandomState(seed)
    a, _, _ = all_pairs(p)
    N = p * p
    nA, nB, nU, nV = int(fa * N), int(fb * N), int(fu * N), int(fv * N)
    if mode == "random":
        idx = rng.permutation(N)
        S_A = idx[:nA]
        S_B = idx[nA:nA + nB]
        rest = idx[nA + nB:]
    elif mode == "region":
        r0 = rng.permutation(np.where(a < HALF)[0])
        r1 = rng.permutation(np.where(a >= HALF)[0])
        S_A, S_B = r0[:nA], r1[:nB]
        rest = rng.permutation(np.concatenate([r0[nA:], r1[nB:]]))
    else:
        raise ValueError(mode)
    return dict(S_A=S_A, S_B=S_B, U=rest[:nU], Val=rest[nU:nU + nV], Test=rest[nU + nV:])


# ----------------------------------------------------------------- model
def softmax(o):
    o = o - o.max(1, keepdims=True)
    e = np.exp(o)
    return e / e.sum(1, keepdims=True)


class MLP:
    """one-hot(2P) -> Linear(H) -> act -> Linear(P)."""

    def __init__(self, H, seed, act="square", init_scale=0.5, p=P):
        r = np.random.RandomState(seed)
        self.act = act
        self.W1 = (r.randn(2 * p, H) * init_scale / np.sqrt(2)).astype(np.float32)
        self.b1 = np.zeros(H, np.float32)
        self.W2 = (r.randn(H, p) / np.sqrt(H)).astype(np.float32)
        self.b2 = np.zeros(p, np.float32)

    def params(self):
        return dict(W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    def forward(self, X):
        z = X @ self.W1 + self.b1
        h = z * z if self.act == "square" else np.maximum(z, 0)
        return h @ self.W2 + self.b2, (X, z, h)

    def backward(self, do, cache):
        X, z, h = cache
        dh = do @ self.W2.T
        dz = dh * (2 * z if self.act == "square" else (z > 0))
        return dict(W1=X.T @ dz, b1=dz.sum(0), W2=h.T @ do, b2=do.sum(0))

    def hidden(self, X):
        return self.forward(X)[1][2]


class AdamW:
    def __init__(self, model, lr, wd, b1=0.9, b2=0.98, eps=1e-8):
        self.m, self.lr, self.wd, self.b1, self.b2, self.eps = model, lr, wd, b1, b2, eps
        self.mm = {k: np.zeros_like(v) for k, v in model.params().items()}
        self.vv = {k: np.zeros_like(v) for k, v in model.params().items()}
        self.t = 0

    def step(self, g):
        self.t += 1
        for k, p in self.m.params().items():
            self.mm[k] = self.b1 * self.mm[k] + (1 - self.b1) * g[k]
            self.vv[k] = self.b2 * self.vv[k] + (1 - self.b2) * g[k] ** 2
            mh = self.mm[k] / (1 - self.b1 ** self.t)
            vh = self.vv[k] / (1 - self.b2 ** self.t)
            if k in ("W1", "W2"):
                p -= self.lr * self.wd * p
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


# ----------------------------------------------------------------- metrics
def acc(model, X, y):
    return float((model.forward(X)[0].argmax(1) == y).mean())


def lin_cka(A, B):
    A = A - A.mean(0)
    B = B - B.mean(0)
    num = np.linalg.norm(B.T @ A, "fro") ** 2
    den = np.linalg.norm(A.T @ A, "fro") * np.linalg.norm(B.T @ B, "fro")
    return float(num / (den + 1e-12))


def js_div(p, q):
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (np.log(a + 1e-12) - np.log(b + 1e-12))).sum(1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


# ----------------------------------------------------------------- training
def train_models(models, labeled, XU, cfg, inter=None, eval_fn=None, yU=None):
    """Train models jointly. labeled: list of (X, y) per model.
    inter in {None, 'self', 'dml', 'agree', 'challenge'}.
    Distillation only ever uses XU (unlabeled inputs); no labels from Val/Test."""
    opts = [AdamW(m, cfg["lr"], cfg["wd"]) for m in models]
    nU = 0 if XU is None else len(XU)
    snap = None
    history = []
    for t in range(cfg["steps"]):
        outs, caches = [], []
        for m, (Xl, yl) in zip(models, labeled):
            Xin = Xl if inter is None else np.concatenate([Xl, XU])
            o, c = m.forward(Xin)
            outs.append(o)
            caches.append(c)
        active = inter is not None and t >= cfg["warm"]
        if active:
            lam = cfg["lam"] * min(1.0, (t - cfg["warm"] + 1) / cfg["ramp"])
            pU = [softmax(o[len(l[1]):]) for o, l in zip(outs, labeled)]
            if inter == "oracle":  # diagnostic only: peer with perfect knowledge of U labels
                oh = np.eye(P, dtype=np.float32)[yU]
                teachers, w = [oh] * len(models), np.ones(nU, np.float32)
            elif inter == "self":
                if snap is None or (t - cfg["warm"]) % cfg["self_refresh"] == 0:
                    snap = [p.copy() for p in pU]
                teachers, w = snap, np.ones(nU, np.float32)
            else:
                teachers = [pU[1], pU[0]]
                js = js_div(pU[0], pU[1])
                if inter == "dml":
                    w = np.ones(nU, np.float32)
                elif inter == "challenge":
                    w = js / (js.mean() + 1e-8)
                elif inter == "agree":
                    s = 1.0 - js / np.log(2)
                    w = s / (s.mean() + 1e-8)
                else:
                    raise ValueError(inter)
                w = np.clip(w, 0, cfg["wclip"]).astype(np.float32)
        for i, (m, (Xl, yl)) in enumerate(zip(models, labeled)):
            nl = len(yl)
            do = np.zeros_like(outs[i])
            pl = softmax(outs[i][:nl])
            pl[np.arange(nl), yl] -= 1.0
            do[:nl] = pl / nl
            if active:
                do[nl:] = lam * w[:, None] * (pU[i] - teachers[i]) / nU
            opts[i].step(m.backward(do, caches[i]))
        if eval_fn is not None and (t % cfg["log_every"] == 0 or t == cfg["steps"] - 1):
            history.append((t, eval_fn(models)))
    return history


# ----------------------------------------------------------------- one seed
def run_seed(seed, mode, cfg, conditions=None):
    a, b, y = all_pairs()
    X = onehot(a, b)
    sp = make_split(seed, mode, cfg["fa"], cfg["fb"], cfg["fu"], cfg["fv"])
    XA, yA = X[sp["S_A"]], y[sp["S_A"]]
    XB, yB = X[sp["S_B"]], y[sp["S_B"]]
    XU = X[sp["U"]]
    Xp, yp = np.concatenate([XA, XB]), np.concatenate([yA, yB])
    Xt, yt, Xv, yv = X[sp["Test"]], y[sp["Test"]], X[sp["Val"]], y[sp["Val"]]
    reg_t = (a[sp["Test"]] >= HALF).astype(int)  # 0 = region of A, 1 = region of B

    def per_model(m, own_region=None):
        r = dict(test=acc(m, Xt, yt), val=acc(m, Xv, yv))
        pred = m.forward(Xt)[0].argmax(1)
        ok = pred == yt
        r["test_r0"], r["test_r1"] = float(ok[reg_t == 0].mean()), float(ok[reg_t == 1].mean())
        return r

    def pair_metrics(mA, mB):
        pa, pb = softmax(mA.forward(Xt)[0]), softmax(mB.forward(Xt)[0])
        ens = float(((pa + pb).argmax(1) == yt).mean())
        return dict(ens_test=ens,
                    disagree=float((pa.argmax(1) != pb.argmax(1)).mean()),
                    cka=lin_cka(mA.hidden(Xt), mB.hidden(Xt)))

    def mk(H, s):
        return MLP(H, seed * 100 + s, act=cfg["act"], init_scale=cfg["init_scale"])

    def eval_pair(models):
        return [acc(m, Xt, yt) for m in models]

    conds = conditions or ["solo", "pooled_small", "pooled_big", "self_teacher", "dml", "agree_only", "challenge"]
    out = {}
    inter_map = {"oracle": "oracle", "self_teacher": "self", "dml": "dml", "agree_only": "agree", "challenge": "challenge"}

    for c in conds:
        if c == "solo":
            mA, mB = mk(cfg["H"], 1), mk(cfg["H"], 2)
            hist = train_models([mA, mB], [(XA, yA), (XB, yB)], None, cfg, None, eval_pair)
            out[c] = dict(A=per_model(mA), B=per_model(mB), **pair_metrics(mA, mB), hist=hist)
        elif c in ("pooled_small", "pooled_big"):
            H = cfg["H"] if c == "pooled_small" else cfg["Hbig"]
            m = mk(H, 3 if c == "pooled_small" else 4)
            ccfg = cfg if c == "pooled_small" else {**cfg, **cfg.get("big_over", {})}
            hist = train_models([m], [(Xp, yp)], None, ccfg, None, eval_pair)
            out[c] = dict(M=per_model(m), hist=hist)
        elif c in inter_map:
            ccfg = {**cfg, **cfg.get("cond_over", {}).get(c, {})}
            mA, mB = mk(cfg["H"], 1), mk(cfg["H"], 2)  # same init as solo -> paired
            hist = train_models([mA, mB], [(XA, yA), (XB, yB)], XU, ccfg, inter_map[c], eval_pair, yU=y[sp["U"]])
            out[c] = dict(A=per_model(mA), B=per_model(mB), **pair_metrics(mA, mB), hist=hist)
        elif c == "dml_same":
            # control: both models see the SAME labeled data (S_A); only init differs. Same distillation as dml.
            ccfg = {**cfg, **cfg.get("cond_over", {}).get("dml", {})}
            mA, mB = mk(cfg["H"], 1), mk(cfg["H"], 2)
            hist = train_models([mA, mB], [(XA, yA), (XA, yA)], XU, ccfg, "dml", eval_pair)
            out[c] = dict(A=per_model(mA), B=per_model(mB), **pair_metrics(mA, mB), hist=hist)
        elif c == "solo_long":
            # control: same as solo but 4x the training steps (does solo just need more time?)
            ccfg = {**cfg, "steps": cfg["steps"] * 4, "log_every": cfg["log_every"] * 4}
            mA, mB = mk(cfg["H"], 1), mk(cfg["H"], 2)
            hist = train_models([mA, mB], [(XA, yA), (XB, yB)], None, ccfg, None, eval_pair)
            out[c] = dict(A=per_model(mA), B=per_model(mB), **pair_metrics(mA, mB), hist=hist)
        elif c == "solo_same":
            mA, mB = mk(cfg["H"], 1), mk(cfg["H"], 2)
            hist = train_models([mA, mB], [(XA, yA), (XA, yA)], None, cfg, None, eval_pair)
            out[c] = dict(A=per_model(mA), B=per_model(mB), **pair_metrics(mA, mB), hist=hist)
        else:
            raise ValueError(c)
    return out