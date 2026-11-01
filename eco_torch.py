"""PyTorch port of the training loop in eco.py.

Data splits, initial weights and metrics come from eco.py (NumPy) so both backends start from
identical numbers. Run test_parity.py first: it checks this file against the NumPy reference.

Maps one-to-one onto eco.train_models:
  - same loss: mean cross-entropy on labeled rows + lam * mean_i( w_i * KL(teacher_i || student_i) ) on U
  - same AdamW (betas 0.9/0.98, eps 1e-8, decoupled decay on W1/W2 only)
  - same conditions: inter in {None, 'self', 'dml', 'agree', 'challenge', 'oracle'}
"""
import math
import numpy as np
import torch
import torch.nn.functional as F

import eco
from eco import P, MLP as NPMLP


class TMLP(torch.nn.Module):
    """one-hot(2P) -> Linear(H) -> act -> Linear(P), initialised from the NumPy MLP for exact parity."""

    def __init__(self, H, seed, act="square", init_scale=0.5, p=P, device="cpu"):
        super().__init__()
        ref = NPMLP(H, seed, act=act, init_scale=init_scale, p=p)
        self.act = act
        T = lambda a: torch.nn.Parameter(torch.from_numpy(a.copy()).to(device))
        self.W1, self.b1, self.W2, self.b2 = T(ref.W1), T(ref.b1), T(ref.W2), T(ref.b2)

    def forward(self, X):
        z = X @ self.W1 + self.b1
        h = z * z if self.act == "square" else torch.relu(z)
        return h @ self.W2 + self.b2, h


def to_t(x, dtype=None, device="cpu"):
    t = torch.from_numpy(np.ascontiguousarray(x))
    return t.to(device=device, dtype=dtype) if dtype else t.to(device)


def js_div(p, q):
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (torch.log(a + 1e-12) - torch.log(b + 1e-12))).sum(1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def make_opt(m, cfg):
    return torch.optim.AdamW(
        [{"params": [m.W1, m.W2], "weight_decay": cfg["wd"]},
         {"params": [m.b1, m.b2], "weight_decay": 0.0}],
        lr=cfg["lr"], betas=(0.9, 0.98), eps=1e-8)


def train_models(models, labeled, XU, cfg, inter=None, yU=None, eval_fn=None):
    """labeled: list of (X float32 tensor, y long tensor) per model. XU: float32 tensor or None."""
    opts = [make_opt(m, cfg) for m in models]
    nU = 0 if XU is None else len(XU)
    snap, history = None, []
    for t in range(cfg["steps"]):
        outs = []
        for m, (Xl, yl) in zip(models, labeled):
            Xin = Xl if inter is None else torch.cat([Xl, XU])
            outs.append(m(Xin)[0])
        active = inter is not None and t >= cfg["warm"]
        if active:
            lam = cfg["lam"] * min(1.0, (t - cfg["warm"] + 1) / cfg["ramp"])
            pU = [F.softmax(o[len(l[1]):].detach(), dim=1) for o, l in zip(outs, labeled)]
            if inter == "oracle":
                teachers = [F.one_hot(yU, P).float()] * len(models)
                w = torch.ones(nU, device=XU.device)
            elif inter == "self":
                if snap is None or (t - cfg["warm"]) % cfg["self_refresh"] == 0:
                    snap = [p.clone() for p in pU]
                teachers, w = snap, torch.ones(nU, device=XU.device)
            else:
                teachers = [pU[1], pU[0]]
                js = js_div(pU[0], pU[1])
                if inter == "dml":
                    w = torch.ones(nU, device=XU.device)
                elif inter == "challenge":
                    w = js / (js.mean() + 1e-8)
                elif inter == "agree":
                    s = 1.0 - js / math.log(2)
                    w = s / (s.mean() + 1e-8)
                else:
                    raise ValueError(inter)
                w = torch.clamp(w, 0, cfg["wclip"])
        for i, (m, (Xl, yl)) in enumerate(zip(models, labeled)):
            nl = len(yl)
            loss = F.cross_entropy(outs[i][:nl], yl)  # mean over labeled rows
            if active:
                logp = F.log_softmax(outs[i][nl:], dim=1)
                kl = F.kl_div(logp, teachers[i], reduction="none").sum(1)  # KL(teacher || student) per row
                loss = loss + lam * (w * kl).mean()
            opts[i].zero_grad(set_to_none=True)
            loss.backward()
            opts[i].step()
        if eval_fn is not None and (t % cfg["log_every"] == 0 or t == cfg["steps"] - 1):
            with torch.no_grad():
                history.append((t, eval_fn(models)))
    return history


@torch.no_grad()
def acc(model, X, y):
    return float((model(X)[0].argmax(1) == y).float().mean())
