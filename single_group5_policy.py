# single_footy_policy.py
# Read-only policy that supports:
#  1) Dense slab [.n,.m,.n,.m,na]  (teacher format; na in {4,5})
#  2) Encoded dict (.npy, "encoded_dict_v1") from trainer.py
#

from __future__ import annotations
import numpy as np
import random
from typing import Dict, Tuple, List

# ---------------- Encoder (matches trainer) ---------------- #

class _EncoderCfg:
    def __init__(self, bin_mode: str = "clip5", include_flags: bool = True):
        self.bin_mode = bin_mode
        self.include_flags = include_flags

class _StateEncoder:
    def __init__(self, n: int, m: int, cfg: _EncoderCfg):
        self.n, self.m, self.cfg = n, m, cfg

    @staticmethod
    def _clip5(d: int) -> int:
        return int(max(-2, min(2, d)))

    @staticmethod
    def _pos(ip, jp, io, jo, ib, jb) -> int:
        # 0: free, 1: self has ball, 2: opp has ball
        if ip == ib and jp == jb:
            return 1
        if io == ib and jo == jb:
            return 2
        return 0

    def encode(self, ip, jp, io, jo, ib, jb) -> Tuple[int, ...]:
        dx_o = self._clip5(io - ip); dy_o = self._clip5(jo - jp)
        dx_b = self._clip5(ib - ip); dy_b = self._clip5(jb - jp)
        pos = self._pos(ip, jp, io, jo, ib, jb)
        if self.cfg.include_flags:
            # Note: these flags are the same as used during training.
            opp_between = int((jp == jb == jo) and ((ip <= ib <= io) or (io <= ib <= ip)))
            ball_below  = int(ib > ip)  # "ball below me" (orientation baked into trained Q)
            return (dx_o, dy_o, dx_b, dy_b, pos, opp_between, ball_below)
        return (dx_o, dy_o, dx_b, dy_b, pos)

# ---------------- Policy ---------------- #

class SingleGroup5:
    """
    Role-agnostic policy. The file you load (P0-trained or P1-trained) determines behavior;
    this class does no mirroring and doesn’t track p0/p1 itself.
    """
    def __init__(self, n: int = 13, m: int = 10, epsilon: float = 0.0,
                 debug: bool = False, rng_seed: int | None = None,
                 stay_debit: float = 0.0):
        self.n, self.m = int(n), int(m)
        self.epsilon = float(epsilon)
        self.debug = bool(debug)
        self.stay_debit = float(stay_debit)
        if rng_seed is not None:
            random.seed(rng_seed); np.random.seed(rng_seed)

        # interior-playable pitch (matches harness)
        self.pitch = np.zeros((self.n, self.m), dtype=bool)
        self.pitch[1:self.n-1, 1:self.m-1] = True
        self.pitch[:, 0] = self.pitch[:, self.m-1] = self.pitch[0, :] = self.pitch[self.n-1, :] = False

        # storage
        self.mode: str | None = None                 # 'dense' | 'encoded'
        self.Q_dense: np.ndarray | None = None       # [n,m,n,m,na]
        self.Q_dict: Dict[Tuple[int, ...], np.ndarray] | None = None
        self.encoder: _StateEncoder | None = None
        self.q_default: float = 0.0
        self.na: int = 5                              # will be overwritten on load

    def _actions(self) -> List[Tuple[int, int]]:
        # If na==5 include Stay; if na==4 omit it.
        base = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        return base + ([(0, 0)] if self.na == 5 else [])

    def load(self, path: str) -> None:
        """
        Load either:
          - Dense slab: a numeric ndarray with shape [n,m,n,m,na]
          - Encoded dict: a pickled dict with keys described in trainer.py
        """
        obj = np.load(path, allow_pickle=True)

        # Dense slab?
        if isinstance(obj, np.ndarray) and obj.ndim == 5 \
           and obj.shape[0] == self.n and obj.shape[1] == self.m \
           and obj.shape[2] == self.n and obj.shape[3] == self.m:
            if obj.shape[4] not in (4, 5):
                raise ValueError(f"Unsupported action dimension {obj.shape[4]}; expected 4 or 5.")
            self.na = int(obj.shape[4])
            self.Q_dense = obj.astype(np.float32, copy=False)
            self.mode = "dense"
            if self.debug:
                print(f"[load] dense slab ok: shape={self.Q_dense.shape} na={self.na}")
            return

        # Encoded dict (object array or dict)
        try:
            payload = obj.item() if hasattr(obj, "item") else obj
        except Exception:
            payload = obj  # if it's already a dict

        if not isinstance(payload, dict) or payload.get("format") != "encoded_dict_v1":
            raise ValueError("Unrecognized model file (need dense slab [n,m,n,m,na] or 'encoded_dict_v1').")

        if int(payload["n"]) != self.n or int(payload["m"]) != self.m:
            raise ValueError(f"Grid mismatch: model ({payload['n']},{payload['m']}) vs requested ({self.n},{self.m}).")

        self.na = int(payload.get("na", 5))
        keys = payload.get("Q_keys", [])
        vals = payload.get("Q_values", np.zeros((0, self.na), dtype=np.float32))
        self.Q_dict = {tuple(k): np.asarray(v, dtype=np.float32) for k, v in zip(keys, vals)}
        enc_cfg = payload.get("encoder", {"bin_mode": "clip5", "include_flags": True})
        self.encoder = _StateEncoder(self.n, self.m, _EncoderCfg(enc_cfg.get("bin_mode", "clip5"),
                                                                 bool(enc_cfg.get("include_flags", True))))
        self.q_default = float(payload.get("optimistic_init", 0.0))
        self.mode = "encoded"
        if self.debug:
            print(f"[load] encoded_dict_v1 ok: states={len(self.Q_dict)} na={self.na}")

    # ---------------- Internals ---------------- #

    def _legal_mask(self, ip: int, jp: int, io: int, jo: int) -> np.ndarray:
        actions = self._actions()
        mask = np.zeros(self.na, dtype=bool)
        for a, (di, dj) in enumerate(actions):
            ii, jj = ip + di, jp + dj
            # legal if inside pitch and not stepping onto opponent
            mask[a] = self.pitch[ii, jj] and not (ii == io and jj == jo)
        return mask

    def _q_for_state(self, ip: int, jp: int, io: int, jo: int, ib: int, jb: int) -> np.ndarray:
        if self.mode == "dense":
            # Q_dense is indexed as [ip, jp, ib, jb, a]
            return self.Q_dense[ip, jp, ib, jb].astype(np.float32, copy=False)
        # encoded dict
        s = self.encoder.encode(ip, jp, io, jo, ib, jb)
        return self.Q_dict.get(s, np.full(self.na, self.q_default, dtype=np.float32))

    @staticmethod
    def _greedy_tiebreak_south(q_eff: np.ndarray) -> int:
        # Prefer South (2) if tied, else random among ties
        mx = np.max(q_eff)
        ties = np.flatnonzero(np.isclose(q_eff, mx))
        return 2 if 2 in ties else int(random.choice(ties)) if ties.size else -1

    # ---------------- Public API ---------------- #

    def action(self, ip: int, jp: int, io: int, jo: int, ib: int, jb: int) -> int:
        """
        Pick an action index in {0..na-1}. Returns -1 only if literally no legal moves.
        Role-agnostic: pass the calling player's (ip,jp) first.
        """
        if self.mode not in ("dense", "encoded"):
            raise RuntimeError("Model not loaded. Call .load(path) first.")

        legal = self._legal_mask(ip, jp, io, jo)

        # epsilon-greedy among legal actions
        if self.epsilon > 0.0 and random.random() < self.epsilon:
            idxs = np.flatnonzero(legal)
            if idxs.size == 0:
                return -1
            a = int(random.choice(idxs))
            if self.debug:
                print(f"[explore] chose a={a}")
            return a

        # greedy (mask illegals; optional small penalty for Stay)
        q = self._q_for_state(ip, jp, io, jo, ib, jb).copy()
        q_eff = q.copy()
        q_eff[~legal] = -np.inf
        if self.na == 5 and self.stay_debit != 0.0:
            q_eff[4] = q_eff[4] - self.stay_debit

        a = self._greedy_tiebreak_south(q_eff)

        if self.debug:
            names = ['N', 'E', 'S', 'W'] + (['Stay'] if self.na == 5 else [])
            lm = ", ".join(names[k] for k in range(self.na) if legal[k]) or "<none>"
            qstr = ", ".join(f"{names[k]}:{q[k]:.2f}" for k in range(self.na))
            estr = ", ".join(f"{names[k]}:{('-inf' if not np.isfinite(q_eff[k]) else f'{q_eff[k]:.2f}')}" for k in range(self.na))
            print(f"state ip={ip} jp={jp} io={io} jo={jo} ib={ib} jb={jb}")
            print(f"Q raw: {qstr}")
            if self.na == 5 and self.stay_debit != 0.0:
                print(f"stay_debit applied: -{self.stay_debit}")
            print(f"legal: {lm}")
            print(f"Q eff: {estr}")
            print(f"greedy -> {names[a] if a >= 0 else '<none>'}")

        return a

    # Optional hook to satisfy tournament APIs that call it
    def notify_goal(self, goal: int) -> None:
        return

# ---------------- CLI Smoke Test ---------------- #

if __name__ == "__main__":
    # Quick sanity check (won't run a game; just loads and queries one state)
    import sys
    if len(sys.argv) < 2:
        print("Usage: python single_footy_policy.py <model.npy> [epsilon]")
        sys.exit(0)
    path = sys.argv[1]
    eps = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    pol = SingleFooty(n=13, m=10, epsilon=eps, debug=True)
    pol.load(path)
    print("Loaded mode:", pol.mode, "na:", pol.na)
    # Dummy query (numbers must be within grid bounds)
    print("Chosen action:",
          pol.action(ip=6, jp=4, io=7, jo=4, ib=6, jb=5))