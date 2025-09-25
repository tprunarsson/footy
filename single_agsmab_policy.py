"""
Read-only policy for a trained Q player (inference-only), mirroring the C++ singleFooty.

Supports two save formats:
- OLD tabular:  Q shape [n, m, n, m, na]                 (no extra state)
- NEW tabular:  Q shape [n, m, n, m, 12, na]             (12-bin extra state relative to BALL)

Actions:
    • 4 actions: N,E,S,W
    • 5 actions: N,E,S,W,Stay (Stay = do nothing)

Selection:
- epsilon-greedy on Q[ ip, jp, ib, jb, (bin12,) : ]
- legality enforced (walls/out-of-bounds blocked); greedy ignores illegal moves
- optional stay_debit subtracts from Stay's Q for tie-breaking / biasing

Usage:
    from single_footy_policy import SingleFooty
    pol = SingleFooty(n=13, m=10, epsilon=0.1, debug=True, stay_debit=0.25)
    pol.load("q_player0.npy")
    a = pol.action(ip=5, jp=4, io=7, jo=4, ib=6, jb=4)
"""
from __future__ import annotations
import numpy as np
import random
from typing import Tuple, List

# Base action sets (we will slice to self.na at runtime)
ACTIONS4: List[Tuple[int, int]] = [
    (-1, 0),  # N
    (0, 1),   # E
    (1, 0),   # S
    (0, -1),  # W
]
ACTIONS5: List[Tuple[int, int]] = ACTIONS4 + [(0, 0)]  # Stay
ACTION_NAMES4 = ['N', 'E', 'S', 'W']
ACTION_NAMES5 = ['N', 'E', 'S', 'W', 'Stay']

# ---- 12-bin opponent zone relative to the BALL (Python version) ----
# bands: near (d<=1), mid (d<=3), far (>3). tweak if you like.
D1_NEAR = 1  # light-blue / yellow (near)
D2_MID  = 3  # purple / orange (mid)

def zone12_bin(ib: int, jb: int, io: int, jo: int) -> int:
    """
    Map opponent (io,jo) relative to ball (ib,jb) into 12 bins.

    Vertical (above/below): near, mid, far  -> bins 0..5
        0 = TOP_NEAR, 1 = BOT_NEAR, 2 = TOP_MID, 3 = BOT_MID, 4 = TOP_FAR, 5 = BOT_FAR
    Horizontal (left/right): near, mid, far -> bins 6..11
        6 = LEFT_NEAR, 7 = RIGHT_NEAR, 8 = LEFT_MID, 9 = RIGHT_MID, 10 = LEFT_FAR, 11 = RIGHT_FAR
    """
    di = io - ib   # +down, -up
    dj = jo - jb   # +right, -left

    if di == 0 and dj == 0:
        # opponent cannot stand on the ball; fall back to a consistent bin
        return 1  # BOT_NEAR

    adi = di if di >= 0 else -di
    adj = dj if dj >= 0 else -dj

    # decide vertical vs horizontal family (disjoint)
    if adi >= adj:
        # vertical family
        if adi <= D1_NEAR:
            band = 0  # near
        elif adi <= D2_MID:
            band = 1  # mid
        else:
            band = 2  # far
        top = (di < 0)
        if band == 0:
            return 0 if top else 1
        elif band == 1:
            return 2 if top else 3
        else:
            return 4 if top else 5
    else:
        # horizontal family
        if adj <= D1_NEAR:
            band = 0  # near
        elif adj <= D2_MID:
            band = 1  # mid
        else:
            band = 2  # far
        left = (dj < 0)
        if band == 0:
            return 6 if left else 7
        elif band == 1:
            return 8 if left else 9
        else:
            return 10 if left else 11


class SingleAgsmab:
    def __init__(self, n: int = 13, m: int = 9, epsilon: float = 0.1, rng_seed: int | None = None,
                 debug: bool = False, stay_debit: float = 0.0):
        self.n = n
        self.m = m
        self.na = 4  # will be set on load()
        self.epsilon = float(epsilon)
        self.debug = bool(debug)
        self.stay_debit = float(stay_debit)  # subtract from Stay's Q for selection/display (if Stay exists)
        if rng_seed is not None:
            random.seed(rng_seed)
            np.random.seed(rng_seed)
        self.Q: np.ndarray | None = None  # shape [n,m,n,m,(12,),na]
        self._has_bins12 = False
        self.pitch = np.zeros((n, m), dtype=bool)
        self._init_pitch()

    def _actions(self):
        if self.na == 5:
            return ACTIONS5, ACTION_NAMES5
        return ACTIONS4, ACTION_NAMES4

    def _init_pitch(self):
        self.pitch[1:self.n-1, 1:self.m-1] = True
        self.pitch[:, 0] = False
        self.pitch[:, self.m-1] = False
        self.pitch[0, :] = False
        self.pitch[self.n-1, :] = False

    def load(self, path: str) -> bool:
        arr = np.load(path)
        if arr.ndim == 5:
            # OLD: [n, m, n, m, na]
            if arr.shape[0] != self.n or arr.shape[1] != self.m or arr.shape[2] != self.n or arr.shape[3] != self.m:
                raise ValueError(f"Unexpected Q shape {arr.shape}; expected ({self.n},{self.m},{self.n},{self.m},na)")
            if arr.shape[4] not in (4, 5):
                raise ValueError(f"Unsupported action dimension {arr.shape[4]} (expected 4 or 5)")
            self._has_bins12 = False
            self.na = int(arr.shape[4])
        elif arr.ndim == 6:
            # NEW: [n, m, n, m, 12, na]
            if arr.shape[0] != self.n or arr.shape[1] != self.m or arr.shape[2] != self.n or arr.shape[3] != self.m:
                raise ValueError(f"Unexpected Q shape {arr.shape}; expected ({self.n},{self.m},{self.n},{self.m},12,na)")
            if arr.shape[4] != 12:
                raise ValueError(f"Unsupported bin dimension {arr.shape[4]} (expected 12)")
            if arr.shape[5] not in (4, 5):
                raise ValueError(f"Unsupported action dimension {arr.shape[5]} (expected 4 or 5)")
            self._has_bins12 = True
            self.na = int(arr.shape[5])
        else:
            raise ValueError(f"Unexpected Q ndim={arr.ndim}; expected 5 or 6")
        self.Q = arr.astype(np.float32, copy=False)
        return True

    def _is_legal(self, ip: int, jp: int, a: int) -> bool:
        actions, _ = self._actions()
        if not (0 <= a < self.na):
            return False
        di, dj = actions[a]
        # Stay is always legal (if available)
        if di == 0 and dj == 0:
            return True
        return self.pitch[ip + di, jp + dj]

    def _qslice(self, ip: int, jp: int, io: int, jo: int, ib: int, jb: int) -> np.ndarray:
        """Return view of Q-values for (state, :action) handling both 5-D and 6-D slabs."""
        if self.Q is None:
            raise RuntimeError("Q not loaded")
        if self._has_bins12:
            b = zone12_bin(ib, jb, io, jo)
            return self.Q[ip, jp, ib, jb, b]
        else:
            return self.Q[ip, jp, ib, jb]

    def action(self, ip: int, jp: int, io: int, jo: int, ib: int, jb: int) -> int:
        """Return an action index in [0, na).

        Parameters (match C++ signature for convenience):
            ip, jp : our player's position
            io, jo : opponent position (used if Q has 12-bin axis)
            ib, jb : ball position (index into Q)
        """
        if self.Q is None:
            raise RuntimeError("Call load(path) before action().")

        tries = 0
        while True:
            tries += 1

            qslice = self._qslice(ip, jp, io, jo, ib, jb)  # shape (na,)
            actions, names = self._actions()

            # Build legal mask once per try (depends only on (ip,jp))
            legal_mask = np.array([self._is_legal(ip, jp, k) for k in range(self.na)], dtype=bool)

            # Debug: raw Q and legal set
            if self.debug:
                if self._has_bins12:
                    b = zone12_bin(ib, jb, io, jo)
                    print(f"state ip={ip} jp={jp} io={io} jo={jo} ib={ib} jb={jb} | bin12={b}")
                else:
                    print(f"state ip={ip} jp={jp} io={io} jo={jo} ib={ib} jb={jb}")
                print("Q:", ", ".join(f"{names[k]}:{qslice[k]:.2f}" for k in range(self.na)))
                print("legal:", ", ".join(names[k] for k in range(self.na) if legal_mask[k]) or "<none>")

            # epsilon-greedy among LEGAL actions
            if random.random() < self.epsilon:
                # explore uniformly among legal
                legal_idxs = np.flatnonzero(legal_mask)
                if legal_idxs.size == 0:
                    if self.debug:
                        print("no legal actions; returning -1 (no-op)")
                    return -1
                a = int(random.choice(legal_idxs))
                if self.debug:
                    print(f"selected (explore legal): {names[a]}")
            else:
                masked = qslice.copy()
                masked[~legal_mask] = -np.inf
                # Apply stay debit (only to the Stay slot if present)
                if self.na == 5 and self.stay_debit != 0.0:
                    masked[4] = masked[4] - self.stay_debit
                a = int(np.argmax(masked))
                if self.debug:
                    eff_str = ", ".join(
                        f"{names[k]}:{(masked[k] if np.isfinite(masked[k]) else float('-inf')):.2f}"
                        for k in range(self.na)
                    )
                    if self.na == 5 and self.stay_debit != 0.0:
                        print(f"stay_debit applied: -{self.stay_debit}")
                    print("Q (effective for greedy):", eff_str)
                    print(f"selected (greedy legal): {names[a]}")

            # Final guard (should always be legal)
            if a >= 0 and self._is_legal(ip, jp, a):
                return a

            # Illegal or no-op sentinel: try again or emergency fallback
            if self.debug:
                print(f"warning: illegal/no-op selection {a}; retry")
            if tries > 16:
                legal = np.flatnonzero(legal_mask)
                if legal.size:
                    a2 = int(random.choice(legal))
                    if self.debug:
                        print(f"fallback selected: {names[a2]}")
                    return a2
                if self.debug:
                    print("fallback: no legal moves, returning -1")
                return -1

    def notify_goal(self, goal: int):
        # No-op: inference-only strategy
        return


# Small CLI demo
if __name__ == "__main__":
    pol = SingleFooty(n=13, m=10, epsilon=0.1, rng_seed=1, debug=True, stay_debit=0.25)
    try:
        pol.load("q_player0.npy")
    except Exception as e:
        print("Load failed:", e)
    # Example state (ensure it's interior indices)
    ip, jp, io, jo, ib, jb = 6, 4, 7, 4, 6, 5
    print("Chosen action:", pol.action(ip, jp, io, jo, ib, jb))
