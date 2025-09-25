"""
single_block_policy.py

Read-only policy for a trained Q player (inference-only).

Supports TWO save formats:
  1) Old: Q shape = (n, m, n, m, na)
  2) New: Q shape = (n, m, n, m, 2, na)   # extra BLOCK axis (0/1)

Actions:
  - 4 actions: N,E,S,W
  - 5 actions: N,E,S,W,Stay (0,0)

Greedy selection is among LEGAL moves only:
  - inside pitch, not a wall
  - cannot step onto opponent's cell
  - Stay (if present) is always legal

BLOCK feature (for new format):
  BLOCK=1 if opponent is 'between' player and their target goal, locally:
    forward toward goal (row-wise), close (Manhattan<=radius), roughly aligned (|Δcol|<=align)
  BLOCK=0 otherwise.

Usage:
    from single_footy_policy import SingleFooty
    pol = SingleFooty(n=13, m=10, epsilon=0.0, debug=True, stay_debit=0.25)
    pol.load("q_player0.npy")  # works for old or new shape
    a = pol.action(ip=5, jp=4, io=7, jo=4, ib=6, jb=4)
"""
from __future__ import annotations
import numpy as np
import random
from typing import Tuple, List

# Base action sets
ACTIONS4: List[Tuple[int,int]] = [
    (-1, 0),  # N
    (0,  1),  # E
    (1,  0),  # S
    (0, -1),  # W
]
ACTIONS5: List[Tuple[int,int]] = ACTIONS4 + [(0, 0)]  # Stay
ACTION_NAMES4 = ['N','E','S','W']
ACTION_NAMES5 = ['N','E','S','W','Stay']

class SingleBlock:
    def __init__(
        self,
        n: int = 13,
        m: int = 10,
        epsilon: float = 0.0,
        rng_seed: int | None = None,
        debug: bool = False,
        stay_debit: float = 0.0,
        block_radius: int = 3,
        block_align: int = 1,
        player_id: int | None = None,
    ):
        """
        player_id:
          - Optional; used only for debug labels. Policy logic is symmetric and
            does not require knowing "who" we are.
        """
        self.n = int(n)
        self.m = int(m)
        self.na = 4            # will be set on load
        self.has_block = False # set on load if Q has BLOCK axis
        self.epsilon = float(epsilon)
        self.debug = bool(debug)
        self.stay_debit = float(stay_debit)
        self.block_radius = int(block_radius)
        self.block_align = int(block_align)
        self.player_id = player_id

        if rng_seed is not None:
            random.seed(rng_seed)
            np.random.seed(rng_seed)

        self.Q: np.ndarray | None = None  # (n,m,n,m,na) or (n,m,n,m,2,na)
        self.pitch = np.zeros((self.n, self.m), dtype=bool)
        self._init_pitch()

    # ---------- geometry / pitch ----------

    def _init_pitch(self):
        # interior True, borders False
        self.pitch[1:self.n-1, 1:self.m-1] = True
        self.pitch[:, 0] = False
        self.pitch[:, self.m-1] = False
        self.pitch[0, :] = False
        self.pitch[self.n-1, :] = False

    def _actions(self):
        if self.na == 5:
            return ACTIONS5, ACTION_NAMES5
        return ACTIONS4, ACTION_NAMES4

    # ---------- BLOCK feature (matches trainer heuristic) ----------

    @staticmethod
    def _dir_row_for_player(p: int) -> int:
        # P0 attacks bottom => +1; P1 attacks top => -1
        return 1 if p == 0 else -1

    def _compute_block(self, p: int, ip: int, jp: int, ib: int, jb: int, io: int, jo: int) -> int:
        """
        Returns 1 if opponent is 'between' player and the goal they attack,
        within a small local window; else 0.
        forward (toward goal), close (Manhattan<=radius), aligned (|Δcol|<=align).
        """
        block = 1 if (abs(ib - io) + abs(jb - jo) <= 2) else 0
        return block
        dir_row = self._dir_row_for_player(p)
        forward = (dir_row * (io - ip)) > 0
        if not forward:
            return 0
        manh = abs(io - ip) + abs(jo - jp)
        if manh > self.block_radius:
            return 0
        if abs(jo - jp) > self.block_align:
            return 0
        return 1

    # ---------- IO ----------

    def load(self, path: str) -> bool:
        arr = np.load(path)
        if arr.ndim == 5:
            # Old: (n,m,n,m,na)
            if (arr.shape[0] != self.n or arr.shape[1] != self.m or
                arr.shape[2] != self.n or arr.shape[3] != self.m):
                raise ValueError(f"Unexpected Q shape {arr.shape}; expected ({self.n},{self.m},{self.n},{self.m},na)")
            if arr.shape[4] not in (4, 5):
                raise ValueError(f"Unsupported action dimension {arr.shape[4]} (expected 4 or 5)")
            self.has_block = False
            self.na = int(arr.shape[4])
            self.Q = arr.astype(np.float32, copy=False)
            return True

        elif arr.ndim == 6:
            # New: (n,m,n,m,2,na)  (BLOCK axis present)
            if (arr.shape[0] != self.n or arr.shape[1] != self.m or
                arr.shape[2] != self.n or arr.shape[3] != self.m or
                arr.shape[4] != 2):
                raise ValueError(f"Unexpected Q shape {arr.shape}; expected ({self.n},{self.m},{self.n},{self.m},2,na)")
            if arr.shape[5] not in (4, 5):
                raise ValueError(f"Unsupported action dimension {arr.shape[5]} (expected 4 or 5)")
            self.has_block = True
            self.na = int(arr.shape[5])
            self.Q = arr.astype(np.float32, copy=False)
            return True

        else:
            raise ValueError(f"Unsupported Q ndim={arr.ndim}. Expected 5 or 6.")

    # ---------- Legality ----------

    def _is_legal(self, ip: int, jp: int, io: int, jo: int, a: int) -> bool:
        actions, _ = self._actions()
        if not (0 <= a < self.na):
            return False
        di, dj = actions[a]
        # Stay (0,0) always legal if available
        if di == 0 and dj == 0:
            return True
        ii, jj = ip + di, jp + dj
        # Walls / interior
        if not self.pitch[ii, jj]:
            return False
        # Cannot step onto opponent
        if ii == io and jj == jo:
            return False
        return True

    # ---------- Policy ----------

    def action(self, ip: int, jp: int, io: int, jo: int, ib: int, jb: int, p: int | None = None) -> int:
        """
        Return an action index in [0, na).

        Parameters:
            ip, jp : our player's position
            io, jo : opponent position
            ib, jb : ball position (indices into Q)
            p      : which player are WE? only needed for BLOCK (if Q has block axis).
                     If None, we infer from self.player_id or assume player 0 for BLOCK.
        """
        if self.Q is None:
            raise RuntimeError("Call load(path) before action().")

        # choose identity for BLOCK computation
        if p is None:
            p = 0 if self.player_id is None else int(self.player_id)

        actions, names = self._actions()

        # Build the relevant Q-slice
        if self.has_block:
            block = self._compute_block(p, ip, jp, io, jo, ib, jb)
            qslice = self.Q[ip, jp, ib, jb, block]  # shape [na]
        else:
            qslice = self.Q[ip, jp, ib, jb]         # shape [na]

        # Build legal mask (depends on both positions)
        legal_mask = np.array([self._is_legal(ip, jp, io, jo, k) for k in range(self.na)], dtype=bool)

        if self.debug:
            who = f"P{p}" if self.player_id is None else f"P{self.player_id}"
            print(f"{who} state ip={ip} jp={jp} | io={io} jo={jo} | ib={ib} jb={jb}"
                  + (f" | block={block}" if self.has_block else ""))
            print("Q raw:      " + ", ".join(f"{names[k]}:{qslice[k]:.2f}" for k in range(self.na)))
            print("legal:      " + (", ".join(names[k] for k in range(self.na) if legal_mask[k]) or "<none>"))

        # epsilon-greedy over LEGAL actions
        if random.random() < self.epsilon:
            legal_idxs = np.flatnonzero(legal_mask)
            if legal_idxs.size == 0:
                if self.debug: print("no legal actions; returning -1 (no-op)")
                return -1
            a = int(random.choice(legal_idxs))
            if self.debug:
                print(f"selected (explore legal): {names[a]}")
            return a

        # Greedy: mask illegal to -inf, optionally penalize Stay
        masked = qslice.astype(np.float64, copy=True)  # avoid -inf float32 issues in prints
        masked[~legal_mask] = -np.inf
        if self.na == 5 and self.stay_debit != 0.0:
            masked[4] = masked[4] - self.stay_debit

        a = int(np.argmax(masked))

        if self.debug:
            eff_str = ", ".join(
                f"{names[k]}:{('-inf' if not np.isfinite(masked[k]) else f'{masked[k]:.2f}')}"
                for k in range(self.na)
            )
            if self.na == 5 and self.stay_debit != 0.0:
                print(f"stay_debit applied: -{self.stay_debit}")
            print("Q effective: " + eff_str)
            print(f"selected (greedy legal): {names[a]}")

        # Final guard
        if self._is_legal(ip, jp, io, jo, a):
            return a

        # Shouldn’t really happen; fallback to any legal, else -1
        legal_idxs = np.flatnonzero(legal_mask)
        if legal_idxs.size:
            a2 = int(random.choice(legal_idxs))
            if self.debug:
                print(f"fallback selected: {names[a2]}")
            return a2
        if self.debug:
            print("fallback: no legal moves, returning -1")
        return -1

    def notify_goal(self, goal: int):
        # No-op for API parity
        return

# Small CLI smoke test
if __name__ == "__main__":
    # Example: loads either (n,m,n,m,na) or (n,m,n,m,2,na)
    pol = SingleBlockPolicy(n=13, m=10, epsilon=0.0, rng_seed=1, debug=True, stay_debit=0.25,
                      block_radius=3, block_align=1, player_id=0)
    try:
        pol.load("q_player0.npy")
    except Exception as e:
        print("Load failed:", e)
        raise

    # Example state (ensure interior)
    ip, jp = 6, 4
    io, jo = 7, 4
    ib, jb = 6, 5
    print("Chosen action:", pol.action(ip, jp, io, jo, ib, jb, p=0))
