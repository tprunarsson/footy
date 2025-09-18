"""
qfooty self-play Q-learning grid football (5 actions, occupancy-aware).

- Players have 5 actions: N, E, S, W, Stay.
- A move into the **opponent's current cell is illegal**.
- Greedy policy and TD bootstrap **mask illegal moves** (walls + opponent cell).
- Safer round init: players don't start on the same cell and the ball avoids both players and goals.

Notes:
- Ball can move in the 4 cardinal directions or rest. A player "Stay" never pushes the ball.
- Saved Q slabs are now shape: **[n, m, n, m, 5]** per player.

CLI demo (small run):
    python footy_trainer.py --episodes 100000 --max-moves 500 --epsilon 0.1

"""
from __future__ import annotations
from dataclasses import dataclass
import argparse
import random
from typing import Tuple, Optional, List

import numpy as np

DEFAULT_REWARD = -1
GOAL_REWARD = 100

# 5 player actions: N,E,S,W,Stay
ACTIONS: List[Tuple[int,int]] = [
    (-1, 0),  # N (i-1)
    (0, 1),   # E (j+1)
    (1, 0),   # S (i+1)
    (0, -1),  # W (j-1)
    (0, 0),   # Stay
]
NA_PLAY = len(ACTIONS)

# Ball actions: N,E,S,W,Rest (no Stay separate from Rest)
BALL_ACTIONS: List[Tuple[int,int]] = [
    (-1, 0), (0, 1), (1, 0), (0, -1), (0, 0)
]
NA_BALL = len(BALL_ACTIONS)

@dataclass
class QParams:
    alpha: float = 0.2
    gamma: float = 0.95
    epsilon: float = 0.1
    optimistic_init: float = 10.0
    illegal_switches_turn: bool = False  # mimic the C++ quirk if True
    illegal_penalty: float = 0.0         # tiny shaping for illegal attempts

class QFootyEnv:
    def __init__(self, n: int = 13, m: int = 9, np_players: int = 2, params: Optional[QParams] = None, rng_seed: Optional[int] = None):
        assert np_players == 2, "Assumes two players."
        self.n = n
        self.m = m
        self.np = np_players
        self.na = NA_PLAY
        self.na_ball = NA_BALL
        self.params = params or QParams()
        if rng_seed is not None:
            random.seed(rng_seed)
            np.random.seed(rng_seed)

        # Pitch grid: interior True, border False
        self.pitch = np.zeros((n, m), dtype=bool)
        self._init_pitch()

        # Reward map depends only on next ball location; separate map per player
        self.reward = np.full((self.np, n, m), DEFAULT_REWARD, dtype=np.float32)
        self._init_reward()

        # Q: [player, player_i, player_j, ball_i, ball_j, action]  (action=5)
        self.Q = np.full((self.np, n, m, n, m, self.na), self.params.optimistic_init, dtype=np.float32)

        # State trackers
        self.ip = np.zeros(self.np, dtype=np.int32)
        self.jp = np.zeros(self.np, dtype=np.int32)
        self.ib: int = -1
        self.jb: int = -1
        self.iip: int = 0  # index of player to move

        # For TD updates per player
        self.S = np.zeros((self.np, 4), dtype=np.int32)  # [ip, jp, ib, jb]
        self.A = np.full(self.np, -1, dtype=np.int32)

        # Book-keeping
        self.goal = [0, 0]

    # ------------------------ Initialization helpers ------------------------ #
    def _init_pitch(self):
        self.pitch[1:self.n-1, 1:self.m-1] = True
        self.pitch[:, 0] = False
        self.pitch[:, self.m-1] = False
        self.pitch[0, :] = False
        self.pitch[self.n-1, :] = False

    def _init_reward(self):
        r = self.reward
        top_row = 1
        bot_row = self.n - 2
        mid_left = self.m // 2 - 1
        mid_right = self.m // 2
        # Player 0 scores at bottom, defends top
        r[0, bot_row, mid_left] = r[0, bot_row, mid_right] = GOAL_REWARD
        r[0, top_row, mid_left] = r[0, top_row, mid_right] = -GOAL_REWARD
        # Player 1 scores at top, defends bottom
        r[1, top_row, mid_left] = r[1, top_row, mid_right] = GOAL_REWARD
        r[1, bot_row, mid_left] = r[1, bot_row, mid_right] = -GOAL_REWARD

    # ------------------------ Episode management --------------------------- #
    def reset_q(self, value: Optional[float] = None):
        val = self.params.optimistic_init if value is None else float(value)
        self.Q.fill(val)

    def _rand_cell(self) -> Tuple[int,int]:
        return random.randint(1, self.n - 2), random.randint(1, self.m - 2)

    def prepare_round(self):
        # Place players on distinct interior cells
        self.ip[0], self.jp[0] = self._rand_cell()
        while True:
            self.ip[1], self.jp[1] = self._rand_cell()
            if (self.ip[1], self.jp[1]) != (self.ip[0], self.jp[0]):
                break
        # Place ball away from goal cells and away from players
        while True:
            self.ib, self.jb = self._rand_cell()
            if self.reward[0, self.ib, self.jb] == DEFAULT_REWARD \
               and (self.ib, self.jb) != (self.ip[0], self.jp[0]) \
               and (self.ib, self.jb) != (self.ip[1], self.jp[1]):
                break
        # Randomize first mover
        self.iip = random.randint(0, self.np - 1)
        # Reset previous S/A
        for p in range(self.np):
            self.S[p] = (self.ip[p], self.jp[p], self.ib, self.jb)
            self.A[p] = -1

    # ------------------------ Core dynamics -------------------------------- #
    def _is_goal_next_ball(self, ib_next: int, jb_next: int) -> bool:
        top_row = 1
        bot_row = self.n - 2
        mid_left = self.m // 2 - 1
        mid_right = self.m // 2
        return (jb_next in (mid_left, mid_right)) and (ib_next in (top_row, bot_row))

    def _is_legal_player_move(self, ip: int, jp: int, io: int, jo: int, a: int) -> bool:
        di, dj = ACTIONS[a]
        # Stay always legal
        if di == 0 and dj == 0:
            return True
        ii, jj = ip + di, jp + dj
        # Must be inside pitch and not a wall
        if not self.pitch[ii, jj]:
            return False
        # Cannot move into opponent's cell
        if ii == io and jj == jo:
            return False
        return True

    def _choose_action(self, p: int) -> int:
        # epsilon-greedy among legal moves (walls + opponent)
        opp = (p + 1) % self.np
        if random.random() < self.params.epsilon:
            # explore uniformly among legal
            legal = [a for a in range(self.na) if self._is_legal_player_move(self.ip[p], self.jp[p], self.ip[opp], self.jp[opp], a)]
            return random.choice(legal) if legal else self.na - 1  # Stay if somehow boxed
        # greedy among legal (mask illegal to -inf)
        qvals = self.Q[p, self.ip[p], self.jp[p], self.ib, self.jb]
        legal_mask = np.array([self._is_legal_player_move(self.ip[p], self.jp[p], self.ip[opp], self.jp[opp], a) for a in range(self.na)])
        masked = np.where(legal_mask, qvals, -np.inf)
        if not np.isfinite(masked).any():
            return self.na - 1  # Stay as fallback
        return int(np.argmax(masked))

    def _next_ball_action(self, p: int, a: int) -> int:
        """Return index into BALL_ACTIONS for how the ball should move."""
        # If player stayed, they can't step onto the ball
        di_p, dj_p = ACTIONS[a]
        if di_p == 0 and dj_p == 0:
            return NA_BALL - 1  # Rest
        # For the ball to move, the player must enter the ball cell
        if self.ib == (self.ip[p] + di_p) and self.jb == (self.jp[p] + dj_p):
            idx = a  # same direction N/E/S/W
            iopp = (p + 1) % self.np
            # If blocked, sample a random legal direction (bounce); if all blocked, rest
            attempts = 0
            while True:
                di_b, dj_b = BALL_ACTIONS[idx]
                ib2, jb2 = self.ib + di_b, self.jb + dj_b
                blocked = (not self.pitch[ib2, jb2]) or (self.ip[iopp] == ib2 and self.jp[iopp] == jb2)
                if not blocked:
                    return idx
                # try another cardinal direction
                idx = random.randrange(4)  # 0..3 (no rest during bounce unless forced)
                attempts += 1
                if attempts > 50:
                    return NA_BALL - 1  # Rest
        return NA_BALL - 1  # Rest

    def step_round(self) -> bool:
        """One (attempted) move. Returns True if terminal (goal) occurred, else False.
        Turn advances only after a **valid** move unless illegal_switches_turn=True.
        """
        assert self.ib != -1 and self.jb != -1, "Call prepare_round() first."
        p = self.iip
        opp = (p + 1) % self.np

        # choose action
        a = self._choose_action(p)

        # legality check (should pass except exploration edge cases)
        if not self._is_legal_player_move(self.ip[p], self.jp[p], self.ip[opp], self.jp[opp], a):
            if self.params.illegal_penalty != 0 and self.A[p] != -1:
                s_ip, s_jp, s_ib, s_jb = self.S[p]
                q_sa = self.Q[p, s_ip, s_jp, s_ib, s_jb, self.A[p]]
                self.Q[p, s_ip, s_jp, s_ib, s_jb, self.A[p]] = q_sa + self.params.alpha * (self.params.illegal_penalty - q_sa)
            if self.params.illegal_switches_turn:
                self.iip = opp
            return False

        # proposed displacements
        di_p, dj_p = ACTIONS[a]
        aball = self._next_ball_action(p, a)
        di_b, dj_b = BALL_ACTIONS[aball]

        # Evaluate next ball location and terminal
        ib_next, jb_next = self.ib + di_b, self.jb + dj_b
        terminal = self._is_goal_next_ball(ib_next, jb_next)

        # TD update for player p (using previous S/A)
        if self.A[p] != -1:
            s_ip, s_jp, s_ib, s_jb = self.S[p]
            q_sa = self.Q[p, s_ip, s_jp, s_ib, s_jb, self.A[p]]
            r = float(self.reward[p, ib_next, jb_next])
            if terminal:
                target = r
            else:
                # bootstrap at next state (after p moves), masking illegal moves w.r.t. **same opponent position**
                ip_next = self.ip[p] + di_p
                jp_next = self.jp[p] + dj_p
                q_next = self.Q[p, ip_next, jp_next, ib_next, jb_next]
                legal_next = np.array([
                    self._is_legal_player_move(ip_next, jp_next, self.ip[opp], self.jp[opp], a2)
                    for a2 in range(self.na)
                ])
                masked_next = np.where(legal_next, q_next, -np.inf)
                max_next = float(np.max(masked_next[np.isfinite(masked_next)]) if np.isfinite(masked_next).any() else 0.0)
                target = r + self.params.gamma * max_next
            self.Q[p, s_ip, s_jp, s_ib, s_jb, self.A[p]] = q_sa + self.params.alpha * (target - q_sa)

        # Terminal: opponent also updates once (as in C++)
        if terminal:
            if self.A[opp] != -1:
                s_ip, s_jp, s_ib, s_jb = self.S[opp]
                q_sa = self.Q[opp, s_ip, s_jp, s_ib, s_jb, self.A[opp]]
                r_opp = float(self.reward[opp, ib_next, jb_next])
                self.Q[opp, s_ip, s_jp, s_ib, s_jb, self.A[opp]] = q_sa + self.params.alpha * (r_opp - q_sa)

        # Commit current state as previous for p
        self.S[p] = (self.ip[p], self.jp[p], self.ib, self.jb)
        self.A[p] = a

        # Apply transition
        self.ip[p] += di_p
        self.jp[p] += dj_p
        self.ib = ib_next
        self.jb = jb_next

        if terminal:
            bot_row = self.n - 2
            self.goal[0] += 1 if ib_next == bot_row else 0
            self.goal[1] += 1 if ib_next == 1 else 0

        # Switch turn after a VALID move
        self.iip = opp
        return terminal

    # ------------------------ Training helpers ----------------------------- #
    def train(self, episodes: int = 10_000, max_moves: int = 1_000) -> dict:
        stats = {"episodes": episodes, "terminals": 0, "avg_len": 0.0}
        tot_len = 0
        for _ in range(episodes):
            self.prepare_round()
            moves = 0
            while moves < max_moves:
                terminal = self.step_round()
                moves += 1
                if terminal:
                    stats["terminals"] += 1
                    break
            tot_len += moves
        stats["avg_len"] = tot_len / episodes if episodes > 0 else 0.0
        return stats

    # ------------------------ Save / Load ---------------------------------- #
    def save_player(self, path: str, p: int):
        """Save only the contiguous slab Q[p, ...] to a .npy file.
        Shape: (n, m, n, m, 5)
        """
        assert 0 <= p < self.np
        np.save(path, self.Q[p])

    def load_player(self, path: str, p: int):
        assert 0 <= p < self.np
        data = np.load(path)
        assert data.shape == self.Q[p].shape, f"Expected {self.Q[p].shape}, got {data.shape}"
        self.Q[p] = data.astype(np.float32, copy=False)

# ------------------------ CLI ------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=100000)
    ap.add_argument('--max-moves', type=int, default=500)
    ap.add_argument('--epsilon', type=float, default=0.1)
    ap.add_argument('--alpha', type=float, default=0.2)
    ap.add_argument('--gamma', type=float, default=0.95)
    ap.add_argument('--optimistic', type=float, default=10.0)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    params = QParams(alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon,
                     optimistic_init=args.optimistic, illegal_switches_turn=False,
                     illegal_penalty=0.0)
    env = QFootyEnv(n=13, m=10, np_players=2, params=params, rng_seed=args.seed)

    print("Training... (demo run)")
    stats = env.train(episodes=args.episodes, max_moves=args.max_moves)
    print("Stats:", stats)

    env.save_player("q_player0.npy", 0)
    env.save_player("q_player1.npy", 1)
    print("Saved q_player0.npy and q_player1.npy")
