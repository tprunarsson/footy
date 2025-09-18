"""
Numba-accelerated self-play Q-learning for grid footy (5 actions, occupancy-aware).

Speedups:
- Hot loops (action selection, legality checks, TD update, ball dynamics) compiled with @njit.
- No Python objects inside the JIT boundary; everything is numpy arrays or scalars.

Compatibility:
- Saves the same per-player slabs as before but with 5 actions: shape (n, m, n, m, 5), float32.
- Rewards & pitch identical to the non-Numba trainer.

Install:
    pip install numba

Run:
    python qfooty_5actions_trainer_numba.py --episodes 20000 --max-moves 500 --epsilon 0.2

"""
from __future__ import annotations
import argparse
import numpy as np
from numba import njit

DEFAULT_REWARD = -1.0
GOAL_REWARD = 100.0

# Actions (player): N,E,S,W,Stay
ACTIONS = np.array([
    [-1, 0],  # N
    [0, 1],   # E
    [1, 0],   # S
    [0, -1],  # W
    [0, 0],   # Stay
], dtype=np.int32)
NA = ACTIONS.shape[0]

# Ball actions: N,E,S,W,Rest(=Stay)
BALL_ACTIONS = ACTIONS.copy()
NA_BALL = BALL_ACTIONS.shape[0]

FNEG_INF = np.float32(-1.0e30)

@njit(cache=True)
def init_pitch(n: int, m: int) -> np.ndarray:
    pitch = np.zeros((n, m), dtype=np.uint8)
    for i in range(1, n-1):
        for j in range(1, m-1):
            pitch[i, j] = 1
    return pitch

@njit(cache=True)
def init_reward(n: int, m: int) -> np.ndarray:
    r = np.empty((2, n, m), dtype=np.float32)
    r.fill(np.float32(DEFAULT_REWARD))
    top_row = 1
    bot_row = n - 2
    ml = m // 2 - 1
    mr = m // 2
    # P0 scores bottom
    r[0, bot_row, ml] = np.float32(GOAL_REWARD)
    r[0, bot_row, mr] = np.float32(GOAL_REWARD)
    r[0, top_row, ml] = np.float32(-GOAL_REWARD)
    r[0, top_row, mr] = np.float32(-GOAL_REWARD)
    # P1 scores top
    r[1, top_row, ml] = np.float32(GOAL_REWARD)
    r[1, top_row, mr] = np.float32(GOAL_REWARD)
    r[1, bot_row, ml] = np.float32(-GOAL_REWARD)
    r[1, bot_row, mr] = np.float32(-GOAL_REWARD)
    return r

@njit(cache=True)
def is_goal_next(n: int, m: int, ib_next: int, jb_next: int) -> bool:
    top_row = 1
    bot_row = n - 2
    ml = m // 2 - 1
    mr = m // 2
    return (jb_next == ml or jb_next == mr) and (ib_next == top_row or ib_next == bot_row)

@njit(cache=True)
def is_legal_move(pitch: np.ndarray, ip: int, jp: int, io: int, jo: int, a: int) -> bool:
    di = ACTIONS[a, 0]
    dj = ACTIONS[a, 1]
    # Stay always legal
    if di == 0 and dj == 0:
        return True
    ii = ip + di
    jj = jp + dj
    # inside & not a wall
    if pitch[ii, jj] == 0:
        return False
    # cannot step onto the opponent
    if ii == io and jj == jo:
        return False
    return True

@njit(cache=True)
def argmax_masked(arr: np.ndarray, legal_mask: np.ndarray) -> int:
    best = FNEG_INF
    best_i = 0
    for k in range(arr.shape[0]):
        if legal_mask[k]:
            v = arr[k]
            if v > best:
                best = v
                best_i = k
    return best_i

@njit(cache=True)
def choose_action(qslice: np.ndarray, pitch: np.ndarray, ip: int, jp: int, io: int, jo: int, epsilon: float) -> int:
    # Build legal mask
    legal = np.zeros(NA, dtype=np.uint8)
    cnt = 0
    for a in range(NA):
        ok = is_legal_move(pitch, ip, jp, io, jo, a)
        legal[a] = 1 if ok else 0
        if ok:
            cnt += 1
    if cnt == 0:
        return 4  # Stay
    if np.random.random() < epsilon:
        # uniform among legal
        # gather indices
        idxs = np.empty(cnt, dtype=np.int32)
        c = 0
        for a in range(NA):
            if legal[a] == 1:
                idxs[c] = a
                c += 1
        pick = int(np.random.randint(0, cnt))
        return int(idxs[pick])
    # greedy among legal
    legal_b = legal.astype(np.bool_)
    return argmax_masked(qslice, legal_b)

@njit(cache=True)
def next_ball_action(pitch: np.ndarray, ip: int, jp: int, io: int, jo: int, ib: int, jb: int, a: int) -> int:
    di_p = ACTIONS[a, 0]
    dj_p = ACTIONS[a, 1]
    # Stay can't step onto the ball
    if di_p == 0 and dj_p == 0:
        return NA_BALL - 1  # Rest
    # Ball moves if we step onto it
    if ib == ip + di_p and jb == jp + dj_p:
        idx = a  # same direction N/E/S/W
        # if blocked by wall or opponent, bounce randomly among cardinals
        attempts = 0
        while True:
            di_b = BALL_ACTIONS[idx, 0]
            dj_b = BALL_ACTIONS[idx, 1]
            ib2 = ib + di_b
            jb2 = jb + dj_b
            blocked = (pitch[ib2, jb2] == 0) or (ib2 == io and jb2 == jo)
            if not blocked:
                return idx
            idx = int(np.random.randint(0, 4))  # 0..3
            attempts += 1
            if attempts > 50:
                return NA_BALL - 1
    return NA_BALL - 1

@njit(cache=True)
def step_round(Q: np.ndarray, pitch: np.ndarray, reward: np.ndarray,
               ip: np.ndarray, jp: np.ndarray,
               ib: int, jb: int, iip: int,
               alpha: float, gamma: float, epsilon: float) -> tuple:
    """One (attempted) move. Returns (terminal, ib, jb, iip, goals0_add, goals1_add).
    Q is updated in-place; ip/jp are updated in-place for the moving player.
    """
    p = iip
    opp = 1 - p

    qslice = Q[p, ip[p], jp[p], ib, jb]
    a = choose_action(qslice, pitch, ip[p], jp[p], ip[opp], jp[opp], epsilon)

    # Final legality check (should pass)
    if not is_legal_move(pitch, ip[p], jp[p], ip[opp], jp[opp], a):
        # no state change; pass turn would be a choice, but keep same player to retry
        # Here we choose to keep same iip to mirror "retry until legal" behavior
        return (False, ib, jb, iip, 0, 0)

    # Displacements
    di_p = ACTIONS[a, 0]
    dj_p = ACTIONS[a, 1]
    aball = next_ball_action(pitch, ip[p], jp[p], ip[opp], jp[opp], ib, jb, a)
    di_b = BALL_ACTIONS[aball, 0]
    dj_b = BALL_ACTIONS[aball, 1]

    ib_next = ib + di_b
    jb_next = jb + dj_b
    terminal = is_goal_next(pitch.shape[0], pitch.shape[1], ib_next, jb_next)

    # TD update using previous S/A (stored in Q by indexing current state)
    # We need the previous (S,A). For speed and simplicity, we directly do standard Q-learning
    # using current state's action a and next state's max.
    r = reward[p, ib_next, jb_next]
    # Compute next-state max for same player at (ip+di_p, jp+dj_p, ib_next, jb_next)
    ipn = ip[p] + di_p
    jpn = jp[p] + dj_p
    q_next_slice = Q[p, ipn, jpn, ib_next, jb_next]
    # mask illegal next actions vs opponent's same position
    legal_next = np.zeros(NA, dtype=np.bool_)
    for k in range(NA):
        legal_next[k] = is_legal_move(pitch, ipn, jpn, ip[opp], jp[opp], k)
    # find max among legal; if none, 0
    max_next = np.float32(0.0)
    any_legal = False
    best = FNEG_INF
    for k in range(NA):
        if legal_next[k]:
            any_legal = True
            v = q_next_slice[k]
            if v > best:
                best = v
    if any_legal:
        max_next = best
    target = r if terminal else (r + np.float32(gamma) * max_next)

    # Q update at current state/action
    qsa = Q[p, ip[p], jp[p], ib, jb, a]
    Q[p, ip[p], jp[p], ib, jb, a] = qsa + np.float32(alpha) * (target - qsa)

    # Apply transitions
    ip[p] = ipn
    jp[p] = jpn
    ib = ib_next
    jb = jb_next

    goals0 = 0
    goals1 = 0
    if terminal:
        bot_row = pitch.shape[0] - 2
        if ib == bot_row:
            goals0 = 1
        elif ib == 1:
            goals1 = 1

    # Switch turn to opponent after a valid move
    iip = opp
    return (terminal, ib, jb, iip, goals0, goals1)

@njit(cache=True)
def rand_interior(n: int, m: int) -> tuple:
    i = np.random.randint(1, n-1)
    j = np.random.randint(1, m-1)
    return i, j

@njit(cache=True)
def prepare_round_jit(pitch: np.ndarray, reward: np.ndarray) -> tuple:
    n, m = pitch.shape
    # place distinct players
    ip0, jp0 = rand_interior(n, m)
    while True:
        ip1, jp1 = rand_interior(n, m)
        if ip1 != ip0 or jp1 != jp0:
            break
    # place ball away from goals and players
    while True:
        ib, jb = rand_interior(n, m)
        if reward[0, ib, jb] == DEFAULT_REWARD and not ((ib == ip0 and jb == jp0) or (ib == ip1 and jb == jp1)):
            break
    iip = np.random.randint(0, 2)
    return ip0, jp0, ip1, jp1, ib, jb, iip

class QFootyNumba:
    def __init__(self, n: int = 13, m: int = 9, alpha: float = 0.2, gamma: float = 0.95, epsilon: float = 0.1, seed: int | None = None):
        self.n = n
        self.m = m
        if seed is not None:
            np.random.seed(seed)
        self.pitch = init_pitch(n, m)            # uint8
        self.reward = init_reward(n, m)          # float32
        self.Q = np.full((2, n, m, n, m, NA), 10.0, dtype=np.float32)
        self.alpha = np.float32(alpha)
        self.gamma = np.float32(gamma)
        self.epsilon = float(epsilon)

    def train(self, episodes: int = 20000, max_moves: int = 500) -> dict:
        goals = np.zeros(2, dtype=np.int64)
        ip = np.zeros(2, dtype=np.int32)
        jp = np.zeros(2, dtype=np.int32)
        terminals = 0
        tot_len = 0
        for _ in range(episodes):
            ip0, jp0, ip1, jp1, ib, jb, iip = prepare_round_jit(self.pitch, self.reward)
            ip[0], jp[0] = ip0, jp0
            ip[1], jp[1] = ip1, jp1
            moves = 0
            while moves < max_moves:
                terminal, ib, jb, iip, g0, g1 = step_round(self.Q, self.pitch, self.reward, ip, jp, ib, jb, iip, self.alpha, self.gamma, self.epsilon)
                goals[0] += g0
                goals[1] += g1
                moves += 1
                if terminal:
                    terminals += 1
                    break
            tot_len += moves
        return {"episodes": episodes, "terminals": int(terminals), "avg_len": (tot_len / episodes) if episodes > 0 else 0.0,
                "goals": (int(goals[0]), int(goals[1]))}

    def save_player(self, path: str, p: int):
        np.save(path, self.Q[p])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=20000)
    ap.add_argument('--max-moves', type=int, default=500)
    ap.add_argument('--epsilon', type=float, default=0.2)
    ap.add_argument('--alpha', type=float, default=0.2)
    ap.add_argument('--gamma', type=float, default=0.95)
    ap.add_argument('--optimistic', type=float, default=10.0)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    env = QFootyNumba(n=13, m=10, alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon, seed=args.seed)
    # optimistic init
    env.Q.fill(np.float32(args.optimistic))

    print("Training (Numba)...")
    stats = env.train(episodes=args.episodes, max_moves=args.max_moves)
    print("Stats:", stats)

    env.save_player("q_player0.npy", 0)
    env.save_player("q_player1.npy", 1)
    print("Saved q_player0.npy and q_player1.npy (5 actions)")
