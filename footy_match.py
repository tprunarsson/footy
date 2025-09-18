"""
Two-player match harness for Q-footy policies.

Key updates:
- Players **cannot move into the opponent's cell** (blocked).
- Optional **Stay** action supported end-to-end if the loaded Q-slab has 5 actions.
  If not, and a player has no legal moves, we now **no-op** (stand still) safely.
- `--labels numbers|colors`, `--start`, `--next`, `--debug` as before.
"""
from __future__ import annotations
import argparse
import random
import time
from typing import Tuple

import numpy as np

from single_footy_policy import SingleFooty

DEFAULT_REWARD = -1
GOAL_REWARD = 100
ACTION_NAMES = ['N','E','S','W','Stay']
ACTIONS = [
    (-1, 0),  # N
    (0, 1),   # E
    (1, 0),   # S
    (0, -1),  # W
    (0, 0),   # Stay
]

# ------------- Environment helpers -------------------- #

def init_pitch(n: int, m: int) -> np.ndarray:
    pitch = np.zeros((n, m), dtype=bool)
    pitch[1:n-1, 1:m-1] = True
    pitch[:, 0] = False
    pitch[:, m-1] = False
    pitch[0, :] = False
    pitch[n-1, :] = False
    return pitch


def init_reward(n: int, m: int) -> np.ndarray:
    reward = np.full((2, n, m), DEFAULT_REWARD, dtype=np.int32)
    top_row = 1
    bot_row = n - 2
    ml = m // 2 - 1
    mr = m // 2
    reward[0, bot_row, ml] = reward[0, bot_row, mr] = GOAL_REWARD
    reward[0, top_row, ml] = reward[0, top_row, mr] = -GOAL_REWARD
    reward[1, top_row, ml] = reward[1, top_row, mr] = GOAL_REWARD
    reward[1, bot_row, ml] = reward[1, bot_row, mr] = -GOAL_REWARD
    return reward


def initialize_round(n: int, m: int) -> Tuple[int,int,Tuple[int,int],Tuple[int,int]]:
    ib = n // 2
    jb = random.randint(1, m - 2)
    ip0 = random.randint(1, (n - 2) // 2)
    jp0 = random.randint(1, m - 2)
    ip1 = random.randint(1, (n - 2) // 2) + n // 2
    jp1 = random.randint(1, m - 2)
    return ib, jb, (ip0, jp0), (ip1, jp1)


def is_goal_next(n: int, m: int, ib_next: int, jb_next: int) -> bool:
    top_row = 1
    bot_row = n - 2
    ml = m // 2 - 1
    mr = m // 2
    return (jb_next in (ml, mr)) and (ib_next in (top_row, bot_row))


# ------------------------ ASCII rendering ---------------------------------- #

def render_ascii(n, m, pitch, ip0, jp0, ip1, jp1, ib, jb, labels="numbers"):
    chars = [['#' if not pitch[i, j] else '.' for j in range(m)] for i in range(n)]
    top_row = 1
    bot_row = n - 2
    ml = m // 2 - 1
    mr = m // 2
    chars[top_row][ml] = chars[top_row][mr] = '+'
    chars[bot_row][ml] = chars[bot_row][mr] = '+'
    if labels == "numbers":
        chars[ip0][jp0] = '0'
        chars[ip1][jp1] = '1'
    else:
        chars[ip0][jp0] = 'R'
        chars[ip1][jp1] = 'B'
    chars[ib][jb] = 'o'
    s = '\n'.join(' '.join(row) for row in chars)
    print(s)


# ------------------------ Match Runner ------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--p0', required=True, help='path to q slab for player 0 (.npy)')
    ap.add_argument('--p1', required=True, help='path to q slab for player 1 (.npy)')
    ap.add_argument('--n', type=int, default=13)
    ap.add_argument('--m', type=int, default=9)
    ap.add_argument('--rounds', type=int, default=49)
    ap.add_argument('--delay', type=float, default=0.0, help='seconds between steps (ASCII mode)')
    ap.add_argument('--ascii', action='store_true', help='render a simple ASCII field each step')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--mirror', action='store_true', help='Use player-0 policy for player-1 via horizontal flip trick')
    ap.add_argument('--labels', choices=['numbers','colors'], default='numbers', help='ASCII labels for players')
    ap.add_argument('--start', choices=['random','p0','p1'], default='random', help='Who starts in round 1')
    ap.add_argument('--next', dest='next_policy', choices=['loser','winner','alternate','random'], default='loser', help='Who starts rounds after each goal')
    ap.add_argument('--debug', action='store_true', help='print per-move Q-values and chosen action')
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    n, m = args.n, args.m
    pitch = init_pitch(n, m)
    reward = init_reward(n, m)

    pol0 = SingleFooty(n=n, m=m, epsilon=0.05, debug=args.debug)
    pol0.load(args.p0)
    pol1 = SingleFooty(n=n, m=m, epsilon=0.05, debug=args.debug)
    if args.mirror:
        pol1.load(args.p0)
        mirror = True
    else:
        pol1.load(args.p1)
        mirror = False

    # Dynamic action sets (support 4- or 5-action slabs per player)
    def actions_for(p):
        # Always slice the exported 5-action list to each player's na
        return ACTIONS[: (pol0.na if p == 0 else pol1.na)]

    goals = [0, 0]
    if args.start == 'p0':
        iip = 0
    elif args.start == 'p1':
        iip = 1
    else:
        iip = random.randint(0, 1)

    for r in range(args.rounds):
        ib, jb, (ip0, jp0), (ip1, jp1) = initialize_round(n, m)
        if args.ascii:
            print(f"\n=== Round {r+1}/{args.rounds} | score P0:{goals[0]} P1:{goals[1]} | start player {iip} ===")
            render_ascii(n, m, pitch, ip0, jp0, ip1, jp1, ib, jb, labels=args.labels)

        while True:
            p = iip
            acts = actions_for(p)
            # choose action
            if p == 0:
                a = pol0.action(ip0, jp0, ip1, jp1, ib, jb)
            else:
                if mirror:
                    jp_p = m - 1 - jp1
                    jo_p = m - 1 - jp0
                    jb_p = m - 1 - jb
                    a_pre = pol1.action(ip1, jp_p, ip0, jo_p, ib, jb_p)
                    # flip E<->W (indices 1 and 3) if present
                    if pol1.na >= 4:
                        if a_pre == 1:
                            a = 3
                        elif a_pre == 3:
                            a = 1
                        else:
                            a = a_pre
                    else:
                        a = a_pre
                else:
                    a = pol1.action(ip1, jp1, ip0, jp0, ib, jb)

            # Derive displacement; handle no-op sentinel (-1) from policy
            if a == -1:
                di_p, dj_p = 0, 0
            else:
                di_p, dj_p = acts[a]

            # Check legality including occupancy; if illegal, fallback to any legal; if none, stand still
            cur_ip = ip0 if p == 0 else ip1
            cur_jp = jp0 if p == 0 else jp1
            opp_ip = ip1 if p == 0 else ip0
            opp_jp = jp1 if p == 0 else jp0

            def legal_move(di, dj):
                ii, jj = cur_ip + di, cur_jp + dj
                return pitch[ii, jj] and not (ii == opp_ip and jj == opp_jp)

            if not legal_move(di_p, dj_p):
                legal = [(k, acts[k]) for k in range(len(acts)) if legal_move(*acts[k])]
                if legal:
                    a, (di_p, dj_p) = random.choice(legal)
                else:
                    di_p, dj_p = 0, 0  # forced stand still

            # Ball dynamics
            aball = None
            if di_p == 0 and dj_p == 0:
                aball = None  # no-op
            else:
                if p == 0 and ib == ip0 + di_p and jb == jp0 + dj_p:
                    aball = (di_p, dj_p)
                elif p == 1 and ib == ip1 + di_p and jb == jp1 + dj_p:
                    aball = (di_p, dj_p)

            di_b, dj_b = 0, 0
            if aball is not None:
                di_b, dj_b = aball
                # bounce if blocked by wall or opponent
                attempts = 0
                while True:
                    ib2, jb2 = ib + di_b, jb + dj_b
                    blocked = (not pitch[ib2, jb2]) or ((ib2 == ip0 and jb2 == jp0) or (ib2 == ip1 and jb2 == jp1))
                    if not blocked:
                        break
                    # pick random cardinal bounce
                    cand = [a for a in ACTIONS[:4] if not ((ib + a[0], jb + a[1]) in [(ip0, jp0), (ip1, jp1)]) and pitch[ib + a[0], jb + a[1]]]
                    if cand:
                        di_b, dj_b = random.choice(cand)
                    else:
                        di_b, dj_b = 0, 0
                        break
                    attempts += 1
                    if attempts > 50:
                        di_b, dj_b = 0, 0
                        break

            # Next ball location
            ib_next, jb_next = ib + di_b, jb + dj_b
            terminal = is_goal_next(n, m, ib_next, jb_next)

            # Apply player move
            if p == 0:
                ip0 += di_p
                jp0 += dj_p
            else:
                ip1 += di_p
                jp1 += dj_p

            # Apply ball move
            ib, jb = ib_next, jb_next

            if terminal:
                bot_row = n - 2
                if ib == bot_row:
                    goals[0] += 1
                    scorer = 0; loser = 1
                elif ib == 1:
                    goals[1] += 1
                    scorer = 1; loser = 0
                else:
                    scorer = p; loser = 1 - p
                if args.next_policy == 'loser':
                    iip = loser
                elif args.next_policy == 'winner':
                    iip = scorer
                elif args.next_policy == 'alternate':
                    iip = 1 - iip
                else:
                    iip = random.randint(0, 1)
                if args.ascii:
                    print(f"GOAL! P0:{goals[0]} P1:{goals[1]} (by P{scorer})")
                    render_ascii(n, m, pitch, ip0, jp0, ip1, jp1, ib, jb, labels=args.labels)
                break

            # Next player's turn
            iip = 1 - iip
            if args.ascii:
                render_ascii(n, m, pitch, ip0, jp0, ip1, jp1, ib, jb, labels=args.labels)
                if args.delay:
                    time.sleep(args.delay)

    print(f"\nFinal score after {args.rounds} rounds: P0 {goals[0]} - P1 {goals[1]}")
    if goals[0] > goals[1]:
        print("Player 0 wins!")
    elif goals[0] < goals[1]:
        print("Player 1 wins!")
    else:
        print("Draw!")


if __name__ == '__main__':
    main()
