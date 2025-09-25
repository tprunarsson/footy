from __future__ import annotations
import argparse, random, time
from typing import List, Tuple, Optional
import numpy as np
import itertools

# Import your two policy families
from single_footy_policy import SingleFooty
from single_block_policy import SingleBlock
from single_agsmab_policy import SingleAgsmab

# --- add near the top ---
import inspect

def _supports_kw(fn, name: str) -> bool:
    try:
        sig = inspect.signature(fn)
        return name in sig.parameters
    except Exception:
        return False

DEFAULT_REWARD = -1
GOAL_REWARD = 100
ACTION_NAMES = ['N','E','S','W','Stay']
ACTIONS = [(-1,0),(0,1),(1,0),(0,-1),(0,0)]  # last is Stay

# ------------- Environment helpers -------------------- #

def init_pitch(n: int, m: int) -> np.ndarray:
    pitch = np.zeros((n, m), dtype=bool)
    pitch[1:n-1, 1:m-1] = True
    pitch[:, 0] = pitch[:, m-1] = False
    pitch[0, :] = pitch[n-1, :] = False
    return pitch

def init_reward(n: int, m: int) -> np.ndarray:
    r = np.full((2, n, m), DEFAULT_REWARD, dtype=np.int32)
    top, bot = 1, n - 2
    ml, mr = m // 2 - 1, m // 2
    # P0 scores bottom / defends top
    r[0, bot, ml] = r[0, bot, mr] = GOAL_REWARD
    r[0, top, ml] = r[0, top, mr] = -GOAL_REWARD
    # P1 scores top / defends bottom
    r[1, top, ml] = r[1, top, mr] = GOAL_REWARD
    r[1, bot, ml] = r[1, bot, mr] = -GOAL_REWARD
    return r

def initialize_round(n: int, m: int) -> tuple[int, int, tuple[int, int], tuple[int, int]]:
    """
    Ball: anywhere on center line.
    P0: inside bottom goal (row n-2, col = mid_left or mid_right).
    P1: inside top goal (row 1,   col = mid_left or mid_right).
    """
    # Ball on center row, random non-wall column
    ib = n // 2
    jb = random.randint(1, m - 2)

    # Goal columns (two cells centered)
    ml = m // 2 - 1
    mr = m // 2

    # Players spawn inside their own goal
    ip0, jp0 = n - 2, random.choice((ml, mr))  # bottom goal cell
    ip1, jp1 = 1,       random.choice((ml, mr))  # top goal cell

    # (Paranoia) Ensure not on ball — shouldn’t happen with center vs goals,
    # but if you ever change grid sizes/goals this keeps things safe.
    if (ip0, jp0) == (ib, jb):
        # move ball one step right (wrap-safe)
        jb = mr if jb == ml else ml
    if (ip1, jp1) == (ib, jb):
        jb = ml if jb == mr else mr

    return ib, jb, (ip0, jp0), (ip1, jp1)

def is_goal_next(n: int, m: int, ib_next: int, jb_next: int) -> bool:
    top, bot = 1, n - 2
    ml, mr = m // 2 - 1, m // 2
    return (jb_next in (ml, mr)) and (ib_next in (top, bot))

# ------------------------ ASCII rendering ---------------------------------- #

def render_ascii(n, m, pitch, ip0, jp0, ip1, jp1, ib, jb, labels="numbers"):
    chars = [['#' if not pitch[i, j] else '.' for j in range(m)] for i in range(n)]
    top, bot = 1, n - 2
    ml, mr = m // 2 - 1, m // 2
    chars[top][ml] = chars[top][mr] = '+'
    chars[bot][ml] = chars[bot][mr] = '+'
    if labels == "numbers":
        chars[ip0][jp0] = '0'; chars[ip1][jp1] = '1'
    else:
        chars[ip0][jp0] = 'R'; chars[ip1][jp1] = 'B'
    chars[ib][jb] = 'o'
    print('\n'.join(' '.join(row) for row in chars))

# ---- Pygame Renderer (optional) ----
try:
    import pygame
except ImportError:
    pygame = None

COLORS = {
    "bg": (34, 139, 34), "wall": (90, 90, 90), "goal": (30, 180, 30),
    "p0": (220, 50, 50), "p1": (50, 100, 240),
    "ball": (210, 180, 20), "text": (240, 240, 240),
}

class PygameRenderer:
    def __init__(self, n=13, m=10, cell=40, margin=20, labels="numbers", fps=30):
        if pygame is None:
            raise RuntimeError("Install pygame: pip install pygame")
        pygame.init()
        self.n, self.m = n, m
        self.cell, self.margin, self.labels, self.fps = cell, margin, labels, fps
        w = margin*2 + m*cell; h = margin*2 + n*cell
        self.screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption("Q-Footy Tournament")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, int(cell*0.6))

    def _rc2xy(self, i, j):
        x = self.margin + j*self.cell; y = self.margin + i*self.cell
        return pygame.Rect(x, y, self.cell, self.cell)

    def draw(self, pitch, ip0, jp0, ip1, jp1, ib, jb, score0, score1, round_idx=None, names=None):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit
        self.screen.fill(COLORS["bg"])
        n, m = pitch.shape
        top, bot = 1, n-2
        ml, mr = m//2 - 1, m//2

        for i in range(n):
            for j in range(m):
                r = self._rc2xy(i, j)
                if not pitch[i, j]:
                    pygame.draw.rect(self.screen, COLORS["wall"], r)
        for j in (ml, mr):
            pygame.draw.rect(self.screen, COLORS["goal"], self._rc2xy(top, j))
            pygame.draw.rect(self.screen, COLORS["goal"], self._rc2xy(bot, j))

        # players
        for (ip, jp, color, label) in [(ip0, jp0, COLORS["p0"], "0"), (ip1, jp1, COLORS["p1"], "1")]:
            r = self._rc2xy(ip, jp)
            pygame.draw.rect(self.screen, color, r, border_radius=int(self.cell*0.2))
            surf = self.font.render(label, True, (255,255,255))
            self.screen.blit(surf, (r.x + r.w*0.33, r.y + r.h*0.18))

        # ball
        rb = self._rc2xy(ib, jb).inflate(-self.cell*0.4, -self.cell*0.4)
        pygame.draw.ellipse(self.screen, COLORS["ball"], rb)

        # HUD
        title = f"{names[0]} vs {names[1]} | " if names else ""
        hud = f"{title}Round {round_idx} | P0 {score0} - {score1} P1"
        hud_surf = self.font.render(hud, True, COLORS["text"])
        self.screen.blit(hud_surf, (self.margin, 5))
        pygame.display.flip()
        self.clock.tick(self.fps)

# ------------------------ Policy registry & loader ------------------------- #

class PolicyWrap:
    def __init__(self, kind: str, path: str, n: int, m: int, epsilon: float, debug: bool, player_id: int):
        self.kind = kind.lower()
        self.path = path
        self.player_id = player_id

        # Try constructing with player_id when supported; fallback otherwise.
        if self.kind == "footy":
            if _supports_kw(SingleFooty.__init__, "player_id"):
                self.obj = SingleFooty(n=n, m=m, epsilon=epsilon, debug=debug, player_id=player_id)
            else:
                self.obj = SingleFooty(n=n, m=m, epsilon=epsilon, debug=debug)
        elif self.kind == "block":
            if _supports_kw(SingleBlock.__init__, "player_id"):
                self.obj = SingleBlock(n=n, m=m, epsilon=epsilon, debug=debug, player_id=player_id)
            else:
                self.obj = SingleBlock(n=n, m=m, epsilon=epsilon, debug=debug)
        elif self.kind == "agsmab":
            # was SingleBlock; fix to SingleAgsmab
            if _supports_kw(SingleAgsmab.__init__, "player_id"):
                self.obj = SingleAgsmab(n=n, m=m, epsilon=epsilon, debug=debug, player_id=player_id)
            else:
                self.obj = SingleAgsmab(n=n, m=m, epsilon=epsilon, debug=debug)
        
        else:
            raise ValueError(f"Unknown policy kind '{kind}'. Use 'footy' or 'block'.")

        self.obj.load(path)
        self.name = f"{self.kind}:{path.split('/')[-1]}"
        self.na = getattr(self.obj, "na", 5)  # will be corrected after load

    def action(self, ip, jp, io, jo, ib, jb) -> int:
        # If the policy's action() supports 'p', pass the player_id; otherwise call the 6-arg version.
        if _supports_kw(self.obj.action, "p"):
            return self.obj.action(ip, jp, io, jo, ib, jb, p=self.player_id)
        else:
            return self.obj.action(ip, jp, io, jo, ib, jb)

# ------------------------ Core match (one pair) ---------------------------- #

def run_match(n, m, pitch, rounds, pol0: PolicyWrap, pol1: PolicyWrap,
              delay=0.0, ascii=False, renderer: Optional[PygameRenderer]=None, labels="numbers",
              start="random", next_policy="loser", fps=30):
    goals = [0, 0]
    # who starts first round
    if start == 'p0': iip = 0
    elif start == 'p1': iip = 1
    else: iip = random.randint(0, 1)

    for r in range(rounds):
        ib, jb, (ip0, jp0), (ip1, jp1) = initialize_round(n, m)
        if renderer:
            renderer.draw(pitch, ip0, jp0, ip1, jp1, ib, jb, goals[0], goals[1], round_idx=r+1, names=(pol0.name, pol1.name))
        elif ascii:
            print(f"\n=== Round {r+1}/{rounds} | score P0:{goals[0]} P1:{goals[1]} | start player {iip} ===")
            render_ascii(n, m, pitch, ip0, jp0, ip1, jp1, ib, jb, labels=labels)

        while True:
            p = iip
            # choose action via the correct policy
            if p == 0:
                a = pol0.action(ip0, jp0, ip1, jp1, ib, jb)
                na = pol0.na
            else:
                a = pol1.action(ip1, jp1, ip0, jp0, ib, jb)
                na = pol1.na

            # map to displacement
            di_p, dj_p = (0, 0) if a == -1 else ACTIONS[a][:2]
            # legality incl. occupancy
            cur_ip, cur_jp = (ip0, jp0) if p == 0 else (ip1, jp1)
            opp_ip, opp_jp = (ip1, jp1) if p == 0 else (ip0, jp0)

            def legal(di, dj):
                ii, jj = cur_ip + di, cur_jp + dj
                return pitch[ii, jj] and not (ii == opp_ip and jj == opp_jp)

            if not legal(di_p, dj_p):
                # pick any legal action (from that policy's action set)
                candidates = []
                for k in range(na):
                    di, dj = ACTIONS[k]
                    if legal(di, dj):
                        candidates.append((k, di, dj))
                if candidates:
                    k, di_p, dj_p = random.choice(candidates)
                    a = k
                else:
                    di_p, dj_p = 0, 0  # forced stay

            # ball dynamics
            aball = None
            if di_p != 0 or dj_p != 0:
                if p == 0 and ib == ip0 + di_p and jb == jp0 + dj_p:
                    aball = (di_p, dj_p)
                elif p == 1 and ib == ip1 + di_p and jb == jp1 + dj_p:
                    aball = (di_p, dj_p)

            di_b, dj_b = 0, 0
            if aball is not None:
                di_b, dj_b = aball
                attempts = 0
                while True:
                    ib2, jb2 = ib + di_b, jb + dj_b
                    blocked = (not pitch[ib2, jb2]) or ((ib2 == ip0 and jb2 == jp0) or (ib2 == ip1 and jb2 == jp1))
                    if not blocked:
                        break
                    # bounce randomly among cardinals that are free
                    cand = [a for a in ACTIONS[:4] if pitch[ib + a[0], jb + a[1]] and not ((ib + a[0], jb + a[1]) in [(ip0, jp0), (ip1, jp1)])]
                    if cand:
                        di_b, dj_b = random.choice(cand)
                    else:
                        di_b, dj_b = 0, 0
                        break
                    attempts += 1
                    if attempts > 50:
                        di_b, dj_b = 0, 0
                        break

            ib_next, jb_next = ib + di_b, jb + dj_b
            terminal = is_goal_next(n, m, ib_next, jb_next)

            # apply player move
            if p == 0:
                ip0 += di_p; jp0 += dj_p
            else:
                ip1 += di_p; jp1 += dj_p
            # apply ball move
            ib, jb = ib_next, jb_next

            if terminal:
                bot_row = n - 2
                if ib == bot_row: scorer, loser = 0, 1; goals[0] += 1
                elif ib == 1:     scorer, loser = 1, 0; goals[1] += 1
                else:             scorer, loser = p, 1 - p
                # choose next starter
                if next_policy == 'loser':   iip = loser
                elif next_policy == 'winner': iip = scorer
                elif next_policy == 'alternate': iip = 1 - iip
                else: iip = random.randint(0, 1)
                if ascii and not renderer:
                    print(f"GOAL! P0:{goals[0]} P1:{goals[1]} (by P{scorer})")
                    render_ascii(n, m, pitch, ip0, jp0, ip1, jp1, ib, jb, labels=labels)
                break

            # next player's turn and render
            iip = 1 - iip
            if renderer:
                renderer.draw(pitch, ip0, jp0, ip1, jp1, ib, jb, goals[0], goals[1], round_idx=r+1, names=(pol0.name, pol1.name))
            elif ascii:
                render_ascii(n, m, pitch, ip0, jp0, ip1, jp1, ib, jb, labels=labels)
            if delay: time.sleep(delay)

    return tuple(goals)

# --- helper function (place near run_match) ---
def animate_all_pairs_once(n, m, pitch, pol0_list, pol1_list, labels, start, next_policy, fps):
    """Play exactly 1 round per (P0[i], P1[j]) pair with pygame rendering."""
    if pygame is None:
        raise RuntimeError("Install pygame: pip install pygame")
    renderer = PygameRenderer(n, m, cell=40, labels=labels, fps=fps)

    print("\nAnimating one round per pair...")
    for i, p0 in enumerate(pol0_list):
        for j, p1 in enumerate(pol1_list):
            print(f"Animating P0[{i}] {p0.name} vs P1[{j}] {p1.name}")
            # one visual round; no ASCII, no delay (FPS controls speed)
            run_match(n, m, pitch, rounds=1, pol0=p0, pol1=p1,
                      delay=0.0, ascii=False, renderer=renderer, labels=labels,
                      start=start, next_policy=next_policy)

# ------------------------ Tournament driver -------------------------------- #

def parse_policy_arg(raw: str) -> Tuple[str,str]:
    """
    Accepts 'kind:path' where kind in {'footy','block'}.
    If no colon, assume 'footy'.
    """
    if ':' in raw:
        kind, path = raw.split(':', 1)
        return kind.strip(), path.strip()
    return 'footy', raw.strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--p0', action='append', required=True,
                    help="policy for player 0, e.g. 'footy:q0.npy' or 'block:q0_block.npy'. Repeatable.")
    ap.add_argument('--p1', action='append', required=True,
                    help="policy for player 1, e.g. 'footy:q1.npy' or 'block:q1_block.npy'. Repeatable.")
    ap.add_argument('--n', type=int, default=13)
    ap.add_argument('--m', type=int, default=10)
    ap.add_argument('--rounds', type=int, default=49)
    ap.add_argument('--delay', type=float, default=0.0)
    ap.add_argument('--ascii', action='store_true')
    ap.add_argument('--pygame', action='store_true')
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--labels', choices=['numbers','colors'], default='numbers')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--epsilon', type=float, default=0.05, help='epsilon for BOTH sides (per-policy).')
    ap.add_argument('--debug', action='store_true')
    ap.add_argument('--start', choices=['random','p0','p1'], default='random')
    ap.add_argument('--next', dest='next_policy', choices=['loser','winner','alternate','random'], default='loser')
    ap.add_argument('--show-pair', nargs=2, type=int, metavar=('I','J'),
                    help='Only play & render the specific pair index: P0[I] vs P1[J].')
    ap.add_argument('--animate', action='store_true',
                help='Animate one round for every P0[i] × P1[j] pair (pygame window).')

    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    n, m = args.n, args.m
    pitch = init_pitch(n, m)

    renderer = None
    if args.pygame:
        renderer = PygameRenderer(n, m, cell=40, labels=args.labels, fps=args.fps)

    # Build policy lists
    pol0_specs = [parse_policy_arg(s) for s in args.p0]
    pol1_specs = [parse_policy_arg(s) for s in args.p1]
    pol0_list = [PolicyWrap(kind, path, n, m, args.epsilon, args.debug, player_id=0) for kind, path in pol0_specs]
    pol1_list = [PolicyWrap(kind, path, n, m, args.epsilon, args.debug, player_id=1) for kind, path in pol1_specs]

    if args.animate:
    # only animate; do not run the text tournament
        animate_all_pairs_once(n, m, pitch, pol0_list, pol1_list,
                            labels=args.labels, start=args.start,
                            next_policy=args.next_policy, fps=args.fps)
        return


    # Report roster
    print("P0 roster:")
    for i, p in enumerate(pol0_list): print(f"  [{i}] {p.name} (na={p.na})")
    print("P1 roster:")
    for j, p in enumerate(pol1_list): print(f"  [{j}] {p.name} (na={p.na})")

    # Run either one highlighted pair or full round-robin
    results = np.zeros((len(pol0_list), len(pol1_list), 2), dtype=np.int64)

    if args.show_pair is not None:
        i, j = args.show_pair
        print(f"\n=== Playing highlighted pair: P0[{i}] {pol0_list[i].name}  vs  P1[{j}] {pol1_list[j].name} ===")
        g0, g1 = run_match(n, m, pitch, args.rounds, pol0_list[i], pol1_list[j],
                           delay=args.delay, ascii=args.ascii and not args.pygame,
                           renderer=renderer, labels=args.labels,
                           start=args.start, next_policy=args.next)
        results[i, j, 0] = g0; results[i, j, 1] = g1
    else:
        # tournament (no rendering to keep it fast; use --show-pair for visuals)
        for i, p0 in enumerate(pol0_list):
            for j, p1 in enumerate(pol1_list):
                g0, g1 = run_match(n, m, pitch, args.rounds, p0, p1,
                                   delay=0.0, ascii=False, renderer=None,
                                   labels=args.labels, start=args.start, next_policy=args.next_policy)
                results[i, j, 0] = g0; results[i, j, 1] = g1
                print(f"Pair P0[{i}] {p0.name} vs P1[{j}] {p1.name}: {g0}-{g1}")

    # Summary table
    print("\n=== Tournament results (P0 goals – P1 goals) ===")
    for i, p0 in enumerate(pol0_list):
        row = []
        for j, p1 in enumerate(pol1_list):
            row.append(f"{results[i,j,0]}-{results[i,j,1]}")
        print(f"[{i}] {p0.name:<30} | " + " | ".join(row))

    # Overall winner by total wins (per pair, who scored more)
    wins0 = (results[:,:,0] > results[:,:,1]).sum()
    wins1 = (results[:,:,1] > results[:,:,0]).sum()
    draws = (results[:,:,0] == results[:,:,1]).sum()
    print(f"\nTotals — P0 pair-wins: {wins0}, P1 pair-wins: {wins1}, draws: {draws}")

if __name__ == '__main__':
    main()
