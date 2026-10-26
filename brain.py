"""A simulated larval zebrafish brain that plays the blackjack table.

This is a model, not a copy of a real fish. It is a network of about 1,100 spiking
neurons (leaky integrate-and-fire) whose wiring follows numbers measured in real
larval zebrafish connectomes:

- Modules. In the hindbrain wiring diagram of Vishwanathan et al. (Nature Neuroscience,
  2024; 2,824 cells, 75,163 synapses), neurons form modules that wire about 6 times more
  densely inside a module than between modules. Here each possible move (HIT, STAND,
  DOUBLE, and three bet sizes) is one excitatory module.
- Synapses per connection. In the same dataset about 65% of connected pairs share a
  single synapse and the most is 21, so connection strengths are drawn from that
  distribution.
- Excitatory vs inhibitory. The 2025 whole-brain larval connectome resource (Harvard and
  Google; about 116,000 neurons, 30 million synapses) labels 26,915 neurons as vglut2a
  (excitatory) and 14,510 as gad1b (inhibitory): about 35% inhibitory. The network keeps
  that ratio with a shared inhibitory pool, so the modules compete and one move wins, the
  way the left and right ARTR populations compete to pick a turn direction
  (Dunn et al., Neuron 2016).

The brain sees the table through sensory neurons: its hand total, whether the hand is
soft, the dealer's up card, whether a double is allowed, and (for bets) how many chips
are left. The first move module to fire hard enough wins. After each hand a
dopamine-like reward signal (chips won or lost, minus what it expected) strengthens or
weakens the sensory-to-move synapses that were active, so the brain learns from play.
Its synapses are saved in brain.npz, so it keeps learning across sessions.

    python brain.py train --hands 3000     # let it play offline and learn
    python brain.py policy                 # what it has learned, as a table
    python brain.py decide 15 6            # one decision: hard 15 vs a 6
    python brain.py reset                  # a fresh, untrained brain

Optional: a real wiring table. --cells cells.csv (columns id,module) and --edges
edges.csv (columns pre,post,synapses) replace the generated recurrent wiring. module is
one of HIT, STAND, DOUBLE, BET_LOW, BET_MID, BET_HIGH, or I for inhibitory.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from blackjack_game import Shoe, dealer_peeks, dealer_should_hit, hand_value, is_blackjack, settle
from utils import ROOT, die

BRAIN_PATH = ROOT / "brain.npz"

ACTIONS = ["HIT", "STAND", "DOUBLE", "BET_LOW", "BET_MID", "BET_HIGH"]
PLAY_ACTIONS = ACTIONS[:3]
BET_ACTIONS = ACTIONS[3:]

SOURCES = [
    "Vishwanathan et al. 2024, Predicting modular functions and neural coding of behavior from a synaptic wiring diagram, Nature Neuroscience",
    "Petkova, Januszewski et al. 2025, A connectomic resource for neural cataloguing and circuit dissection of the larval zebrafish brain, bioRxiv 10.1101/2025.06.10.658982",
    "Dunn et al. 2016, Brain-wide mapping of neural activity controlling zebrafish exploratory locomotion, Neuron (ARTR)",
]

# Numbers taken from the papers above.
MODULE_DENSITY_RATIO = 6.0  # within-module vs between-module wiring (Vishwanathan 2024)
P_SINGLE_SYNAPSE = 0.65  # share of connections made of one synapse (Vishwanathan 2024)
MAX_SYNAPSES = 21  # most synapses on one connection (Vishwanathan 2024)
INHIBITORY_FRACTION = 14510 / (14510 + 26915)  # gad1b / (gad1b + vglut2a), 2025 resource

# Neuron model (standard LIF values; times in ms, voltages in mV).
DT = 1.0
TAU_E, TAU_I = 20.0, 10.0
TAU_SYN_E, TAU_SYN_I = 5.0, 8.0
V_REST, V_TH, V_RESET = -70.0, -50.0, -60.0
REFRACTORY = 2.0
BACKGROUND = 17.5  # mean drive, just under threshold
NOISE = 3.0

MODULE_SIZE = 80
DECISION_MS = 1000
COMMIT_HZ = 28.0
COMMIT_MARGIN_HZ = 14.0
GO_DRIVE = 1.0  # extra drive to the moves on offer
# Weight of one synapse (mV jump in synaptic current), by connection type.
W_EE, W_IE, W_EI, W_II = 2.8, 0.75, 0.55, 0.30

# Sensory layer: groups of neurons with bell-shaped tuning.
GROUP = 10
TOTALS = list(range(4, 22))  # hand total 4..21
UPCARDS = list(range(2, 12))  # dealer up card 2..11 (ace = 11)
BANK_LEVELS = 5


def _sensory_layout() -> dict[str, slice]:
    layout, i = {}, 0
    for name, groups in (("total", len(TOTALS)), ("soft", 1), ("up", len(UPCARDS)), ("double", 1), ("bet", 1), ("bank", BANK_LEVELS), ("bias", 1)):
        layout[name] = slice(i, i + groups * GROUP)
        i += groups * GROUP
    layout["_n"] = slice(0, i)
    return layout


SENSORY = _sensory_layout()
N_SENS = SENSORY["_n"].stop


@dataclass
class Context:
    """What the brain is shown: a play decision or a bet decision."""

    kind: str  # "play" or "bet"
    total: int = 0
    soft: bool = False
    up: int = 0  # dealer up card value, ace = 11
    can_double: bool = False
    bank_frac: float = 1.0  # chips left / chips at the start

    def describe(self) -> str:
        if self.kind == "bet":
            return f"bet with {self.bank_frac:.0%} of the starting chips"
        return f"{'soft' if self.soft else 'hard'} {self.total} vs {self.up if self.up < 11 else 'A'}"


def encode(ctx: Context, peak_hz: float = 30.0) -> np.ndarray:
    """Firing rate (Hz) of every sensory neuron for this table state."""
    rates = np.zeros(N_SENS, np.float32)

    def tuned(name: str, values: list[float], x: float, width: float) -> None:
        s = SENSORY[name]
        for g, v in enumerate(values):
            rates[s.start + g * GROUP : s.start + (g + 1) * GROUP] = peak_hz * np.exp(-0.5 * ((x - v) / width) ** 2)

    def flag(name: str, on: bool) -> None:
        rates[SENSORY[name]] = peak_hz if on else 0.0

    if ctx.kind == "play":
        tuned("total", TOTALS, ctx.total, 1.2)
        flag("soft", ctx.soft)
        tuned("up", UPCARDS, ctx.up, 0.9)
        flag("double", ctx.can_double)
    else:
        flag("bet", True)
        tuned("bank", list(np.linspace(0.0, 2.0, BANK_LEVELS)), min(2.0, ctx.bank_frac), 0.35)
    flag("bias", True)
    return rates


@dataclass
class Decision:
    """One decision: what was chosen, when, and the activity that led to it."""

    context: Context
    labels: list[str]
    modules: list[int]
    choice: int  # index into labels
    commit_ms: float
    rates: np.ndarray  # (ms, options) smoothed module firing rate, Hz
    raster: list[tuple[float, int]]  # (ms, display row) for a sample of neurons
    sens_counts: np.ndarray
    post_counts: np.ndarray
    reward: float | None = None

    @property
    def label(self) -> str:
        return self.labels[self.choice]


class Brain:
    def __init__(self, seed: int = 7, cells: Path | None = None, edges: Path | None = None) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.hands_trained = 0
        self.baseline = {"play": 0.0, "bet": 0.0}
        if cells is not None and edges is not None:
            self._load_wiring(cells, edges)
        else:
            self._generate_wiring()
        self.n_e = len(self.module_of)
        self.n_i = self.W_ie.shape[0]
        # Plastic sensory-to-move synapses.
        self.W_in = self.rng.uniform(0.10, 0.20, (N_SENS, self.n_e)).astype(np.float32)
        self.col_sum = self.W_in.sum(axis=0).mean()
        self.wiring_source = "generated from published statistics" if cells is None else f"loaded from {Path(cells).name} and {Path(edges).name}"
        # A fixed sample of neurons per module for the activity display.
        self.display_rows: dict[int, int] = {}
        row = 0
        for m in range(len(ACTIONS)):
            members = np.flatnonzero(self.module_of == m)
            for n in members[:12]:
                self.display_rows[int(n)] = row
                row += 1
        self.display_inh = list(range(min(18, self.n_i)))
        self.display_total = row + len(self.display_inh)

    @property
    def neuron_count(self) -> int:
        return self.n_e + self.n_i + N_SENS

    # Wiring --------------------------------------------------------------------
    def _synapses(self, mask: np.ndarray) -> np.ndarray:
        """Synapse counts for each connected pair: geometric, P(1) = 0.65, at most 21."""
        k = self.rng.geometric(P_SINGLE_SYNAPSE, size=mask.shape)
        return np.where(mask, np.minimum(k, MAX_SYNAPSES), 0).astype(np.float32)

    def _generate_wiring(self) -> None:
        n_mod = len(ACTIONS)
        n_e = n_mod * MODULE_SIZE
        n_i = int(round(n_e * INHIBITORY_FRACTION / (1 - INHIBITORY_FRACTION)))
        self.module_of = np.repeat(np.arange(n_mod), MODULE_SIZE)
        same = self.module_of[:, None] == self.module_of[None, :]
        p_in = 0.20
        p = np.where(same, p_in, p_in / MODULE_DENSITY_RATIO)
        np.fill_diagonal(p, 0.0)
        conn = self.rng.random((n_e, n_e)) < p  # [post, pre]
        self.W_ee = self._synapses(conn) * W_EE
        self.W_ie = self._synapses(self.rng.random((n_i, n_e)) < 0.20) * W_IE
        self.W_ei = self._synapses(self.rng.random((n_e, n_i)) < 0.35) * W_EI
        self.W_ii = self._synapses(self.rng.random((n_i, n_i)) < 0.20) * W_II
        self._balance()

    def _balance(self) -> None:
        """Homeostatic scaling: every excitatory neuron gets the same total excitatory and
        inhibitory input as the average, keeping the wiring pattern but no built-in favourite."""
        for W in (self.W_ee, self.W_ei):
            rows = W.sum(axis=1, keepdims=True)
            W *= rows.mean() / np.maximum(rows, 1e-6)

    def _load_wiring(self, cells_path: Path, edges_path: Path) -> None:
        ids_e, ids_i, module = [], [], []
        with open(cells_path, newline="") as fh:
            for row in csv.DictReader(fh):
                m = row["module"].strip().upper()
                if m == "I":
                    ids_i.append(row["id"])
                elif m in ACTIONS:
                    ids_e.append(row["id"])
                    module.append(ACTIONS.index(m))
        if not ids_e or not ids_i:
            die("The cells table needs excitatory cells assigned to moves and some inhibitory (I) cells.")
        index_e = {c: i for i, c in enumerate(ids_e)}
        index_i = {c: i for i, c in enumerate(ids_i)}
        self.module_of = np.array(module)
        n_e, n_i = len(ids_e), len(ids_i)
        self.W_ee = np.zeros((n_e, n_e), np.float32)
        self.W_ie = np.zeros((n_i, n_e), np.float32)
        self.W_ei = np.zeros((n_e, n_i), np.float32)
        self.W_ii = np.zeros((n_i, n_i), np.float32)
        with open(edges_path, newline="") as fh:
            for row in csv.DictReader(fh):
                pre, post, k = row["pre"], row["post"], float(row.get("synapses") or 1)
                if pre in index_e and post in index_e:
                    self.W_ee[index_e[post], index_e[pre]] += k * W_EE
                elif pre in index_e and post in index_i:
                    self.W_ie[index_i[post], index_e[pre]] += k * W_IE
                elif pre in index_i and post in index_e:
                    self.W_ei[index_e[post], index_i[pre]] += k * W_EI
                elif pre in index_i and post in index_i:
                    self.W_ii[index_i[post], index_i[pre]] += k * W_II
        self._balance()

    # Dynamics ------------------------------------------------------------------
    def decide(self, ctx: Context, labels: list[str], rng: np.random.Generator | None = None, full: bool = True) -> Decision:
        """Run the network until one offered move wins. ``labels`` are the options shown."""
        rng = rng or self.rng
        modules = [_module_for(label, ctx.kind, i, len(labels)) for i, label in enumerate(labels)]
        rates_in = encode(ctx)
        p_spike = rates_in * DT / 1000.0
        n_e, n_i = self.n_e, self.n_i
        go = np.full(n_e, -8.0, np.float32)  # moves not on offer are held down
        for m in modules:
            go[self.module_of == m] = GO_DRIVE
        v_e = V_REST + rng.uniform(0, 15, n_e).astype(np.float32)
        v_i = V_REST + rng.uniform(0, 15, n_i).astype(np.float32)
        s_ee = np.zeros(n_e, np.float32)
        s_ei = np.zeros(n_e, np.float32)
        s_ie = np.zeros(n_i, np.float32)
        s_ii = np.zeros(n_i, np.float32)
        ref_e = np.zeros(n_e, np.float32)
        ref_i = np.zeros(n_i, np.float32)
        decay_e, decay_i = np.float32(np.exp(-DT / TAU_SYN_E)), np.float32(np.exp(-DT / TAU_SYN_I))
        k_e, k_i = DT / TAU_E, DT / TAU_I
        noise_e, noise_i = NOISE * np.sqrt(2 * k_e), NOISE * np.sqrt(2 * k_i)
        member = [np.flatnonzero(self.module_of == m) for m in modules]
        smooth = np.zeros(len(modules))
        alpha = DT / 25.0
        rate_trace = []
        raster: list[tuple[float, int]] = []
        sens_counts = np.zeros(N_SENS, np.float32)
        post_counts = np.zeros(n_e, np.float32)
        commit_ms, choice = None, None
        steps = int(DECISION_MS / DT)
        for step in range(steps):
            t = step * DT
            sens = np.flatnonzero(rng.random(N_SENS) < p_spike)
            if sens.size:
                s_ee += self.W_in[sens].sum(axis=0)
                if commit_ms is None:
                    sens_counts[sens] += 1
            v_e += k_e * (V_REST - v_e + BACKGROUND + go + s_ee - s_ei) + noise_e * rng.standard_normal(n_e).astype(np.float32)
            v_i += k_i * (V_REST - v_i + BACKGROUND - 2.0 + s_ie - s_ii) + noise_i * rng.standard_normal(n_i).astype(np.float32)
            v_e[ref_e > 0] = V_RESET
            v_i[ref_i > 0] = V_RESET
            ref_e -= DT
            ref_i -= DT
            fe = np.flatnonzero(v_e >= V_TH)
            fi = np.flatnonzero(v_i >= V_TH)
            v_e[fe] = V_RESET
            v_i[fi] = V_RESET
            ref_e[fe] = REFRACTORY
            ref_i[fi] = REFRACTORY
            s_ee *= decay_e
            s_ei *= decay_i
            s_ie *= decay_e
            s_ii *= decay_i
            if fe.size:
                s_ee += self.W_ee[:, fe].sum(axis=1)
                s_ie += self.W_ie[:, fe].sum(axis=1)
                if commit_ms is None:
                    post_counts[fe] += 1
            if fi.size:
                s_ei += self.W_ei[:, fi].sum(axis=1)
                s_ii += self.W_ii[:, fi].sum(axis=1)
            fired = np.zeros(n_e, bool)
            fired[fe] = True
            inst = np.array([fired[idx].mean() * 1000.0 / DT for idx in member])
            smooth += alpha * (inst - smooth)
            rate_trace.append(smooth.copy())
            if full:
                for n in fe:
                    r = self.display_rows.get(int(n))
                    if r is not None:
                        raster.append((t, r))
                for n in fi:
                    if n < len(self.display_inh):
                        raster.append((t, len(self.display_rows) + int(n)))
            if commit_ms is None:
                order = np.argsort(smooth)[::-1]
                top = smooth[order[0]]
                second = smooth[order[1]] if len(order) > 1 else 0.0
                if top >= COMMIT_HZ and top - second >= COMMIT_MARGIN_HZ:
                    commit_ms, choice = t, int(order[0])
                    if not full:
                        break
        if choice is None:
            # Nobody crossed the line in time: the most active module takes it.
            totals = np.array([post_counts[idx].sum() for idx in member])
            choice = int(np.argmax(totals + rng.random(len(totals)) * 1e-3))
            commit_ms = float(DECISION_MS)
        return Decision(ctx, list(labels), modules, choice, float(commit_ms), np.array(rate_trace), raster, sens_counts, post_counts)

    # Learning -------------------------------------------------------------------
    def learn(self, decisions: list[Decision], reward: float, lr: float = 0.004) -> None:
        """Dopamine-like update: active sensory-to-move synapses of the chosen move are
        strengthened when the hand beat expectation and weakened when it fell short."""
        for d in decisions:
            kind = d.context.kind
            rpe = reward - self.baseline[kind]
            self.baseline[kind] += 0.02 * (reward - self.baseline[kind])
            cols = np.flatnonzero(self.module_of == d.modules[d.choice])
            pre = d.sens_counts / max(1.0, d.sens_counts.max())
            post = d.post_counts[cols] / max(1.0, d.post_counts[cols].max())
            self.W_in[:, cols] += lr * rpe * np.outer(pre, post + 0.1)
            np.clip(self.W_in, 0.0, 1.0, out=self.W_in)
            # Keep each neuron's total input fixed, so learning moves preference, not volume.
            sums = self.W_in[:, cols].sum(axis=0, keepdims=True)
            self.W_in[:, cols] *= self.col_sum / np.maximum(sums, 1e-6)
            d.reward = reward

    # Saving --------------------------------------------------------------------
    def save(self, path: Path = BRAIN_PATH) -> None:
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp,
            seed=self.seed,
            W_in=self.W_in,
            W_ee=self.W_ee,
            W_ie=self.W_ie,
            W_ei=self.W_ei,
            W_ii=self.W_ii,
            module_of=self.module_of,
            hands=self.hands_trained,
            base_play=self.baseline["play"],
            base_bet=self.baseline["bet"],
            source=self.wiring_source,
        )
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path = BRAIN_PATH, seed: int = 7) -> "Brain":
        if not path.exists():
            return cls(seed)
        data = np.load(path, allow_pickle=False)
        brain = cls.__new__(cls)
        brain.seed = int(data["seed"])
        brain.rng = np.random.default_rng()
        brain.W_in = data["W_in"]
        brain.W_ee, brain.W_ie, brain.W_ei, brain.W_ii = data["W_ee"], data["W_ie"], data["W_ei"], data["W_ii"]
        brain.module_of = data["module_of"]
        brain.n_e, brain.n_i = len(brain.module_of), brain.W_ie.shape[0]
        brain.hands_trained = int(data["hands"])
        brain.baseline = {"play": float(data["base_play"]), "bet": float(data["base_bet"])}
        brain.wiring_source = str(data["source"])
        brain.col_sum = np.random.default_rng(brain.seed).uniform(0.10, 0.20, (N_SENS, brain.n_e)).sum(axis=0).mean()
        brain.display_rows, row = {}, 0
        for m in range(len(ACTIONS)):
            for n in np.flatnonzero(brain.module_of == m)[:12]:
                brain.display_rows[int(n)] = row
                row += 1
        brain.display_inh = list(range(min(18, brain.n_i)))
        brain.display_total = row + len(brain.display_inh)
        return brain


def _module_for(label: str, kind: str, index: int, count: int) -> int:
    if kind == "play":
        return ACTIONS.index(label)
    # Bets: smallest offered -> BET_LOW, largest -> BET_HIGH.
    if count == 1:
        return ACTIONS.index("BET_LOW")
    return ACTIONS.index(BET_ACTIONS[round(index * 2 / (count - 1))])


def play_context(cards, dealer_up, can_double: bool) -> Context:
    total, soft = hand_value(cards)
    up = 11 if dealer_up.rank == "A" else dealer_up.points
    return Context("play", total=total, soft=soft, up=up, can_double=can_double)


# Offline play -------------------------------------------------------------------
def play_hand(brain: Brain, shoe: Shoe, bankroll: int, start: int, bets=(10, 25, 50), learn: bool = True) -> tuple[int, list[Decision]]:
    """One hand of table rules with the brain making every choice. Returns chip change."""
    if shoe.needs_shuffle():
        shoe.shuffle()
    offered = [b for b in bets if b <= bankroll] or [bankroll]
    bet_decision = brain.decide(Context("bet", bank_frac=bankroll / start), [str(b) for b in offered], full=False)
    bet = int(bet_decision.label)
    fish = [shoe.draw(), shoe.draw()]
    dealer = [shoe.draw(), shoe.draw()]
    plays: list[Decision] = []
    if not (is_blackjack(fish) or (dealer_peeks(dealer[0]) and is_blackjack(dealer))):
        first = True
        while hand_value(fish)[0] < 21:
            can_double = first and bankroll >= bet * 2
            labels = ["HIT", "STAND"] + (["DOUBLE"] if can_double else [])
            d = brain.decide(play_context(fish, dealer[0], can_double), labels, full=False)
            plays.append(d)
            if d.label == "STAND":
                break
            fish.append(shoe.draw())
            if d.label == "DOUBLE":
                bet *= 2
                break
            first = False
        if hand_value(fish)[0] <= 21:
            while dealer_should_hit(dealer):
                dealer.append(shoe.draw())
    _outcome, delta = settle(fish, dealer, bet)
    if learn:
        if plays:
            brain.learn(plays, delta / max(1, bet))
        brain.learn([bet_decision], delta / 50.0)
        brain.hands_trained += 1
    return delta, [bet_decision] + plays


def policy_table(brain: Brain, trials: int = 6) -> str:
    rng = np.random.default_rng(1)
    ups = [2, 4, 6, 7, 9, 10, 11]
    lines = ["hard total vs dealer up card (H = hit, S = stand, D = double; majority of trials)", "      " + "  ".join(f"{u if u < 11 else 'A':>2}" for u in ups)]
    for total in range(8, 21):
        row = []
        for up in ups:
            votes = {"HIT": 0, "STAND": 0, "DOUBLE": 0}
            for _ in range(trials):
                votes[brain.decide(Context("play", total=total, up=up, can_double=True), ["HIT", "STAND", "DOUBLE"], rng, full=False).label] += 1
            row.append(max(votes, key=votes.get)[0])
        lines.append(f"  {total:>2}  " + "  ".join(f"{c:>2}" for c in row))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="A simulated zebrafish brain for the blackjack table.")
    parser.add_argument("command", choices=("train", "policy", "decide", "reset", "info"))
    parser.add_argument("args", nargs="*")
    parser.add_argument("--hands", type=int, default=2000)
    parser.add_argument("--brain", type=Path, default=BRAIN_PATH)
    parser.add_argument("--seed", type=int, default=7, help="Which simulated fish (wiring seed) for a fresh brain.")
    parser.add_argument("--cells", type=Path, default=None)
    parser.add_argument("--edges", type=Path, default=None)
    a = parser.parse_args(argv)

    if a.command == "reset" or (a.cells and a.edges):
        brain = Brain(a.seed, a.cells, a.edges)
        brain.save(a.brain)
        print(f"Fresh brain saved to {a.brain} ({brain.neuron_count:,} neurons, wiring {brain.wiring_source}).")
        if a.command == "reset":
            return
    brain = Brain.load(a.brain, a.seed)

    if a.command == "info":
        print(f"{brain.neuron_count:,} neurons ({N_SENS} sensory, {brain.n_e} excitatory move neurons in {len(ACTIONS)} modules, {brain.n_i} inhibitory)")
        print(f"Wiring: {brain.wiring_source}. Hands learned from: {brain.hands_trained:,}.")
        for s in SOURCES:
            print(f"  - {s}")
    elif a.command == "decide":
        total, up = int(a.args[0]), int(a.args[1]) if len(a.args) > 1 else 10
        d = brain.decide(Context("play", total=total, up=up, can_double=True), ["HIT", "STAND", "DOUBLE"], full=False)
        print(f"{d.context.describe()}: {d.label} (committed at {d.commit_ms:.0f} ms)")
    elif a.command == "policy":
        print(f"After {brain.hands_trained:,} hands:")
        print(policy_table(brain))
    elif a.command == "train":
        shoe = Shoe(6, random.Random(a.seed))
        start, bank, window = 1000, 1000, []
        for n in range(1, a.hands + 1):
            delta, _ = play_hand(brain, shoe, max(bank, 10), start)
            bank = max(10, bank + delta)
            window.append(delta)
            if n % 100 == 0:
                print(f"  hand {brain.hands_trained:>6,}: average {np.mean(window):+.1f} chips/hand over the last 100", flush=True)
                window = []
                brain.save(a.brain)
        brain.save(a.brain)
        print(f"Saved {a.brain}. Run python brain.py policy to see what it learned.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        raise SystemExit(130)
