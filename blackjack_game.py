"""Blackjack rules for the fish table. Play chips only; nothing here touches money.

House rules (printed on the table):
- 6-deck shoe, reshuffled once three quarters have been dealt.
- Dealer stands on all 17s and peeks for blackjack under an ace or a ten.
- Blackjack pays 3 to 2. Double on any first two cards (one more card, then stand).
- No splits, no insurance, no surrender.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from utils import ROOT, atomic_write_text, iso_timestamp

BLACKJACK_PATH = ROOT / "blackjack.json"
RANKS = ("A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K")
SUITS = ("S", "H", "D", "C")


@dataclass(frozen=True)
class Card:
    rank: str
    suit: str

    @property
    def points(self) -> int:
        if self.rank == "A":
            return 1
        if self.rank in ("J", "Q", "K"):
            return 10
        return int(self.rank)

    @property
    def red(self) -> bool:
        return self.suit in ("H", "D")

    def code(self) -> str:
        return f"{self.rank}{self.suit}"


def hand_value(cards: list[Card]) -> tuple[int, bool]:
    """Best total and whether it is soft (an ace counted as 11)."""
    total = sum(card.points for card in cards)
    if any(card.rank == "A" for card in cards) and total + 10 <= 21:
        return total + 10, True
    return total, False


def is_blackjack(cards: list[Card]) -> bool:
    return len(cards) == 2 and hand_value(cards)[0] == 21


def describe(cards: list[Card]) -> str:
    total, soft = hand_value(cards)
    if is_blackjack(cards):
        return "BLACKJACK"
    if total > 21:
        return f"BUST {total}"
    return f"SOFT {total}" if soft and total < 21 else str(total)


def dealer_should_hit(cards: list[Card]) -> bool:
    return hand_value(cards)[0] < 17


def dealer_peeks(upcard: Card) -> bool:
    return upcard.rank == "A" or upcard.points == 10


class Shoe:
    def __init__(self, decks: int = 6, rng: random.Random | None = None) -> None:
        self.decks = decks
        self.rng = rng or random.SystemRandom()
        self.cards: list[Card] = []
        self.shuffle()

    def shuffle(self) -> None:
        self.cards = [Card(r, s) for _ in range(self.decks) for s in SUITS for r in RANKS]
        self.rng.shuffle(self.cards)

    def stack(self, cards: list[Card]) -> None:
        """Put these cards on top, dealt in this order (story mode)."""
        for card in reversed(cards):
            self.cards.append(card)

    def needs_shuffle(self) -> bool:
        return len(self.cards) < self.decks * 52 * 0.25

    def draw(self) -> Card:
        if not self.cards:
            self.shuffle()
        return self.cards.pop()


OUTCOME_TEXT = {
    "blackjack": "BLACKJACK!",
    "win": "FISH WIN",
    "dealer_bust": "DEALER BUSTS",
    "push": "PUSH",
    "lose": "DEALER WINS",
    "bust": "BUST",
    "dealer_blackjack": "DEALER BLACKJACK",
}


def settle(fish: list[Card], dealer: list[Card], bet: int) -> tuple[str, int]:
    """Outcome key and chip change for a finished hand. ``bet`` already includes any double."""
    fish_total = hand_value(fish)[0]
    dealer_total = hand_value(dealer)[0]
    fish_bj, dealer_bj = is_blackjack(fish), is_blackjack(dealer)
    if fish_bj and dealer_bj:
        return "push", 0
    if fish_bj:
        return "blackjack", (bet * 3) // 2
    if dealer_bj:
        return "dealer_blackjack", -bet
    if fish_total > 21:
        return "bust", -bet
    if dealer_total > 21:
        return "dealer_bust", bet
    if fish_total > dealer_total:
        return "win", bet
    if fish_total < dealer_total:
        return "lose", -bet
    return "push", 0


# Bankroll ------------------------------------------------------------------------
def load_bankroll(path: Path = BLACKJACK_PATH, start: int = 1000) -> dict:
    if path.exists():
        try:
            data = json.loads(path.read_text())
            if isinstance(data, dict) and isinstance(data.get("bankroll"), int):
                data.setdefault("hands", [])
                data.setdefault("start", start)
                return data
        except json.JSONDecodeError:
            pass
    return {"bankroll": start, "start": start, "hands": []}


def save_bankroll(data: dict, path: Path = BLACKJACK_PATH) -> None:
    atomic_write_text(path, json.dumps(data, indent=2) + "\n")


def record_hand(data: dict, fish: list[Card], dealer: list[Card], bet: int, decisions: list[str], outcome: str, delta: int) -> None:
    data["bankroll"] = int(data["bankroll"]) + delta
    data["hands"].append(
        {
            "timestamp": iso_timestamp(),
            "fish": [c.code() for c in fish],
            "dealer": [c.code() for c in dealer],
            "bet": bet,
            "decisions": decisions,
            "outcome": outcome,
            "delta": delta,
            "bankroll": data["bankroll"],
        }
    )
