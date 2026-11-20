"""Broker-agnostic order math: target weights + current book -> dollar orders."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Order:
    symbol: str
    side: str          # "buy" | "sell"
    notional: float    # USD, positive
    close_all: bool = False

    def __str__(self) -> str:
        tag = " (close position)" if self.close_all else ""
        return f"{self.side.upper():4s} {self.symbol:6s} ${self.notional:>12,.2f}{tag}"


def plan_orders(
    target_weights: dict[str, float],
    positions_usd: dict[str, float],
    equity: float,
    min_trade_usd: float = 50.0,
) -> list[Order]:
    """Sells first (frees cash), then buys. Symbols not in target are closed."""
    symbols = set(target_weights) | set(positions_usd)
    orders = []
    for s in sorted(symbols):
        tgt = target_weights.get(s, 0.0) * equity
        cur = positions_usd.get(s, 0.0)
        delta = tgt - cur
        if abs(delta) < min_trade_usd:
            continue
        if delta < 0:
            orders.append(Order(s, "sell", -delta, close_all=target_weights.get(s, 0.0) == 0.0))
        else:
            orders.append(Order(s, "buy", delta))
    return sorted(orders, key=lambda o: o.side != "sell")
