"""Alpaca execution (paper by default). Requires `alpaca-py` and keys in .env.

Safety rails:
  - paper=True unless settings say otherwise AND you pass --live on the command line
  - dry_run=True prints the plan and sends nothing
"""
from __future__ import annotations

import logging

from .orders import Order, plan_orders

log = logging.getLogger(__name__)


class AlpacaBroker:
    def __init__(self, api_key: str | None, secret: str | None, paper: bool = True):
        if not api_key or not secret:
            raise RuntimeError("ALPACA_API_KEY / ALPACA_SECRET_KEY missing — add them to .env")
        from alpaca.trading.client import TradingClient

        self.paper = paper
        self.client = TradingClient(api_key, secret, paper=paper)

    def equity(self) -> float:
        return float(self.client.get_account().equity)

    def positions_usd(self) -> dict[str, float]:
        return {p.symbol: float(p.market_value) for p in self.client.get_all_positions()}

    def market_open_today(self) -> bool:
        return bool(self.client.get_clock().is_open)

    def submit(self, orders: list[Order]) -> list[str]:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        ids = []
        for o in orders:
            if o.close_all:
                resp = self.client.close_position(o.symbol)
            else:
                req = MarketOrderRequest(
                    symbol=o.symbol,
                    notional=round(o.notional, 2),
                    side=OrderSide.BUY if o.side == "buy" else OrderSide.SELL,
                    time_in_force=TimeInForce.DAY,
                )
                resp = self.client.submit_order(req)
            ids.append(str(getattr(resp, "id", resp)))
            log.info("sent %s", o)
        return ids

    def rebalance(self, target_weights: dict[str, float], min_trade_usd: float = 50.0, dry_run: bool = True):
        eq = self.equity()
        orders = plan_orders(target_weights, self.positions_usd(), eq, min_trade_usd)
        mode = "PAPER" if self.paper else "LIVE"
        print(f"[{mode}] equity ${eq:,.2f} — {len(orders)} order(s)")
        for o in orders:
            print("   ", o)
        if dry_run:
            print("dry_run=true — nothing sent. Set execution.dry_run: false to trade.")
            return orders, []
        return orders, self.submit(orders)
