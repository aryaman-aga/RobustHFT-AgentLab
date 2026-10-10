"""Transaction cost model: proportional fee + symmetric slippage in basis points."""
from __future__ import annotations


def fill_price(price: float, side: int, slippage_bps: float) -> float:
    """Buy (side>0) pays up, sell (side<0) receives less."""
    return price * (1 + side * slippage_bps / 1e4)


def fee(notional: float, fee_bps: float) -> float:
    return abs(notional) * fee_bps / 1e4
