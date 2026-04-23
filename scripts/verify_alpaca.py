"""Standalone smoke test: load env, build paper client, print account."""

from __future__ import annotations

from aitrade.brokers.alpaca import build_client


def main() -> None:
    broker = build_client()
    acct = broker.get_account()
    print(f"Alpaca {'PAPER' if acct.is_paper else 'LIVE'} account {acct.account_id}")
    print(f"  cash         : {acct.cash:,.2f} {acct.currency}")
    print(f"  equity       : {acct.equity:,.2f} {acct.currency}")
    print(f"  buying_power : {acct.buying_power:,.2f} {acct.currency}")


if __name__ == "__main__":
    main()
