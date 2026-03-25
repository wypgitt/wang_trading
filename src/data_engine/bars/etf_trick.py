"""
ETF Trick for Futures Roll (AFML Ch. 2)

Constructs a synthetic continuous futures series by tracking the dollar
value of a rolling investment, avoiding the problems of backward-adjusted
(Panama canal) methods that distort returns.

The standard approach of back-adjusting futures prices by the roll gap
creates a fictional price history where returns are correct but price
levels are wrong (and can go negative). The ETF trick instead models
what an investor would actually experience: rolling a fixed dollar
investment from the expiring contract to the next.

Usage:
    trick = ETFTrick()
    continuous_series = trick.compute(contracts_df)
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from loguru import logger


class ETFTrick:
    """
    AFML's ETF trick for constructing continuous futures series.

    Given a DataFrame of contract prices with roll dates, produces a
    continuous price series that preserves true returns without the
    distortions of backward adjustment.
    """

    @staticmethod
    def compute(
        prices: pd.DataFrame,
        roll_dates: list[pd.Timestamp],
        initial_value: float = 100.0,
    ) -> pd.Series:
        """
        Compute the ETF trick continuous series.

        Args:
            prices:     DataFrame with columns for each contract (e.g., 'ESH24', 'ESM24').
                        Index is datetime. Each column is the price of one contract.
            roll_dates: List of timestamps when the position rolls to the next contract.
            initial_value: Starting value of the synthetic ETF (default: 100).

        Returns:
            pd.Series: Continuous price series representing the ETF value.
        """
        if prices.empty or len(prices.columns) < 2:
            raise ValueError("Need at least 2 contract columns")

        contracts = list(prices.columns)
        result = pd.Series(index=prices.index, dtype=float)

        # Start with initial_value invested in the first contract
        current_contract_idx = 0
        current_contract = contracts[current_contract_idx]
        etf_value = initial_value

        # Number of units held (etf_value / price at start)
        first_valid = prices[current_contract].first_valid_index()
        if first_valid is None:
            raise ValueError(f"No valid prices for {current_contract}")

        units = etf_value / prices.loc[first_valid, current_contract]

        roll_dates_set = set(roll_dates)
        for dt in prices.index:
            # Check if we should roll
            if (
                dt in roll_dates_set
                and current_contract_idx + 1 < len(contracts)
            ):
                # Value at roll = units * price of old contract at roll date
                old_price = prices.loc[dt, current_contract]
                if not np.isnan(old_price) and old_price > 0:
                    etf_value = units * old_price

                    # Roll to next contract
                    current_contract_idx += 1
                    current_contract = contracts[current_contract_idx]
                    new_price = prices.loc[dt, current_contract]

                    if not np.isnan(new_price) and new_price > 0:
                        units = etf_value / new_price
                        logger.debug(
                            f"Roll at {dt}: {contracts[current_contract_idx-1]} → "
                            f"{current_contract}, ETF value: {etf_value:.2f}"
                        )

            # Current ETF value
            price = prices.loc[dt, current_contract]
            if not np.isnan(price):
                etf_value = units * price
                result.loc[dt] = etf_value

        return result.dropna()

    @staticmethod
    def detect_roll_dates(
        front_oi: pd.Series,
        back_oi: pd.Series,
    ) -> list[pd.Timestamp]:
        """
        Detect roll dates by open interest crossover.

        The roll date is when the back contract's open interest exceeds
        the front contract's, indicating that the market has shifted
        to the next contract.

        Args:
            front_oi: Open interest for the front (expiring) contract
            back_oi:  Open interest for the back (next) contract

        Returns:
            List of timestamps where rolls should occur
        """
        crossover = (back_oi > front_oi) & (back_oi.shift(1) <= front_oi.shift(1))
        return list(crossover[crossover].index)
