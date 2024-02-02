import numpy as np


class PowerSystem():
    """
    Class to create a Power System scenario.
    """

    def __init__(self,
                 hours=24,
                 min_players=10
                 ):

        """
        Attributes:

        hours: Number of hours during which trading is possible.
        min_players: Minimum number of hourly transactions.
        max_players: Maximum number of hourly transactions.
        min_price: Minimum energy price in cents/KWh.
        max_price: Maximum energy price in cents/KWh.
        min_bid: Minimum value of the energy bid range.
        max_bid: Maximum value of the energy bid range.
        min_ask: Minimum value of the energy ask range.
        max_ask: Maximum value of the energy ask range.
        """

        self.hours = hours
        self.min_players = min_players
        self.max_players = max_players
        self.min_price = min_price
        self.max_price = max_price
        self.min_bid = min_bid
        self.max_bid = max_bid
        self.min_ask = max_bid
        self.max_ask = 2*max_bid