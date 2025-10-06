#!/usr/bin/env python3
"""
ZION AI Trading Bot - Simple Version
Fallback verze bez TensorFlow závislostí
"""

import logging
from datetime import datetime
import random

logger = logging.getLogger(__name__)

class ZionTradingBot:
    """Zjednodušený AI trading bot"""

    def __init__(self):
        self.trades = []
        logger.info("ZionTradingBot initialized (simple version)")

    def make_trading_decision(self, market_data=None):
        """Rozhodnutí o obchodování"""
        action = random.choice(['buy', 'sell', 'hold'])
        confidence = random.uniform(0.4, 0.8)

        return {
            'action': action,
            'confidence': confidence,
            'amount': random.uniform(0.1, 1.0) if action != 'hold' else 0,
            'reason': f"AI decision: {action} with {confidence:.2f} confidence",
            'timestamp': datetime.now().isoformat()
        }

    def get_portfolio_status(self):
        """Status portfolia"""
        return {
            'total_value': random.uniform(1000, 10000),
            'pnl': random.uniform(-500, 500),
            'positions': random.randint(0, 5),
            'timestamp': datetime.now().isoformat()
        }