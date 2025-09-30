#!/usr/bin/env python3
"""
ZION 2.6.75 Oracle Network AI
Decentralized Data Feeds, Multi-Source Consensus, Anomaly Detection & Predictive Intelligence
"""
import asyncio
import random
import time
import json
import hashlib
import statistics
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
import logging
from datetime import datetime

try:
    import numpy as np  # type: ignore
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import pandas as pd  # type: ignore
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from sklearn.ensemble import IsolationForest  # type: ignore
    from sklearn.linear_model import LinearRegression  # type: ignore
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class DataFeedType(Enum):
    PRICE = "price"
    WEATHER = "weather"
    HASHRATE = "hashrate"
    LATENCY = "latency"
    ENERGY = "energy"
    CUSTOM = "custom"


class ConsensusMethod(Enum):
    MEDIAN = "median"
    WEIGHTED = "weighted"
    TRUST_SCORE = "trust_score"
    TIME_WEIGHTED = "time_weighted"


@dataclass
class FeedSource:
    source_id: str
    name: str
    reliability: float  # 0-1 static weight
    trust_score: float = 0.5  # dynamic
    latency_ms: float = 50.0
    last_value: Optional[float] = None
    last_update: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OracleFeed:
    feed_id: str
    feed_type: DataFeedType
    symbol: str
    consensus: ConsensusMethod
    sources: Dict[str, FeedSource] = field(default_factory=dict)
    values_history: List[Dict[str, Any]] = field(default_factory=list)
    last_consensus_value: Optional[float] = None
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnomalyEvent:
    event_id: str
    feed_id: str
    timestamp: float
    severity: float
    description: str
    raw_values: Dict[str, float]
    consensus_value: Optional[float]


@dataclass
class PredictionModel:
    model_id: str
    feed_id: str
    model_type: str
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ZionOracleAI:
    """ZION Oracle Network AI"""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        self.feeds: Dict[str, OracleFeed] = {}
        self.anomalies: List[AnomalyEvent] = []
        self.prediction_models: Dict[str, PredictionModel] = {}
        self._isolation_models: Dict[str, Any] = {}
        self._regression_models: Dict[str, Any] = {}
        self.logger.info("🔮 ZionOracleAI initialized")

    # ------------------------- FEED MANAGEMENT -----------------------------
    async def create_feed(self, feed_type: DataFeedType, symbol: str, consensus: ConsensusMethod = ConsensusMethod.MEDIAN) -> Dict[str, Any]:
        try:
            feed_id = self._generate_id('feed')
            feed = OracleFeed(
                feed_id=feed_id,
                feed_type=feed_type,
                symbol=symbol,
                consensus=consensus,
                metadata={'created_at_iso': datetime.utcnow().isoformat()}
            )
            self.feeds[feed_id] = feed
            return {'success': True, 'feed_id': feed_id}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def add_source(self, feed_id: str, name: str, reliability: float) -> Dict[str, Any]:
        if feed_id not in self.feeds:
            return {'success': False, 'error': 'Feed not found'}
        source_id = self._generate_id('src')
        self.feeds[feed_id].sources[source_id] = FeedSource(
            source_id=source_id,
            name=name,
            reliability=max(0.0, min(1.0, reliability)),
            trust_score=0.5 + (reliability*0.5)
        )
        return {'success': True, 'source_id': source_id}

    # ------------------------- DATA INGESTION ------------------------------
    async def submit_value(self, feed_id: str, source_id: str, value: float) -> Dict[str, Any]:
        if feed_id not in self.feeds:
            return {'success': False, 'error': 'Feed not found'}
        feed = self.feeds[feed_id]
        if source_id not in feed.sources:
            return {'success': False, 'error': 'Source not found'}
        src = feed.sources[source_id]
        now = time.time()
        src.last_value = value
        src.last_update = now
        # synthetic latency
        src.latency_ms = random.uniform(20, 200)
        feed.values_history.append({
            'timestamp': now,
            'source_id': source_id,
            'value': value
        })
        # Trigger consensus if enough submissions recently
        if self._should_update_consensus(feed):
            cval = self._compute_consensus(feed)
            feed.last_consensus_value = cval
            self._update_trust_scores(feed, cval)
            self._detect_anomaly(feed, cval)
        return {'success': True, 'consensus': feed.last_consensus_value}

    def _should_update_consensus(self, feed: OracleFeed) -> bool:
        if len(feed.values_history) < max(3, len(feed.sources)//2):
            return False
        # update if last values newer than last consensus timestamp
        return True

    def _compute_consensus(self, feed: OracleFeed) -> Optional[float]:
        latest_values = {}
        for src_id, src in feed.sources.items():
            if src.last_value is not None:
                latest_values[src_id] = src.last_value
        if not latest_values:
            return None
        vals = list(latest_values.values())
        if feed.consensus == ConsensusMethod.MEDIAN:
            vals_sorted = sorted(vals)
            mid = len(vals_sorted)//2
            if len(vals_sorted)%2==1:
                return vals_sorted[mid]
            else:
                return (vals_sorted[mid-1]+vals_sorted[mid])/2
        elif feed.consensus == ConsensusMethod.WEIGHTED:
            total_w = 0
            accum = 0
            for sid,v in latest_values.items():
                w = feed.sources[sid].reliability
                total_w += w
                accum += v*w
            return accum/total_w if total_w>0 else statistics.mean(vals)
        elif feed.consensus == ConsensusMethod.TRUST_SCORE:
            total_w = 0
            accum = 0
            for sid,v in latest_values.items():
                w = feed.sources[sid].trust_score
                total_w += w
                accum += v*w
            return accum/total_w if total_w>0 else statistics.mean(vals)
        elif feed.consensus == ConsensusMethod.TIME_WEIGHTED:
            now = time.time()
            total_w = 0
            accum = 0
            for sid, v in latest_values.items():
                age = now - feed.sources[sid].last_update
                w = max(0.1, 1.0/(1.0+age))
                total_w += w
                accum += v*w
            return accum/total_w if total_w>0 else statistics.mean(vals)
        return statistics.mean(vals)

    def _update_trust_scores(self, feed: OracleFeed, consensus_value: Optional[float]):
        if consensus_value is None:
            return
        for src in feed.sources.values():
            if src.last_value is None:
                continue
            deviation = abs(src.last_value - consensus_value)
            # dynamic adjustment
            adjust = -0.02 if deviation > (abs(consensus_value)*0.05 + 0.0001) else 0.01
            src.trust_score = min(1.0, max(0.0, src.trust_score + adjust))

    # ------------------------- ANOMALY DETECTION --------------------------
    def _detect_anomaly(self, feed: OracleFeed, consensus_value: Optional[float]):
        if consensus_value is None:
            return
        latest_values = {sid: s.last_value for sid,s in feed.sources.items() if s.last_value is not None}
        if len(latest_values) < 3:
            return
        values_array = list(latest_values.values())
        mean_v = statistics.mean(values_array)
        stdev_v = statistics.pstdev(values_array) if len(values_array)>1 else 0
        # simple rule-based outlier
        if stdev_v > 0 and any(abs(v-mean_v) > 3*stdev_v for v in values_array):
            event = AnomalyEvent(
                event_id=self._generate_id('anom'),
                feed_id=feed.feed_id,
                timestamp=time.time(),
                severity=0.7,
                description='Standard deviation outlier cluster detected',
                raw_values={sid: float(v) for sid,v in latest_values.items()},
                consensus_value=consensus_value
            )
            self.anomalies.append(event)
        # ML-based (if available) - IsolationForest
        if SKLEARN_AVAILABLE and len(values_array) >= 5:
            try:
                import numpy as np  # type: ignore
                arr = np.array(values_array).reshape(-1,1)
                model = self._isolation_models.get(feed.feed_id)
                if model is None:
                    model = IsolationForest(n_estimators=50, contamination=0.2)
                    model.fit(arr)
                    self._isolation_models[feed.feed_id] = model
                preds = model.predict(arr)
                if any(p==-1 for p in preds):
                    event = AnomalyEvent(
                        event_id=self._generate_id('anom'),
                        feed_id=feed.feed_id,
                        timestamp=time.time(),
                        severity=0.5,
                        description='IsolationForest anomaly pattern detected',
                        raw_values={sid: float(v) for sid,v in latest_values.items()},
                        consensus_value=consensus_value
                    )
                    self.anomalies.append(event)
            except Exception as e:
                self.logger.warning(f"Isolation anomaly detection failed: {e}")

    # ------------------------- PREDICTION ---------------------------------
    async def build_prediction_model(self, feed_id: str) -> Dict[str, Any]:
        if feed_id not in self.feeds:
            return {'success': False, 'error': 'Feed not found'}
        feed = self.feeds[feed_id]
        if len(feed.values_history) < 10:
            return {'success': False, 'error': 'Insufficient history'}
        if not SKLEARN_AVAILABLE:
            return {'success': False, 'error': 'sklearn not available'}
        try:
            import numpy as np  # type: ignore
            data = feed.values_history[-200:]
            times = np.array([d['timestamp'] for d in data])
            values = np.array([d['value'] for d in data])
            # normalize time
            t0 = times[0]
            X = (times - t0).reshape(-1,1)
            model = LinearRegression()
            model.fit(X, values)
            self._regression_models[feed_id] = model
            model_id = self._generate_id('pred')
            pmodel = PredictionModel(
                model_id=model_id,
                feed_id=feed_id,
                model_type='linear_regression',
                metadata={'samples': len(data)}
            )
            self.prediction_models[model_id] = pmodel
            return {'success': True, 'model_id': model_id}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    async def predict_value(self, feed_id: str, horizon_seconds: float) -> Dict[str, Any]:
        if feed_id not in self.feeds:
            return {'success': False, 'error': 'Feed not found'}
        model = self._regression_models.get(feed_id)
        feed = self.feeds[feed_id]
        if not model:
            return {'success': False, 'error': 'No model'}
        try:
            import numpy as np  # type: ignore
            if not feed.values_history:
                return {'success': False, 'error': 'No history'}
            last_time = feed.values_history[-1]['timestamp']
            t0 = feed.values_history[0]['timestamp']
            X_pred = (last_time + horizon_seconds - t0)
            pred = model.predict([[X_pred]])[0]
            return {'success': True, 'predicted_value': float(pred), 'horizon': horizon_seconds}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    # ------------------------- ANALYTICS ----------------------------------
    async def get_oracle_analytics(self) -> Dict[str, Any]:
        try:
            total_feeds = len(self.feeds)
            total_sources = sum(len(f.sources) for f in self.feeds.values())
            feed_types = {}
            consensus_methods = {}
            for f in self.feeds.values():
                feed_types[f.feed_type.value] = feed_types.get(f.feed_type.value,0)+1
                consensus_methods[f.consensus.value] = consensus_methods.get(f.consensus.value,0)+1
            anomalies_recent = [a for a in self.anomalies if time.time()-a.timestamp < 3600]
            return {
                'success': True,
                'feeds': total_feeds,
                'sources': total_sources,
                'feed_types': feed_types,
                'consensus_methods': consensus_methods,
                'anomalies_total': len(self.anomalies),
                'anomalies_last_hour': len(anomalies_recent),
                'prediction_models': len(self.prediction_models)
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    # ------------------------- EXPORT -------------------------------------
    async def export_feed(self, feed_id: str) -> Dict[str, Any]:
        if feed_id not in self.feeds:
            return {'success': False, 'error': 'Feed not found'}
        feed = self.feeds[feed_id]
        return {'success': True, 'data': asdict(feed)}

    # ------------------------- UTILITY ------------------------------------
    def _generate_id(self, prefix: str) -> str:
        return f"{prefix}_{random.getrandbits(48):012x}"

    # ------------------------- DEMO ---------------------------------------
    async def demo(self):
        print("🔮 ZION Oracle AI Demo")
        create = await self.create_feed(DataFeedType.PRICE, symbol="ZIONUSD", consensus=ConsensusMethod.TRUST_SCORE)
        if not create['success']:
            print(create)
            return
        fid = create['feed_id']
        for name in ["ExchangeA","ExchangeB","DEX1","DEX2","Aggregator"]:
            await self.add_source(fid, name=name, reliability=random.uniform(0.6,0.95))
        base_price = 1.23
        for i in range(30):
            tasks = []
            for sid in list(self.feeds[fid].sources.keys()):
                noise = random.uniform(-0.01,0.01)
                if random.random()<0.05:
                    noise += random.uniform(0.05,0.1)  # occasional anomaly
                tasks.append(self.submit_value(fid, sid, base_price+noise))
            await asyncio.gather(*tasks)
            await asyncio.sleep(0.01)
        if SKLEARN_AVAILABLE:
            model_res = await self.build_prediction_model(fid)
            print("Model build:", model_res)
            if model_res['success']:
                prediction = await self.predict_value(fid, horizon_seconds=300)
                print("Prediction:", prediction)
        analytics = await self.get_oracle_analytics()
        print("Analytics:", analytics)
        print("Anomalies detected:", len(self.anomalies))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    oracle_ai = ZionOracleAI()
    asyncio.run(oracle_ai.demo())
