
import numpy as np
import pandas as pd
from collections import deque
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_curve, auc
import warnings
from typing import Dict, List, Tuple, Optional
warnings.filterwarnings('ignore')

class RealtimeFatigueMonitor:
    """
    Real-time fatigue monitoring system with validation and practical applications.
    """
    def __init__(
        self,
        window_size: int = 10,
        buffer_size: int = 50,
        threshold_sensitivity: float = 0.8,
        anomaly_contamination: float = 0.1
    ):
        self.shot_buffer = deque(maxlen=buffer_size)
        self.score_buffer = deque(maxlen=buffer_size)
        self.window_size = window_size
        self.threshold_sensitivity = threshold_sensitivity
        self.anomaly_detector = IsolationForest(
            contamination=anomaly_contamination,
            random_state=42
        )
        self.performance_metrics = {
            'shot_percentage': deque(maxlen=buffer_size),
            'coach_ratings': deque(maxlen=buffer_size),
            'correlation_scores': deque(maxlen=buffer_size)
        }
        self.fatigue_thresholds = {'low': 30, 'moderate': 60, 'high': 80}
        self.confidence_scores = deque(maxlen=buffer_size)

    def process_shot(
        self,
        shot_data: Dict,
        coach_rating: Optional[float] = None
    ) -> Dict:
        features = self._extract_features_optimized(shot_data)
        self.shot_buffer.append(features)
        fatigue_score, confidence = self._calculate_windowed_fatigue_score()
        self.score_buffer.append(fatigue_score)
        self.confidence_scores.append(confidence)
        is_anomaly = self._detect_anomalies(features)

        if coach_rating is not None:
            self._update_validation_metrics(fatigue_score, coach_rating)

        alerts = self._generate_alerts(fatigue_score)
        return {
            'fatigue_score': fatigue_score,
            'confidence': confidence,
            'alerts': alerts,
            'is_anomaly': is_anomaly,
            'trend': self._calculate_trend()
        }

    def _extract_features_optimized(self, shot_data: Dict) -> Dict:
        tracking_data = shot_data.get("tracking", [])
        if not tracking_data:
            raise ValueError("No tracking data found in shot_data.")

        positions = []
        for frame in tracking_data:
            ball_data = frame.get("data", {}).get("ball", [np.nan, np.nan, np.nan])
            if not isinstance(ball_data, list) or len(ball_data) != 3:
                ball_data = [np.nan, np.nan, np.nan]  # Handle invalid data
            positions.append(ball_data)

        positions = np.array(positions)
        positions = np.nan_to_num(positions, nan=0.0)  # Replace NaN with zeros

        velocities = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        accelerations = np.diff(velocities)
        jerk = np.diff(accelerations)

        return {
            'mean_velocity': np.mean(velocities),
            'velocity_std': np.std(velocities),
            'mean_jerk': np.mean(np.abs(jerk)),
            'smoothness': 1 / (1 + np.std(jerk)),
            'shot_duration': len(tracking_data),
            'max_height': np.max(positions[:, 2]),
            'release_angle': self._calculate_release_angle(positions)
        }

    def _calculate_windowed_fatigue_score(self) -> Tuple[float, float]:
        if len(self.shot_buffer) < self.window_size:
            return 50.0, 0.5

        recent_shots = list(self.shot_buffer)[-self.window_size:]
        weights = np.exp(np.linspace(-1, 0, len(recent_shots)))
        weights /= weights.sum()

        weighted_scores = []
        for shot, weight in zip(recent_shots, weights):
            score = (
                0.3 * shot['smoothness'] +
                0.3 * (1 - shot['velocity_std']) +
                0.4 * (1 - shot['mean_jerk'])
            )
            weighted_scores.append(score * weight)

        confidence = 1 / (1 + np.std(weighted_scores))
        fatigue_score = 100 * (1 - np.sum(weighted_scores))
        return np.clip(fatigue_score, 0, 100), confidence

    def _detect_anomalies(self, features: Dict) -> bool:
        if len(self.shot_buffer) < 10:
            return False

        feature_matrix = np.array([
            [shot['mean_velocity'], shot['smoothness'], shot['mean_jerk']]
            for shot in self.shot_buffer
        ])
        self.anomaly_detector.fit(feature_matrix)
        current_features = np.array([[features['mean_velocity'], features['smoothness'], features['mean_jerk']]])
        return self.anomaly_detector.predict(current_features)[0] == -1

    def _update_validation_metrics(self, fatigue_score: float, coach_rating: float) -> None:
        self.performance_metrics['coach_ratings'].append(coach_rating)
        if len(self.performance_metrics['coach_ratings']) >= 3:
            correlation = stats.pearsonr(
                list(self.score_buffer)[-len(self.performance_metrics['coach_ratings']):],
                list(self.performance_metrics['coach_ratings'])
            )[0]
            self.performance_metrics['correlation_scores'].append(correlation)

    def _calculate_trend(self) -> Dict:
        if len(self.score_buffer) < 3:
            return {'slope': 0, 'r_squared': 0}

        x = np.arange(len(self.score_buffer))
        y = np.array(list(self.score_buffer))
        slope, intercept = np.polyfit(x, y, 1)
        r_squared = np.corrcoef(x, y)[0, 1]**2

        return {'slope': slope, 'r_squared': r_squared}

    def _generate_alerts(self, fatigue_score: float) -> List[str]:
        alerts = []
        if fatigue_score > self.fatigue_thresholds['high']:
            alerts.append("HIGH FATIGUE: Consider resting player")
        elif fatigue_score > self.fatigue_thresholds['moderate']:
            alerts.append("MODERATE FATIGUE: Monitor closely")

        if len(self.score_buffer) >= 3:
            recent_trend = self._calculate_trend()
            if recent_trend['slope'] > 5 and recent_trend['r_squared'] > 0.7:
                alerts.append("RAPID FATIGUE INCREASE DETECTED")

        return alerts

    def get_validation_metrics(self) -> Dict:
        return {
            'coach_correlation': np.mean(self.performance_metrics['correlation_scores'])
                if self.performance_metrics['correlation_scores'] else None,
            'confidence_mean': np.mean(self.confidence_scores),
            'anomaly_rate': sum(1 for s in self.shot_buffer if self._detect_anomalies(s)) /
                          len(self.shot_buffer) if self.shot_buffer else 0
        }

    def update_thresholds(self, new_thresholds: Dict) -> None:
        self.fatigue_thresholds.update(new_thresholds)

    def _calculate_release_angle(self, positions: np.ndarray) -> float:
        release_idx = np.argmax(np.diff(positions[:, 2]) > 0)
        if release_idx + 3 >= len(positions):
            return 0

        release_vector = positions[release_idx + 3] - positions[release_idx]
        return np.degrees(np.arctan2(release_vector[2], np.linalg.norm(release_vector[:2])))
