import os
import json
from tryy import RealtimeFatigueMonitor

def test_fatigue_monitor(directory_path):
    # Initialize the RealtimeFatigueMonitor
    fatigue_monitor = RealtimeFatigueMonitor()

    # Iterate through all JSON files in the directory
    
    json_files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
    for file_name in json_files:
            with open(os.path.join(directory_path, file_name), 'r') as f:
                shot_data = json.load(f)
                result = fatigue_monitor.process_shot(shot_data)     
            # Print results for each shot
            print(f"File: {file_name}")
            print(f"Fatigue Score: {result['fatigue_score']:.2f}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Alerts: {', '.join(result['alerts'])}")
            print(f"Is Anomaly: {result['is_anomaly']}")
            print(f"Trend: Slope = {result['trend']['slope']:.2f}, R-squared = {result['trend']['r_squared']:.2f}")
            print("---")

    # Print overall validation metrics
    validation_metrics = fatigue_monitor.get_validation_metrics()
    print("Validation Metrics:")
    print(f"Coach Correlation: {validation_metrics['coach_correlation']}")
    print(f"Confidence Mean: {validation_metrics['confidence_mean']:.2f}")
    print(f"Anomaly Rate: {validation_metrics['anomaly_rate']:.2f}")

# Usage
test_fatigue_monitor("data/P0001")
