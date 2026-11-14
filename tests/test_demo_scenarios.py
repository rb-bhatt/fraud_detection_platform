import json
from pathlib import Path

import pandas as pd

from api.model_loader import feature_name, model


SCENARIOS_PATH = (
    Path(__file__).parents[1] / "dashboard" / "demo-scenarios.json"
)


def test_demo_scenarios_cover_all_decisions():
    scenarios = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))
    frame = pd.DataFrame(
        [scenario["payload"] for scenario in scenarios.values()]
    )[feature_name]
    probabilities = model.predict_proba(frame)[:, 1]

    decisions = {
        "Approved" if probability < 0.6
        else "Review" if probability < 0.8
        else "Block"
        for probability in probabilities
    }

    assert decisions == {"Approved", "Review", "Block"}
