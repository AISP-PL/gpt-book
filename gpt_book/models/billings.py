"""
 Dataclass for storing billings informations for every openai model api call.
 - Data default is being stored inside local json file in data/billings.json
 - Dictionary is stored : calls counter per gpt-model name.
 - Cost is being stored as dictionary : 1 call cost per gpt-model name.
 - Property methods added for gettin cost (price x calls).
"""

import json
from dataclasses import asdict, dataclass, field
from typing import Dict

# Path: gpt_book/models/billings.py


@dataclass
class Billings:
    """
    Dataclass for storing billings informations for every openai model api.
    """

    # Path: gpt_book/models/billings.py
    api_calls: Dict[str, int] = field(default_factory=dict)
    # Cost per api cal
    costs: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Load data from json file."""
        self.load()

    def cost(self, model: str) -> float:
        """All api calls costs for model."""
        # Check if model exists
        if (model not in self.api_calls) or (model not in self.costs):
            return 0.0

        return self.costs[model] * self.api_calls[model]

    @property
    def total_cost(self) -> float:
        """Total cost for all models."""
        return sum(self.cost(model) for model in self.api_calls)

    def add_api_call(self, model: str):
        """Add call to model."""
        # Model : Create if not exists
        if model in self.api_calls:
            self.api_calls[model] += 1
        else:
            self.api_calls[model] = 1

        # Save ; Every call
        self.save()

    def load(self, path: str = "data/billings.json"):
        """Load data from json file."""
        try:
            with open(path, "r") as file:
                data = json.load(file)
                self.api_calls = data["api_calls"]
                self.costs = data["costs"]

        except FileNotFoundError:
            pass

    def save(self, path: str = "data/billings.json"):
        """Save data to json file."""
        with open(path, "w") as file:
            data_dict = asdict(self)
            json.dump(data_dict, file, indent=4)
