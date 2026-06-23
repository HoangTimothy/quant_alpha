"""Model factory — instantiate any model by name."""

from __future__ import annotations

from .base_model import BaseModel
from .logistic_model import LogisticModel
from .random_forest_model import RandomForestModel
from .xgboost_model import XGBoostModel
from .lightgbm_model import LightGBMModel
from .mlp_model import MLPModel
from .tcnn_model import TemporalCNNModel
from .transformer_model import TransformerModel
from .tft_model import TFTModel

MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "logistic": LogisticModel,
    "random_forest": RandomForestModel,
    "xgboost": XGBoostModel,
    "lightgbm": LightGBMModel,
    "mlp": MLPModel,
    "tcnn": TemporalCNNModel,
    "transformer": TransformerModel,
    "tft": TFTModel,
}


def create_model(name: str, params: dict | None = None) -> BaseModel:
    """Create a model instance by name.

    Parameters
    ----------
    name : str
        Model name (must be a key in MODEL_REGISTRY).
    params : dict, optional
        Model-specific hyperparameters.

    Returns
    -------
    BaseModel
        Instantiated (unfitted) model.

    Raises
    ------
    ValueError
        If the model name is not registered.
    """
    name = name.lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](params=params)


def list_models() -> list[str]:
    """Return all registered model names."""
    return list(MODEL_REGISTRY.keys())
