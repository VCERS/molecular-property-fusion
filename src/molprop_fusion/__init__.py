"""Prediction-level fusion for continuous molecular-property regression."""

__version__ = "0.2.0"

from molprop_fusion.estimator import MolecularPropertyFusionRegressor, load_model

__all__ = ["MolecularPropertyFusionRegressor", "load_model"]
