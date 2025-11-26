"""
Bayesian modeling module for Texas Mushrooms.
Uses PyMC to model species presence/abundance.
"""
import pymc as pm
import pandas as pd
import arviz as az
from typing import Optional, List, Dict, Any


class BayesianMushroomModel:
    def __init__(self, data: pd.DataFrame, target_col: str = "count"):
        """
        Initialize the Bayesian model.

        Args:
            data: DataFrame containing predictors and target.
            target_col: Name of the target column (e.g., mushroom count).
        """
        self.data = data
        self.target_col = target_col
        self.model: Optional[pm.Model] = None
        self.trace: Optional[az.InferenceData] = None

    def build_poisson_model(
        self, predictors: List[str], coords: Optional[Dict[str, List[Any]]] = None
    ) -> None:
        """
        Build a Poisson regression model.

        y ~ Poisson(exp(alpha + beta * X))
        """
        with pm.Model(coords=coords) as self.model:
            # Data containers
            try:
                X = pm.Data("X", self.data[predictors].values, mutable=True)
                y = pm.Data("y", self.data[self.target_col].values, mutable=True)
            except (AttributeError, TypeError):
                X = pm.Data("X", self.data[predictors].values)
                y = pm.Data("y", self.data[self.target_col].values)

            # Priors
            alpha = pm.Normal("alpha", mu=0, sigma=1)
            betas = pm.Normal("betas", mu=0, sigma=1, shape=len(predictors))

            # Linear predictor
            mu = pm.math.exp(alpha + pm.math.dot(X, betas))

            # Likelihood
            pm.Poisson("obs", mu=mu, observed=y)

    def build_zip_model(self, predictors: List[str]) -> None:
        """
        Build a Zero-Inflated Poisson (ZIP) model.
        Useful for mushroom data which has many zeros (days with no mushrooms).
        """
        with pm.Model() as self.model:
            # Data
            # Note: pm.MutableData was renamed/moved in some versions or might be pm.Data
            try:
                X = pm.Data("X", self.data[predictors].values, mutable=True)
                y = pm.Data("y", self.data[self.target_col].values, mutable=True)
            except (AttributeError, TypeError):
                # Fallback for older PyMC versions or different APIs
                X = pm.Data("X", self.data[predictors].values)
                y = pm.Data("y", self.data[self.target_col].values)

            # Priors for Poisson component
            alpha = pm.Normal("alpha", mu=0, sigma=1)
            betas = pm.Normal("betas", mu=0, sigma=1, shape=len(predictors))

            # Prior for Zero-inflation probability (psi)
            psi = pm.Beta("psi", alpha=1, beta=1)

            # Linear predictor
            mu = pm.math.exp(alpha + pm.math.dot(X, betas))

            # Likelihood
            pm.ZeroInflatedPoisson("obs", psi=psi, mu=mu, observed=y)

    def sample(
        self, draws: int = 1000, tune: int = 1000, chains: int = 2, cores: int = 1
    ) -> None:
        """
        Sample from the posterior.
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model first.")

        with self.model:
            self.trace = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores)

    def plot_trace(self) -> None:
        """
        Plot the trace.
        """
        if self.trace is None:
            raise ValueError("Model not sampled yet.")
        az.plot_trace(self.trace)

    def predict(self, new_data: pd.DataFrame, predictors: List[str]) -> Any:
        """
        Generate posterior predictive samples for new data.
        """
        if self.model is None or self.trace is None:
            raise ValueError("Model must be built and sampled.")

        with self.model:
            pm.set_data({"X": new_data[predictors].values})
            ppc = pm.sample_posterior_predictive(self.trace)

        return ppc
