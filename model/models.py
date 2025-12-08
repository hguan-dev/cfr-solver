import logging
from abc import ABC, abstractmethod

import numpy as np
import polars as pl
import sklearn
from constants import RANDOM_SEED

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)


class ModelBase(ABC):
    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def fit(self, x: pl.DataFrame, y: pl.DataFrame) -> None:
        pass

    @abstractmethod
    def predict(self, x: pl.DataFrame) -> np.typing.NDArray:
        pass


class MarginalProbabilityModel(ModelBase):
    """
    Compute marginal probability as a baseline estimate.
    """

    def __init__(self):
        self.name = "Marginal Probability Baseline"

    def get_name(self) -> str:
        return self.name

    def fit(self, x: pl.DataFrame, y: pl.DataFrame) -> None:
        logger.info(f"Training {self.name}...")
        assert x is not None  # x is not used
        self.marginal_probs = y.to_numpy().mean(axis=0)

    def predict(self, x: pl.DataFrame) -> np.ndarray:
        n_samples = x.height
        return np.tile(self.marginal_probs, (n_samples, 1))


class UniformProbabilityModel(ModelBase):
    """
    Compute uniform distribution as a baseline estimate.
    """

    def __init__(self, n_classes=3):
        self.name = "Uniform Probability Baseline"
        self.n_classes = n_classes

    def get_name(self) -> str:
        return self.name

    def fit(self, x: pl.DataFrame, y: pl.DataFrame) -> None:
        logger.info(f"Training {self.name}...")
        assert x is not None  # x is not used
        assert y is not None  # y is not used

    def predict(self, x: pl.DataFrame) -> np.ndarray:
        n_samples = x.height
        return np.ones((n_samples, self.n_classes)) / self.n_classes


class ConstrainedLinearModel(ModelBase):
    """
    Train three regularized linear models for each label and enforce probability axioms.
    """

    def __init__(self, alpha=0.005):
        self.name = "Constrained Linear Model"
        self.model = sklearn.multioutput.MultiOutputRegressor(
            sklearn.linear_model.Ridge(alpha=alpha, random_state=RANDOM_SEED)
        )

    def get_name(self) -> str:
        return self.name

    def fit(self, x: pl.DataFrame, y: pl.DataFrame) -> None:
        logger.info(f"Training {self.name}...")
        self.model.fit(x, y)
        for i, estimator in enumerate(self.model.estimators_):
            print(f"Output {i+1}:")
            print(f"  Coefficients: {estimator.coef_}")
            print(f"  Intercept: {estimator.intercept_}")

    def predict(self, x: pl.DataFrame) -> np.typing.NDArray:
        pred = self.model.predict(x)
        # non-negativity
        pred = np.maximum(pred, 0)

        # sum-to-one
        row_sums = pred.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid division by zero
        probs = pred / row_sums

        return probs


def train_models(x: pl.DataFrame, y: pl.DataFrame) -> list[ModelBase]:
    """
    Cost function: KL divergence. This is appropriate since we are essentially
    learning a distribution that best approximates another one.
    """
    trained_models: list[ModelBase] = []

    def add_model(model_class) -> None:
        model = model_class()
        model.fit(x, y)
        trained_models.append(model)

    # baselines: trivial models
    add_model(MarginalProbabilityModel)
    add_model(UniformProbabilityModel)

    # constrained linear model
    add_model(ConstrainedLinearModel)

    return trained_models
