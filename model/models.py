import logging
import pathlib
from abc import ABC, abstractmethod

import joblib
import numpy as np
import polars as pl
import sklearn
import torch
import xgboost as xgb
from constants import RANDOM_SEED

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)

np.random.seed(RANDOM_SEED)


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
        logger.info("Training completed!")

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
        logger.info("Training completed!")

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
        logger.info("Training completed!")
        # for i, estimator in enumerate(self.model.estimators_):
        #     print(f"Output {i+1}:")
        #     print(f"  Coefficients: {estimator.coef_}")
        #     print(f"  Intercept: {estimator.intercept_}")

    def predict(self, x: pl.DataFrame) -> np.typing.NDArray:
        pred = self.model.predict(x)
        # non-negativity
        pred = np.maximum(pred, 0)

        # sum-to-one
        row_sums = pred.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid division by zero
        probs = pred / row_sums

        return probs


class BoostedTreeModel(ModelBase):
    """
    Train XGBoost classifier for each action. This is fundamentally different from the
    regression approach: we are not trying to predict the whole distribution, but rather
    just the most probable action. Then, we use the prediction probabilities for the most
    probable action as a proxy for the CFR distribution.
    """

    def __init__(self):
        self.name = "Boosted Tree"
        self.model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.1,
            objective="multi:softprob",
            num_class=3,
            random_state=RANDOM_SEED,
            n_jobs=10,
            tree_method="exact",  # skip quantization
            verbosity=0,
        )

    def get_name(self) -> str:
        return self.name

    def fit(self, x: pl.DataFrame, y: pl.DataFrame) -> None:
        logger.info(f"Training {self.name}...")
        # TAKE <= 1M SAMPLES
        sample_size = 1000000
        if len(x) > sample_size:
            indices = np.random.choice(len(x), sample_size, replace=False)
            x = x[indices]
            y = y[indices]

        x_np = x.to_numpy()
        y_np = y.to_numpy()
        y_labels = np.argmax(y_np, axis=1)

        self.model.fit(x_np, y_labels)
        logger.info(f"Trained on {len(x_np)} samples")

    def predict(self, x: pl.DataFrame) -> np.typing.NDArray:
        return self.model.predict_proba(x.to_numpy())


class NeuralNetworkModel(ModelBase):
    """
    Neural network model for predicting CFR distributions.
    Uses regression approach with softmax output to predict
    probability distributions directly.

    My gut tells me this is better than doing classification
    since it captures the nuances in how the features affect
    each action's probabilities separately. With XGBoost we
    didn't have a choice. With NN, a softmax output layer will
    ensure probabilities are valid.
    """

    def __init__(
        self,
        hidden_layers: tuple[int, ...] = (128, 64, 32),
        dropout_rate: float = 0.1,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        epochs: int = 50,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.name = "Neural Network"
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device

    def get_name(self) -> str:
        return self.name

    def _build_model(self, input_dim: int, output_dim: int = 3) -> torch.nn.Module:
        """Build the neural network architecture."""
        layers = []
        prev_dim = input_dim

        # hidden layers
        for hidden_dim in self.hidden_layers:
            layers.append(torch.nn.Linear(prev_dim, hidden_dim))

            # relu is probably the most appropriate activation, since
            # all features and cfr output probabilities are nonnegative.
            layers.append(torch.nn.ReLU())
            if self.dropout_rate > 0:
                layers.append(torch.nn.Dropout(self.dropout_rate))
            prev_dim = hidden_dim

        # output layer (softmax is applied separately, no activation here)
        layers.append(torch.nn.Linear(prev_dim, output_dim))
        return torch.nn.Sequential(*layers).to(self.device)

    def fit(self, x: pl.DataFrame, y: pl.DataFrame) -> None:
        logger.info(f"Training {self.name}...")
        x_np = x.to_numpy()
        y_np = y.to_numpy()

        self.input_dim = x_np.shape[1]
        self.scaler = sklearn.preprocessing.StandardScaler()
        x_np = self.scaler.fit_transform(x_np)

        # convert to torch tensors
        x_tensor = torch.FloatTensor(x_np).to(self.device)
        y_tensor = torch.FloatTensor(y_np).to(self.device)

        # build model skeleton
        self.model = self._build_model(input_dim=self.input_dim, output_dim=3)

        # kl divergence as loss function
        criterion = torch.nn.KLDivLoss(reduction="batchmean")

        # adaptive learning rate
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=10
        )

        # training
        self.model.train()
        dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_x)

                # apply softmax on outputs to get probabilities
                probs = torch.softmax(outputs, dim=1)

                # torch.nn.KLDivLoss requires log probabilities as input
                # and true probabilities as target
                loss = criterion(torch.log(probs + 1e-10), batch_y)

                # backward pass
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            scheduler.step(avg_loss)
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.6f}")

        logger.info("Training completed!")

    def predict(self, x: pl.DataFrame) -> np.typing.NDArray:
        if self.model is None or self.scaler is None:
            raise ValueError("Model must be trained before prediction")

        # convert to numpy and normalize
        x_np = x.to_numpy()
        x_np = self.scaler.transform(x_np)

        # convert to tensor
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x_np).to(self.device)
            outputs = self.model(x_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_np = probs.cpu().numpy()

        # ensure numerical stability
        pred_np = np.maximum(pred_np, 0)
        row_sums = pred_np.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        pred_np = pred_np / row_sums

        return pred_np


def train_models(x: pl.DataFrame, y: pl.DataFrame) -> list[ModelBase]:
    """
    Cost function: KL divergence. This is appropriate since we are essentially
    learning a distribution that best approximates another one.
    """
    trained_models: list[ModelBase] = []

    def add_model(model_class) -> None:
        model = model_class()
        model.fit(x, y)
        joblib.dump(
            model,
            pathlib.Path(".") / f"{model.get_name().lower().replace(" ", "_")}.joblib",
        )
        trained_models.append(model)

    # baselines: trivial models
    add_model(MarginalProbabilityModel)
    add_model(UniformProbabilityModel)

    # constrained linear model
    add_model(ConstrainedLinearModel)

    # boosted tree model
    add_model(BoostedTreeModel)

    # neural network model
    add_model(NeuralNetworkModel)

    return trained_models
