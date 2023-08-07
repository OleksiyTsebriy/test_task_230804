from abc import ABC, abstractmethod
from random import randint

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier


class DigitClassificationInterface(ABC):
    """Abstract interface for digit classification

    Methods
    ----------
    train()
        Abstract method for model training
    predict(input: np.ndarray), int
        Abstract method for model inference
    load(path: str)
        Abstract method for model loading
    save(path: str)
        Abstract method for model saving
    """

    @abstractmethod
    def train() -> None:
        ...

    @abstractmethod
    def predict(self, input: np.ndarray) -> int:
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        ...


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


class DigitCNNClassifier(DigitClassificationInterface):
    """PyTorch CNN implementation of DigitClassificationInterface

    Attributes
    ----------
    model: CNN
        PyTorch architecture of CNN network

    Methods
    ----------
    train()
        Method for model training
    predict(input: np.ndarray), int
        Method for model inference
    load(path: str)
        Method for model loading
    save(path: str)
        Method for model saving
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = CNN()

    def train(self) -> None:
        raise NotImplementedError()

    def predict(self, input: np.ndarray) -> int:
        """Inference input using CNN model

        Parameters
        ----------
        input : np.ndarray
            Input image matrix

        Returns
        -------
        int
            Image classification
        """
        return self.model(input).item()

    def load(self, path: str) -> None:
        """Load model state

        Parameters
        ----------
        path : str
            Path to the model state
        """
        self.model.load_state_dict(torch.load(path))

    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)


class DigitRFClassifier(DigitClassificationInterface):
    """Random Forest implementation of DigitClassificationInterface

    Attributes
    ----------
    model: sklearn.ensemble.RandomForestClassifier
        Random Forest classification object

    Methods
    ----------
    train()
        Method for model training
    predict(input: np.ndarray), int
        Method for model inference
    load(path: str)
        Method for model loading
    save(path: str)
        Method for model saving
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = RandomForestClassifier()

    def train(self) -> None:
        raise NotImplementedError()

    def predict(self, input: np.ndarray) -> int:
        return self.model(input.flatten())

    def load(self, path: str) -> None:
        self.model = joblib.load(path)

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)


class DigitRandClassifier(DigitClassificationInterface):
    """PyTorch CNN implementation of DigitClassificationInterface

    Attributes
    ----------
    model: CNN
        PyTorch architecture of CNN network

    Methods
    ----------
    train()
        Method for model training
    predict(input: np.ndarray), int
        Method for model inference
    load(path: str)
        Method for model loading
    save(path: str)
        Method for model saving
    """

    def __init__(self) -> None:
        super().__init__()

    def train(self):
        raise NotImplementedError()

    def predict(self, input: np.ndarray) -> int:
        input = input[9:19, 9:19]
        return randint(0, 9)

    def load(self, path: str) -> None:
        return

    def save(self, path: str) -> None:
        return


MODELS = {
    "cnn": DigitCNNClassifier,
    "rf": DigitRFClassifier,
    "rand": DigitRandClassifier
}


class DigitClassifier:
    def __init__(self, algorithm: str) -> None:
        """Model constructor

        Parameters
        ----------
        algorithm : str, {'cnn', 'rf', 'rand'}
            Type of the model
        """
        super().__init__()
        algorithm = algorithm.lower()
        if algorithm not in MODELS.keys():
            raise NotImplementedError()

        self.model: DigitClassificationInterface = MODELS[algorithm]()

    def train(self) -> None:
        raise NotImplementedError()

    def predict(self, input: np.ndarray) -> int:
        return self.model(input)

    def load(self, path: str) -> None:
        self.model.load(path)

    def save(self, path: str) -> None:
        self.model.save(path)
