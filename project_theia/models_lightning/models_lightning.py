from typing import Literal

from project_theia.models_lightning.mnist import model_lightning_mnist


MODEL_CLASSES = [
    model_lightning_mnist.MnistClassifier,
]

MODEL_CONFIGS_LITERAL = Literal[
    model_lightning_mnist.MnistClassifier.CONFIG_CLASS
]

MODELS = {model.NAME: model for model in MODEL_CLASSES}

MODEL_NAME_FROM_CONFIG_NAME = {
    model.CONFIG_CLASS.__name__: model.__name__ for model in MODEL_CLASSES
}

MODEL_FROM_CONFIG_NAME = {model.CONFIG_CLASS.__name__: model for model in MODEL_CLASSES}

MODEL_FROM_NAME = {model.__name__: model for model in MODEL_CLASSES}
