"""Built-in Word2Vec model factory registrations.

Importing this module registers the supported architectures once.
"""

from ..registry import MODEL_REG
from .cbow import CBOW
from .skipgram import SkipGram

MODEL_REG.register("cbow")(CBOW)
MODEL_REG.register("skipgram")(SkipGram)


def make_model(name: str, **kwargs):
    """Instantiate a registered model, preserving the training-engine contract."""
    return MODEL_REG.create(name, **kwargs)
