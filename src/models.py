from .mantis import MantisClassifier
from aeon.classification.dictionary_based import WEASEL_V2
from aeon.classification.interval_based import QUANTClassifier
from aeon.classification.convolution_based import MiniRocketClassifier
from aeon.classification.dictionary_based import MrSQMClassifier
from costream.evaluation import ModelSpec
from src.fall_lm import FallLM
from typing import Optional, Union, List


def get_model_specs(
    random_state: Optional[int] = None,
    include: Optional[Union[str, List[str]]] = None,
    exclude: Optional[Union[str, List[str]]] = None,
) -> List[ModelSpec]:
    """
    Returns model specifications with optional filtering and random seed control.

    Parameters
    ----------
    random_state : int, optional
        Random seed to apply to all models that support it. If None, defaults to 42.

    include : str or list of str, optional
        Include only these model(s) by name. If None, includes all models.
        Can be a single model name (str) or list of model names.

    exclude : str or list of str, optional
        Exclude these model(s) by name. If None, excludes nothing.
        Can be a single model name (str) or list of model names.
        Takes precedence over include if both are specified.

    Returns
    -------
    list of ModelSpec
        List of model specifications matching the filter criteria.

    Examples
    --------
    >>> get_model_specs(random_state=0)  # All models with seed 0
    >>> get_model_specs(include="WEASEL")  # Only WEASEL
    >>> get_model_specs(include=["WEASEL", "Quant"])  # WEASEL and Quant
    >>> get_model_specs(exclude="FallLM")  # All except FallLM
    >>> get_model_specs(exclude=["FallLM", "Mantis"])  # All except FallLM and Mantis
    """

    if random_state is None:
        random_state = 42

    # Convert include/exclude to lists for uniform handling
    include_list = None
    if include is not None:
        include_list = [include] if isinstance(include, str) else include

    exclude_list = []
    if exclude is not None:
        exclude_list = [exclude] if isinstance(exclude, str) else exclude

    # Define all available models
    all_models = {
        "WEASEL": ModelSpec(
            name="WEASEL",
            estimator=WEASEL_V2(random_state=random_state),
        ),
        "MrSQM": ModelSpec(
            name="MrSQM",
            estimator=MrSQMClassifier(random_state=random_state, nsax=5, nsfa=0),
        ),
        "Quant": ModelSpec(
            name="Quant",
            estimator=QUANTClassifier(random_state=random_state),
        ),
        "MiniRocket": ModelSpec(
            name="MiniRocket",
            estimator=MiniRocketClassifier(random_state=random_state),
        ),
        "Mantis": ModelSpec(
            name="Mantis",
            estimator=MantisClassifier(
                device="mps",
                new_seq_len=320,
                fine_tuning_type="full",
                finetune_epochs=100,
                random_state=random_state,
            ),
        ),
        "FallLM": ModelSpec(
            name="FallLM",
            estimator=FallLM(
                n_bins=5,
                word_size=3,
                alpha=1.5,
                ngram_range=(4, 5),
                use_diff=False,
            ),
        ),
    }

    # Filter models
    result_names = include_list if include_list is not None else list(all_models.keys())
    result_names = [name for name in result_names if name not in exclude_list]

    return [all_models[name] for name in result_names if name in all_models]
