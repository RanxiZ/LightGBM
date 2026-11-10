# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Load optional model families when their public classes are requested."""
from importlib import import_module

_GROUPS = (
 (("CatBoostModel",), (("catboost_model", "CatBoostModel"),)),
 (("DEnsembleModel", "LGBModel"), (("double_ensemble", "DEnsembleModel"), ("gbdt", "LGBModel"))),
 (("XGBModel",), (("xgboost", "XGBModel"),)),
 (("LinearModel",), (("linear", "LinearModel"),)),
 (("ALSTM", "GATs", "GRU", "LSTM", "DNNModelPytorch", "TabnetModel", "SFM_Model", "TCN", "ADD"),
  (("pytorch_alstm", "ALSTM"), ("pytorch_gats", "GATs"), ("pytorch_gru", "GRU"),
   ("pytorch_lstm", "LSTM"), ("pytorch_nn", "DNNModelPytorch"), ("pytorch_tabnet", "TabnetModel"),
   ("pytorch_sfm", "SFM_Model"), ("pytorch_tcn", "TCN"), ("pytorch_add", "ADD"))),
)
__all__ = [name for names, _ in _GROUPS for name in names] + ["pytorch_classes", "all_model_classes"]

def _load_group(index):
    names, modules = _GROUPS[index]
    try:
        values = [getattr(import_module("." + module, __name__), name) for module, name in modules]
    except ModuleNotFoundError:
        values = [None] * len(names)
    globals().update(zip(names, values))
    if index == 4:
        globals()["pytorch_classes"] = tuple(values) if all(v is not None for v in values) else ()

def __getattr__(name):
    if name == "all_model_classes":
        for index in range(len(_GROUPS)):
            if _GROUPS[index][0][0] not in globals():
                _load_group(index)
        value = tuple(globals()[n] for n in ("CatBoostModel", "DEnsembleModel", "LGBModel", "XGBModel", "LinearModel")) + globals()["pytorch_classes"]
        globals()[name] = value
        return value
    if name == "pytorch_classes":
        _load_group(4)
        return globals()[name]
    for index, (names, _) in enumerate(_GROUPS):
        if name in names:
            _load_group(index)
            return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return sorted(set(globals()) | set(__all__))
