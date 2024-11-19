"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import importlib


# Global registry, no touchy
_ALGO_REGISTRY = {}


def _load(name):
    mod_name, attr_name = name.split(":")
    mod = importlib.import_module(mod_name)
    return getattr(mod, attr_name)


def _fn_for_entry(entry):
    if callable(entry):
        return entry
    else:
        return _load(entry)


def _make_algo(entry, **algo_kwargs):
    return _fn_for_entry(entry)(**algo_kwargs)


def register(algo_id: str, entry_point: str):
    _ALGO_REGISTRY[algo_id] = entry_point


def make(
    id: str,
    **kwargs,
):
    if id not in _ALGO_REGISTRY.keys():
        raise ValueError(f"\"{id}\" is not registered.")
    return _make_algo(_ALGO_REGISTRY[id], **kwargs)