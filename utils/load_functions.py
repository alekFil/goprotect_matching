import json
import pickle
from pathlib import Path

import joblib
import torch


def load_resources(model_name, resources_type, file_type):
    model_path = Path(f"resources/{model_name}") / f"{resources_type}.{file_type}"

    if file_type == "pt":
        resources = torch.load(model_path)
    elif file_type == "joblib":
        with open(model_path, "rb") as file:
            resources = joblib.load(file)
    elif file_type == "pickle":
        with open(model_path, "rb") as file:
            resources = pickle.load(file)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    return resources


def load_evaluations(model_name):
    eval_path = Path("evaluations") / f"{model_name}.json"
    with open(eval_path, "r") as file:
        evaluations = json.load(file)
    return evaluations


def load_plots(model_name):
    plots_path = Path("plots") / f"{model_name}.json"
    try:
        with open(plots_path, "r") as file:
            plots = json.load(file)
        return plots
    except FileNotFoundError:
        return False
