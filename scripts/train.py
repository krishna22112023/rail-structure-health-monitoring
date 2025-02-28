import sys
import json
import logging
import click
import os
import joblib
from typing import Dict
from pathlib import Path

import pyprojroot
root = pyprojroot.find_root(pyprojroot.has_dir("config"))
sys.path.append(str(root))

logger = logging.getLogger(__name__)

from config import settings
from src.trainer_sklearn import TrainerSklearn
from src.trainer_dl import TrainerDL

from utils import logger as logging
from utils.plot import plot_confusion_matrix,plot_metrics_epoch

@click.command()
@click.option("--model_name", type=str, help="Model name (grad_boost, random_forest, ann, cnn)")
@click.argument("path", type=str)
@click.option("--model_params", type=Dict, help="Model parameters as a JSON string")
def train(model_name,model_params,path):

    if model_params is None:
        with open(Path(settings.BASE,settings.model_params), "r") as f:
            model_params = json.load(f)
    else:
        model_params = json.loads(model_params) 

    if model_name is None:
        model_name = settings.model_name
    else:
        model_name = model_name.lower()   

    os.makedirs(Path(settings.BASE,f"outputs/{model_name}"),exist_ok=True)

    # Train the model and compute losses and accuracy on train and validation sets
    print("Training model...")
    if model_name == "grad_boost" or model_name == "random_forest":
        trainer = TrainerSklearn(model_name, model_params, Path(settings.BASE,path))
        metrics, model, confusion_matrix = trainer.train_model()
    elif model_name == "ann" or model_name == "cnn":
        trainer = TrainerDL(model_name, model_params, Path(settings.BASE,path))
        metrics, model, confusion_matrix = trainer.train_model()
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    print("Done.")

    #saving model, confusion matrix, training results
    joblib.dump(model, Path(settings.BASE) / f"outputs/{model_name}/model.joblib")
    with open (Path(settings.BASE,f"outputs/{model_name}/metrics.json"), "w") as f:
        json.dump(metrics, f)
    plot_confusion_matrix(confusion_matrix, settings.CATEGORIES, title=f"Confusion Matrix for {model_name}", cmap="Blues", path=Path(settings.BASE,f"outputs/{model_name}/confusion_matrix.jpg"))
    plot_metrics_epoch({key: metrics[key] for key in ["train_loss", "val_loss"]}, title=f"Loss for {model_name}", xlabel="Epoch", ylabel="Loss", figsize=(10, 6), path=Path(settings.BASE,f"outputs/{model_name}/loss.jpg"))
    plot_metrics_epoch({key: metrics[key] for key in ["train_acc", "val_acc"]}, title=f"Accuracy for {model_name}", xlabel="Epoch", ylabel="Accuracy", figsize=(10, 6), path=Path(settings.BASE,f"outputs/{model_name}/accuracy.jpg"))
    print(f"Model files saved to outputs/{model_name}")

if __name__ == "__main__":
    train()

