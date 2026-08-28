import json
import mlflow
import os
import tempfile
import hydra
from omegaconf import DictConfig


_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
]


@hydra.main(version_base=None, config_name='config', config_path='.')
def go(config: DictConfig):

    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    steps_param = config["main"]["steps"]
    print(f"steps_param = {steps_param}")

    if steps_param == "all":
        active_steps = _steps
    elif isinstance(steps_param, str):
        active_steps = [steps_param]
    else:
        active_steps = list(steps_param)

    print(f"active_steps = {active_steps}")

    with tempfile.TemporaryDirectory() as tmp_dir:

        print(f"tmp_dir = {tmp_dir}")

        if "download" in active_steps:
            print("Running download step...")
            _ = mlflow.run(
                os.path.join(
                    hydra.utils.get_original_cwd(),
                    "components",
                    "get_data"
                ),
                "main",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw_file_as_downloaded",
                },
            )

        if "basic_cleaning" in active_steps:
            print("Running basic_cleaning step...")
            _ = mlflow.run(
                os.path.join(
                    hydra.utils.get_original_cwd(),
                    "src",
                    "basic_cleaning"
                ),
                "main",
                parameters={
                    "input_artifact": "sample.csv:latest",
                    "output_artifact": "clean_sample.csv",
                    "output_type": "clean_sample",
                    "output_description": "Data_with_outliers_removed",
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"],
                },
            )

        if "data_check" in active_steps:
            print("Running data_check step...")
            _ = mlflow.run(
                os.path.join(
                    hydra.utils.get_original_cwd(),
                    "src",
                    "data_check"
                ),
                "main",
                parameters={
                    "csv": "clean_sample.csv:latest",
                    "ref": "clean_sample.csv:reference",
                    "kl_threshold": config["data_check"]["kl_threshold"],
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"],
                },
            )

        if "data_split" in active_steps:
            print("Running data_split step...")
            _ = mlflow.run(
                os.path.join(
                    hydra.utils.get_original_cwd(),
                    "components",
                    "train_val_test_split"
                ),
                "main",
                parameters={
                    "input": "clean_sample.csv:latest",
                    "test_size": config["modeling"]["test_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"],
                },
            )

        if "train_random_forest" in active_steps:
            print("Running train_random_forest step...")

            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(
                    dict(config["modeling"]["random_forest"].items()),
                    fp
                )

            _ = mlflow.run(
                os.path.join(
                    hydra.utils.get_original_cwd(),
                    "src",
                    "train_random_forest"
                ),
                "main",
                parameters={
                    "trainval_artifact": "trainval_data.csv:latest",
                    "val_size": config["modeling"]["val_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"],
                    "rf_config": rf_config,
                    "max_tfidf_features": config["modeling"]["max_tfidf_features"],
                    "output_artifact": "model_export",
                },
            )

        if "test_regression_model" in active_steps:
            print("Running test_regression_model step...")
            _ = mlflow.run(
                os.path.join(
                    hydra.utils.get_original_cwd(),
                    "components",
                    "test_regression_model"
                ),
                "main",
                parameters={
                    "mlflow_model": "model_export:prod",
                    "test_dataset": "test_data.csv:latest",
                },
            )

    print("Pipeline complete!")


if __name__ == "__main__":
    go()