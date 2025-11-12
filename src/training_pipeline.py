# training pipeline: train, eval, tune

from pathlib import Path
import argparse
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.feature_pipeline import split_data
import mlflow
import optuna
from joblib import load, dump


DATA_PATH = Path(__file__).parent.parent / 'usdvnd' / 'processed' / 'cleaned_features_regression.csv'

SAVE_MODEL_PATH = Path(__file__).parent.parent / 'models' / 'xgboost_regression'
SAVE_MODEL_PATH.mkdir(parents=True, exist_ok=True)

SAVE_TUNED_MODEL_PATH = Path(__file__).parent.parent / 'models' / 'xgboost_regression' / 'tuned_model'
SAVE_TUNED_MODEL_PATH.mkdir(parents=True, exist_ok=True)

MLFLOW_TRACKING_URI = Path(__file__).parent.parent / 'mlruns'


def train(
    data_path: Path = DATA_PATH,
    save_model_path: Path = SAVE_MODEL_PATH
) -> None:
    """
    Train the model
    Args:
        data_path: Path = DATA_PATH: path to the data
        save_model_path: Path = SAVE_MODEL_PATH: path to save the model
    Returns:
        None
    """
    df = pd.read_csv(data_path)
    df.set_index('Ngày', inplace=True)
    df.index = pd.to_datetime(df.index)
    X_train, y_train, X_test, y_test = split_data(df, split_ratio=1)

    # train the model
    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)

    # save the model
    dump(model, str(save_model_path / 'best_model.pkl'))
    print(f"Model saved to {save_model_path / 'best_model.pkl'}")

def eval(
    data_path: Path = DATA_PATH,
    model_path: Path = SAVE_MODEL_PATH,
) -> None:
    """
    Evaluate the model
    Args:
        data_path: Path = DATA_PATH: path to the data
        model_path: Path = SAVE_MODEL_PATH: path to the model
    Returns:
        tuple[float, float, float]: MAE, MSE, R2
    """
    df = pd.read_csv(data_path)
    df.set_index('Ngày', inplace=True)
    df.index = pd.to_datetime(df.index)
    X_train, y_train, X_test, y_test = split_data(df, split_ratio=0.8)

    # load the model
    model = load(str(model_path))

    # evaluate the model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mae, mse, r2


def tune(
    data_path: Path = DATA_PATH,
    n_trials: int = 10,
    tracking_uri: Path = MLFLOW_TRACKING_URI,
    experiment_name: str = "xgboost_regression",
    save_tuned_model_path: Path = SAVE_TUNED_MODEL_PATH,
) -> tuple[dict, dict]:
    """
    Tune the model
    Args:
        data_path: Path = DATA_PATH: path to the data
        n_trials: int = 10: number of trials
        tracking_uri: Path = MLFLOW_TRACKING_URI: path to the tracking URI
        experiment_name: str = "xgboost_regression": name of the experiment
        save_tuned_model_path: Path = SAVE_TUNED_MODEL_PATH: path to save the tuned model
    Returns:
        tuple[dict, dict]: best parameters and metrics
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    if experiment_name:
        mlflow.set_experiment(experiment_name)

    df = pd.read_csv(data_path)
    df.set_index('Ngày', inplace=True)
    df.index = pd.to_datetime(df.index)
    X_train, y_train, X_test, y_test = split_data(df, split_ratio=0.8)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "hist",
        }

        with mlflow.start_run(nested=True):
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mlflow.log_params(params)
            mlflow.log_metrics({"mae": mae, "mse": mse, "r2": r2})
            mlflow.xgboost.log_model(model, "model")
        return mae
    # create study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    
    # End any active run from the study
    mlflow.end_run()
    
    best_params = study.best_trial.params
    
    # retrain the model with the best parameters
    best_model = xgb.XGBRegressor(**best_params)
    best_model.fit(X_train, y_train)
    
    # Evaluate the best model
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    log_metrics = {
        "mae": mae,
        "mse": mse,
        "r2": r2
    }

    dump(best_model, str(save_tuned_model_path / 'tuned_model.pkl'))
    print(f"Tuned model saved to {save_tuned_model_path / 'tuned_model.pkl'}")
    
    # log the best model
    with mlflow.start_run(run_name="best_xgboost_model"):
        mlflow.log_params(best_params)
        mlflow.log_metrics(log_metrics)
        mlflow.xgboost.log_model(best_model, "model")
    
    print(f"Best parameters: {best_params}")
    print(f"Log metrics: {log_metrics}")
    return best_params, log_metrics

def main():
    parser = argparse.ArgumentParser(description="Training pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # train command
    train_parser = subparsers.add_parser("train", description="Train the model")
    train_parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    train_parser.add_argument("--save-model-path", type=Path, default=SAVE_MODEL_PATH)
    train_parser.set_defaults(func=train)

    # tune command
    tune_parser = subparsers.add_parser("tune", description="Tune the model")
    tune_parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    tune_parser.add_argument("--n-trials", type=int, default=10)
    tune_parser.add_argument("--tracking-uri", type=Path, default=MLFLOW_TRACKING_URI)
    tune_parser.add_argument("--experiment-name", type=str, default="xgboost_regression")
    tune_parser.add_argument("--save-tuned-model-path", type=Path, default=SAVE_TUNED_MODEL_PATH)
    tune_parser.set_defaults(func=tune)

    # eval command
    eval_parser = subparsers.add_parser("eval", description="Evaluate the model")
    eval_parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    eval_parser.add_argument("--model-path", type=Path, default=SAVE_MODEL_PATH)
    eval_parser.set_defaults(func=eval)

    args = parser.parse_args()
    if args.command == "train":
        train(args.data_path, args.save_model_path)
    elif args.command == "tune":
        tune(args.data_path, args.n_trials, args.tracking_uri, args.experiment_name, args.save_tuned_model_path)
    elif args.command == "eval":
        eval(args.data_path, args.model_path)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

    # guide to use the pipeline
    """
    To train the model:
    python training_pipeline.py train --data-path <data-path> --save-model-path <save-model-path>
    To tune the model:
    python training_pipeline.py tune --data-path <data-path> --n-trials <n-trials> --tracking-uri <tracking-uri> --experiment-name <experiment-name> --save-tuned-model-path <save-tuned-model-path>
    To evaluate the model:
    python training_pipeline.py eval --data-path <data-path> --model-path <model-path>
    """