import os
import pandas as pd
import numpy as np

import os
import json
import tools
import pandas as pd
import pickle
import numpy as np
from typing import Optional

import lightning.pytorch as pl
from pytorch_forecasting import DeepAR, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from lightning.pytorch.callbacks import EarlyStopping

############## load dataset

def set_dataset(dir: str, metadata: dict, target_column: str = "forecast_time_observed_wspd"):
    whole = pd.read_csv(dir, index_col=0)
    whole.index = pd.to_datetime(whole.index)
    
    # Define columns to drop
    useless_columns = (
        [c for c in whole.columns if "forecast_time" in c and "wspd" not in c] +
        ["is_valid"] +
        [c for c in whole.columns if "observed" in c and "0" not in c and "forecast" not in c] +
        [c for c in whole.columns if "wdir" in c]
    )
    
    # Drop useless columns and select feature columns in one step
    df = whole.drop(columns=useless_columns)
    feature_columns = [c for c in df.columns if c != target_column]
    print(feature_columns)

    df["time_idx"] = range(len(df))
    df["series_idx"] = "0"
    
    # Define train and test split based on basis_time
    df["basis_time"] = pd.to_datetime(df["basis_time"])
    
    train_data = df[(df["basis_time"] >= "2023-01-01 00:00:00") & (df["basis_time"] <= "2023-06-29 23:00:00")]
    test_data = df[(df["basis_time"] >= "2023-06-30 00:00:00") & (df["basis_time"] <= "2023-12-31 23:00:00")]
    
    # Common parameters for TimeSeriesDataSet
    common_params = {
        "time_idx": "time_idx",
        "target": target_column,
        "group_ids": ["series_idx"],
        "static_categoricals": ["series_idx"],
        "time_varying_known_reals": feature_columns,
        "time_varying_unknown_reals": [target_column],
        "max_encoder_length": metadata["max_encoder_length"],
        "max_prediction_length": metadata["forecasting_horizon"]
    }
    
    # Create datasets
    train = TimeSeriesDataSet(train_data, **common_params)
    test = TimeSeriesDataSet(test_data, **common_params)
    
    test_evaluation = test_data.copy()
    
    return train, test, test_evaluation


def set_dataloader(train, test, metadata):
    batch_size = metadata["batch_size"]
    num_workers = metadata["num_workers"]
    
    common_params = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "persistent_workers": True
    }
    
    train_dataloader = train.to_dataloader(train=True, **common_params)
    test_dataloader = test.to_dataloader(train=False, **common_params)
    
    return train_dataloader, test_dataloader

######### train and forecast
import pandas as pd
import pickle
import numpy as np

import lightning.pytorch as pl
from pytorch_forecasting import DeepAR
from lightning.pytorch.callbacks import EarlyStopping

def train_model(train_dataset, metadata):
    """
    Train the DeepAR model using PyTorch Lightning.
    """
    early_stop_callback = EarlyStopping(monitor="train_loss", patience=10, mode="min")

    trainer = pl.Trainer(
        max_epochs = metadata["max_epochs"],
        accelerator="gpu",
        enable_model_summary=True,
        gradient_clip_val=1e-3,
        callbacks=[early_stop_callback],
        enable_checkpointing=True
    )

    model = DeepAR.from_dataset(
        train_dataset,
        learning_rate=metadata["learning_rate"],
        hidden_size=metadata["hidden_size"],
        rnn_layers=metadata["rnn_layers"],
        optimizer="Adam",
    )

    trainer.fit(model, train_dataloaders=train_dataset.to_dataloader(train=True, batch_size=metadata["batch_size"]))

    best_model_path = trainer.checkpoint_callback.best_model_path
    net = DeepAR.load_from_checkpoint(best_model_path)

    return net

def forecast(net, test_dataloader, save_root, metadata, save_prediction=True) -> dict:
    
    
    predictions = net.predict(test_dataloader, return_x=True, return_index=True, n_samples=1000, mode="raw")
    index = predictions.index.time_idx.values  # 예측값의 인덱스
    output = predictions.output.prediction

    if output is None or output.nelement() == 0:
        print("⚠ 예측값이 없습니다. 모델이 데이터를 제대로 처리하지 못했습니다.")
        return {}

    predictions_numpy = output.cpu().numpy()
    mu_values = predictions_numpy.mean(axis=1).flatten()  # 평균 (mu)
    sigma_values = predictions_numpy.std(axis=1).flatten()  # 표준편차 (sigma)

    if len(index) > 24:
        index = index[24:]
        mu_values = mu_values[24:]
        sigma_values = sigma_values[24:]
    else:
        print("⚠ Warning: 데이터가 24개보다 적어 조정할 수 없습니다.")
        return {}

    time_index_prediction = {
        int(idx): {"mu": float(mu), "sigma": float(sigma)}
        for idx, mu, sigma in zip(index, mu_values, sigma_values)
    }

    if save_prediction:
        with open(f"{save_root}/time_index_prediction.pkl", "wb") as f:
            pickle.dump(time_index_prediction, f)

    return time_index_prediction


################################# SIMULATION TOOLS #################################

NUM_SIMULATION = 1000

def gaussian_sampling(row): 
    return np.random.normal(loc=row["mu"], scale=row["sigma"], size=NUM_SIMULATION)

def simulate(pred_df: pd.DataFrame) -> pd.DataFrame:
    """ 
    Gaussian sampling using predicted value 
    
    PARAMTERS 
    ---------
    prediction : pd.DataFrame
        The prediction dataset. <MUST CONTAIN "mu" AND "sigma" COLUMNS>
    
    RETURNS
    -------
    simulation : pd.DataFrame
        The simulated values based on Gaussian sampling.
    """
    print("Applying Gaussian sampling...")

    simulation = pred_df.apply(gaussian_sampling, axis=1)
    simulation = pd.DataFrame(simulation.to_list(), index=pred_df.index)
    simulation.dropna(inplace=True)   
    simulation.index.name = "forecasting_time"

    return simulation

def to_bucket(simulation_single_row:np.array, WSPD_BASIS):
    dist = {i: 0 for i in WSPD_BASIS}
    increment =  WSPD_BASIS[1] - WSPD_BASIS[0]
    for value in simulation_single_row: 
        dist[list(dist.keys())[int(value//increment)]] += 1

    return dist.values() 


def transform_to_pdf(simulation:pd.DataFrame, observed_Y:pd.Series) -> pd.DataFrame: 
    """ 
    Convert the simulation result to distribution format.
    """
    Y_BASIS_START = observed_Y.min()
    Y_BASIS_END = observed_Y.max()

    Y_BASIS = np.linspace(Y_BASIS_START, Y_BASIS_END, 300)

    simulation[simulation < Y_BASIS_START] = pd.to_numeric(Y_BASIS_START)
    simulation[simulation > Y_BASIS_END] = pd.to_numeric(Y_BASIS_END)
    simulation = simulation - Y_BASIS_START
    print("Converting simulation result to distribution format...")

    distribution = pd.DataFrame(simulation.apply(func=to_bucket, axis=1, args=(Y_BASIS,)).to_list(), 
                                index=simulation.index,
                                columns=Y_BASIS)
    
    return distribution / NUM_SIMULATION


def get_weighted_average(data:pd.DataFrame, weight:np.ndarray): 
    average = np.inner(data.values, weight) 
    res = pd.DataFrame(average, index=data.index, columns=["mu"])
    res.index.name = "forecasting_time"
    res.index = pd.to_datetime(res.index) 

    return res


################################# EVALUATION TOOLS #################################

def rmse(observed_y, expect):
    """Root-mean-square deviation (RMSE)

    Args:
        observed_y : Observed dependent variable
        expect : Expected value of dependent variable.
    """
    return np.sqrt(np.sum((observed_y - expect)**2) / len(observed_y))


def mae(observed_y, expㅇect):
    """Mean-absolute error (MAE)

    Args:
        observed_y : Observed dependent variable
        expect : Expected value of dependent variable.
    """
    return np.sum(np.abs(observed_y - expect)) / len(observed_y)


def mape(observed_y, expect):
    """Mean-absolute percentage error (MAPE)

    Args:
        observed_y : Observed dependent variable
        expect : Expected value of dependent variable.
    """
    if len(observed_y[observed_y == 0]):
        return None

    return 100*(np.sum(np.abs((observed_y - expect) / observed_y)) / len(observed_y))

def nmape(observed_y, expect, location, farm):

    capacity_mapping = {
        ("sinan", "w100002"): 62700,
        ("dongbok", "w100001"): 30000,
        ("gasiri", "w100026"): 15000
    }
    capacity = capacity_mapping.get((location, farm))
    if capacity is None:
        raise ValueError("Invalid location or farm. Please provide a valid combination.")

    valid_indices = observed_y >= capacity * 0.1

    if not valid_indices.any():
        return None

    valid_observed_y = observed_y[valid_indices]
    valid_expect = expect[valid_indices]

    return 100 * (np.sum(np.abs(valid_observed_y - valid_expect) / capacity) / len(valid_observed_y))  # sinan

def crps(observed_y, expected_distributions:pd.DataFrame) -> float: 
    """Continuous Ranked Probability Score (CRPS)

    Args:
        observed_y : Observerd dependent variable
        is_valid : Indicator of valid data 
        expect_distribution (pd.DataFrame): Expected probability density function of dependent variable.
    """
    
    distribution_keys = expected_distributions.columns.to_numpy()
    distribution_key_increment = distribution_keys[1] - distribution_keys[0]
    degenerated_distribution = np.array([
        np.heaviside(distribution_keys - y, 1) for y in observed_y
    ])
    
    estimated_cdf = np.cumsum(expected_distributions.to_numpy(), axis=1)
    crps = np.sum((estimated_cdf - degenerated_distribution)**2 * distribution_key_increment, axis=1)
    
    return crps.mean()

def evaluate(observed_y:pd.Series, pred_df:pd.DataFrame, expected_distribution:pd.DataFrame=None, LOCATION=None, FARM=None, is_valid:Optional[pd.Series]=None): 
    res = dict() 

    # Ensure pred_df is a DataFrame
    if not isinstance(pred_df, pd.DataFrame):
        raise ValueError("Prediction must be a DataFrame containing 'mu' column.")

    # Check if 'drift' column exists
    if "mu" not in pred_df.columns:
        raise ValueError("'mu' column is missing in the prediction DataFrame.")
    
    pred_df.dropna(inplace=True)

    if not isinstance(pred_df, pd.DataFrame):
        raise ValueError("Prediction must be a DataFrame containing 'mu' column.")

    # Check if 'drift' column exists
    if "mu" not in pred_df.columns:
        raise ValueError("'mu' column is missing in the prediction DataFrame.")

    
    # Filter with is_valid if provided
    if is_valid is not None:
        #is_valid = prediction.index.intersection(is_valid.index)
        #prediction = prediction.loc[is_valid] 
        valid_index = pred_df.index.intersection(is_valid.index)
        pred_df = pred_df.loc[valid_index][is_valid.loc[valid_index]]
    
    # Ensure intersection of indices
    intersection_index = observed_y.index.intersection(pred_df.index)
    
    # Align observed_y and prediction to intersection_index
    observed_y = observed_y.loc[intersection_index].to_numpy()
    pred_df = pred_df.loc[intersection_index]

    expected_value = pred_df["mu"].to_numpy()

    # Compute evaluation metrics
    res["RMSE"] = rmse(observed_y, expected_value)
    res["MAE"] = mae(observed_y, expected_value)
    res["nMAPE"] =nmape(observed_y, expected_value, location=LOCATION, farm=FARM)
    res["MAPE"] = mape(observed_y, expected_value)
    #res["SMAPE"] = smape(observed_y, expected_value)

    # Optional: CRPS calculation
    if expected_distribution is not None: 
        expected_distribution = expected_distribution.loc[intersection_index]
        if is_valid is not None: 
            expected_distribution = expected_distribution.loc[is_valid]
        res["CRPS"] = crps(observed_y, expected_distribution) 

    return res 

def save_evaluation_result(save_dir: str, res: dict):
    inference_result = {
        "evaluation": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in res.items()}
    }
    with open(save_dir, "w") as f:
        json.dump(inference_result, f, indent=4)

def create_new_dir(path:str): 
    if not os.path.exists(path): 
        os.makedirs(path)
        print("created new directory: ", path)


################ run model 

LOCATION = "sinan"                  
FARM = "w100002"

for num in range(1, 37):
    metadata = {
                "num_workers": 12,
                "hidden_size": 128,
                "rnn_layers": 8,
                "max_epochs": 50,
                "max_encoder_length": 48,
                "batch_size": 64,
                "learning_rate": 1e-3,
                "forecasting_horizon": num
            }

    data_dir = f"/content/drive/MyDrive/Lab_DeepAR/training_data/sinan/horizon_{num}.csv"
    save_root = "/content/drive/MyDrive/Lab_DeepAR/Result"
    target_column = "forecast_time_observed_wspd"

    whole = pd.read_csv(data_dir, index_col=0)
    whole.index = pd.to_datetime(whole.index)

    train, test, test_df = set_dataset(data_dir, metadata, target_column)
    train_dataloader, test_dataloader = set_dataloader(train, test, metadata)

    model = train_model(train, metadata)
    time_index_prediction = forecast(model, test_dataloader, save_root, metadata, save_prediction=True)

    df_prediction = pd.DataFrame.from_dict(time_index_prediction, orient="index")  # Dictionary → DataFrame
    index_map = {i: t for i, t in enumerate(whole.index)}  

    df_prediction.index = df_prediction.index.map(index_map)
    df_prediction.index.name = "forecasting_time"  # 인덱스 이름 설정

    test_df = whole.loc["2023-06-30 00:00:00":"2023-12-31 23:00:00"]

    simulation = simulate(df_prediction) 
    observed_Y = test_df[target_column]
    distribution = transform_to_pdf(simulation, observed_Y)

    Y_COLUMN_I = ["forecast_time_observed_wspd"] 
    IS_VALID_I = ["is_valid"]

    observed_y = test_df[Y_COLUMN_I]
    is_valid_df = test_df[IS_VALID_I]

    evaluate_result = evaluate(
            observed_y=observed_y['forecast_time_observed_wspd'], 
            pred_df=df_prediction, 
            expected_distribution=distribution, 
            LOCATION = "sinan", 
            FARM="w100002",
            is_valid= is_valid_df["is_valid"])
        #evaluate_result = tools.evaluate(observed_y=valid[Y_COLUMN], prediction=forecast, expected_distribution=distribution)
    print(evaluate_result)

    SAVE_PATH = f'/content/drive/MyDrive/Lab_DeepAR/Result'
    create_new_dir(SAVE_PATH)
    save_evaluation_result(os.path.join(SAVE_PATH, f"result_{num}.json"), evaluate_result)
    df_prediction.to_csv(os.path.join(SAVE_PATH, f"forecast_{num}.csv"))
