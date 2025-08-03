import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from model_factory import get_model

def train_and_log(df_path, target, model_name):
    df = pd.read_csv(df_path)
    X = df.drop(columns=[target, 'date'])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42)

    mlflow.set_experiment("Sales Forecasting")

    with mlflow.start_run(run_name=f"{model_name}_{target}"):
        model = get_model(model_name)
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)

        if "log" in target:
            y_test_real = np.expm1(y_test)
            y_pred_real = np.expm1(y_pred)
        else:
            y_test_real = y_test
            y_pred_real = y_pred

        mae = mean_absolute_error( y_test_real,y_pred_real)
        rmse = root_mean_squared_error(y_test_real,y_pred_real)

        mlflow.log_param("model",model_name)
        mlflow.log_param("target",target)
        mlflow.log_metric("mae",mae)
        mlflow.log_metric("rmse",rmse)
        mlflow.sklearn.log_model(model,"model")

        print(f"{model_name} on {target} : MAE = {mae:.2f} , RMSE = {rmse:.2f}")
