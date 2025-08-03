from train import train_and_log
import os

# df_path = os.path.join("..","data","processed","clean-data.csv")
for model_name in ["linear", "rf", "xgb"]:
    train_and_log("../data/processed/clean-data.csv",target="sales",model_name=model_name)
    train_and_log("../data/processed/after_log_data.csv",target="sales_log",model_name=model_name)