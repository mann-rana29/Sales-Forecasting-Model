from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def get_model(name):
    if name == 'linear':
        return LinearRegression()
    elif name == 'rf':
        return RandomForestRegressor(n_estimators=100,max_depth=10)
    elif name == 'xgb':
        return XGBRegressor(n_estimators = 100 , learning_rate = 0.1)
    else:
        raise ValueError(f"Model '{name}' not implemented")