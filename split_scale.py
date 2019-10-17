import stats
import numpy as numpy
import wrangle
import env


def split_my_data(df, train_size):
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(df, train_size = train_size, random_state = 123)
    return train, test


def standard_scaler(train, test):
    scaler=StandardScaler(copy=True,with_mean=True,with_std=True).fit(train)
    train_scaled_data=scaler.transform(train)
    test_scaled_data=scaler.transform(test)
    train_scaled=pd.DataFrame(train_scaled_data,columns=train.columns).set_index([train.index])
    test_scaled=pd.DataFrame(test_scaled_data,columns=test.columns).set_index([test.index])
    return scaler, train_scaled, test_scaled

def scale_inverse(x):
    scale_inverse = pd.DataFrame(scaler.inverse_transform(x), columns=x.columns.values).set_index([x.index.values])
    return scaled_inverse



def uniform_scaler(train, test):
    scaler = QuantileTransformer(n_quantiles=100, output_distribution='uniform', random_state=123, copy=True).fit(train)
    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return train_scaled, test_scaled

def gaussian_scaler(train, test):
    scaler = PowerTransformer(method='yeo-johnson', standardize=False, copy=True).fit(train)
    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return train_scaled, test_scaled


def min_max_scaler(train, test):
    scaler = MinMaxScaler(copy=True, feature_range=(0,1)).fit(train)
    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return train_scaled, test_scaled

def iqr_robust_scaler(train, test):
    scaler = RobustScaler(quantile_range=(25.0,75.0), copy=True, with_centering=True, with_scaling=True).fit(train)
    train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return train_scaled, test_scaled
