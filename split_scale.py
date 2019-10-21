import stats
import stats_functions

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

def split_data(df):
    train, test = split_my_data(df, .70)
    scaler, train_scaled, test_scaled = standard_scaler(train, test)
    X_train = train.drop(columns = ["home_value"])
    X_test = test.drop(columns = ["home_value"])
    y_train = train[["home_value"]]
    y_test = test[["home_value"]]
    X_train_scaled = train_scaled.drop(columns = ["home_value"])
    X_test_scaled = test_scaled.drop(columns = ["home_value"])
    y_train_scaled = train_scaled[["home_value"]]
    y_test_scaled = test_scaled[["home_value"]]
    return train, test, X_train, X_test, y_train, y_test, X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled