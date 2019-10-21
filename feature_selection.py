import stats
import stats_functions
df, info, head = prepare_df()
train, test, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = split_data(df)

def optimal_number_of_features(X_train, y_train, X_test, y_test):
    number_of_attributes = X_train.shape[1]
    number_of_features_list=np.arange(1,number_of_attributes)
    high_score=0
    number_of_features=0           
    score_list =[]
    for n in range(len(number_of_features_list)):
        model = LinearRegression()
        rfe = RFE(model,number_of_features_list[n])
        X_train_rfe = rfe.fit_transform(X_train,y_train)
        X_test_rfe = rfe.transform(X_test)
        model.fit(X_train_rfe,y_train)
        score = model.score(X_test_rfe,y_test)
        score_list.append(score)
        if(score>high_score):
            high_score = score
            number_of_features = number_of_features_list[n]
    return number_of_features, high_score

def optimal_features(X_train, y_train, number_of_features):
    cols = list(X_train.columns)
    model = LinearRegression()
    rfe = RFE(model, number_of_features)
    X_rfe = rfe.fit_transform(X_train,y_train)  
    model.fit(X_rfe,y_train)
    temp = pd.Series(rfe.support_,index = cols)
    selected_features_rfe = temp[temp==True].index
    return selected_features_rfe

def optimal_dataframe(X_train, X_test, selected_features_rfe):
    X_train_df = X_train[selected_features_rfe]
    X_test_df = X_test[selected_features_rfe]
    return X_train_df, X_test_df



def select_features():
    n_features, high_score = optimal_number_of_features(X_train,y_train,X_test,y_test)
    selected_features = optimal_features(X_train,y_train,n_features)
    X_train_sf, X_test_sf = optimal_dataframe(X_train, X_test, selected_features)
    return X_train_sf, X_test_sf


