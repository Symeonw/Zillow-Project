import stats
import stats_functions

def select_kbest_freg_scaled(X_train,y_train, k):
   k_selector = SelectKBest(f_regression, k = k).fit(X_train,y_train)
   k_support = k_selector.get_support()
   k_feature = X_train.loc[:, k_support].columns.tolist()
   return k_feature, k_selector
   
def select_kbest_freg_scaled(X_train_scaled, y_train_scaled, k):
    k_feature_scaled, k_selector_scaled = select_kbest_freg_unscaled(X_train=X_train_scaled, y_train=y_train_scaled, k=k)
    return f_feature_scaled, f_selector_scaled

def ols_backward_elimination(X_train, y_train):
    cols = (X_train.columns)
    pmax = 1
    while pmax > .05:
        if len(cols)!=0:
            X_1 = X_train[cols]
            model = sm.old(y_train, X_1).fit()
            pmax = max(model.pvalues)
            pmax_id = model.pvalues.idxmax()
        else:
            break
        if (pmax > .05):
            cols.remove(pmax_id)
    return pd.Series(cols)

def lasso_cv_coef(X_train, y_train):
    las = LassoCV()
    las.fit(X_train, y_train)
    coef = pd.Series(las.coef_, index = X_train_columns)
    bar_plot = sns.barplor(x = X_train.columns, y = las.coef_)
    return coef, bar_plot

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
    return number_of_features

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

