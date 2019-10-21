import stats
import stats_functions
url = f'mysql+pymysql://{user}:{password}@{host}/zillow'
df = pd.read_sql('''select bathroomcnt, bedroomcnt, fireplacecnt, yearbuilt, taxvaluedollarcnt,calculatedfinishedsquarefeet from properties_2017 join propertylandusetype using (propertylandusetypeid) join predictions_2017 using (parcelid) where transactiondate between "2017-05-01" and "2017-06-30" and propertylandusedesc = "Single Family Residential";''', url)
df = df.dropna()
df.columns = ("bathroom_count", "bedroom_count", "fireplace_count", "year_built", "home_value", "square_feet")
train, test=  train_test_split(df, train_size = .70, random_state = 123)
X_train = train.drop(columns = ["home_value"])
X_test = test.drop(columns = ["home_value"])
y_train = train[["home_value"]]
y_test = test[["home_value"]]
X_train_scaled = train_scaled.drop(columns = ["home_value"])
X_test_scaled = test_scaled.drop(columns = ["home_value"])
y_train_scaled = train_scaled[["home_value"]]
y_test_scaled = test_scaled[["home_value"]]

scaler, train_scaled, test_scaled = standard_scaler(train, test)
k_feature, k_selector = select_kbest_freg_unscaled(X_train, y_train, 3)
k_feature
k_selector
f_feature_scaled, f_selector_scaled = select_kbest_freg_scaled(X_train_scaled, y_train_scaled,3)
f_feature_scaled
selected_features_rfe = optimal_features(X_train, y_train, 3)
selected_features_rfe
X_train_df, X_test_df = optimal_dataframe(X_train, X_test, selected_features_rfe)
X_train_df

sns.heatmap(train.corr(), cmap="bone_r", annot=True)




lm = LinearRegression().fit(X_train, y_train)
y_inter = lm.intercept_
lm_coef = lm.coef_
lm_coef
y_preds = lm.predict(X_test)
y_preds = [int(lm) for lm in y_preds]
y_baseline = y_test.mean()
#create DataFrame for Regress
predict=pd.DataFrame({"actual":y_train.home_value}).reset_index(drop=True)
predict["baseline"] = y_train.mean()[0]
y_baseline = predict[["baseline"]]



#Regress 0 
y_preds = preds[0]
y_preds = [int(x) for x in y_preds]
lm=LinearRegression()
lm.fit(X_train,y_train)
lm_y_intercept=lm.intercept_
lm_coefficents=lm.coef_
y_pred_lm=lm.predict(X_test)
y_pred_lm
y_pred_raveled=y_pred_lm.ravel().reshape(635)
y_test=np.array(y_test).ravel().reshape(635)
y_pred_raveled = [int(x) for x in y_pred_raveled]
dflm = pd.DataFrame({'predicted_values':y_preds,'real_values':y_test})
mse_lm = mean_squared_error(y_test, y_preds)
r2_lm = r2_score(y_test, y_preds)
r2_lm
mse_lm
dflm
#Regress 1
lm1 = LinearRegression().fit(X_train[["bathroom_count", "fireplace_count", "square_feet"]], y_train)
lm1_predict = lm1.predict(X_train[["bathroom_count", "fireplace_count", "square_feet"]])
lm1_predict = [int(lm) for lm in lm1_predict]
predict["lm1"]= lm1_predict

predict.head()
#Regress 2
lm2 = LinearRegression().fit(X_train_scaled[["bathroom_count", "fireplace_count", "year_built"]], y_train)
lm2_predict = lm2.predict(X_train_scaled[["bathroom_count", "fireplace_count", "year_built"]])
lm2_predict = [int(lm) for lm in lm2_predict]
predict["lm2"]= lm2_predict
#Regress 3
lm3 = LinearRegression().fit(X_train_scaled[["bathroom_count", "fireplace_count", "bedroom_count"]], y_train)

lm3_predict = lm3.predict(X_train_scaled[["bathroom_count", "fireplace_count", "bedroom_count"]])
lm3_predict = [int(x) for x in lm3_predict]
predict["lm3"]= lm3_predict
predict
lm3.score(X_test, y_test)
#Eval baseline
MSE_baseline = mean_squared_error(predict.actual, predict.baseline)
SSE_baseline = MSE_baseline*len(predict.actual)
RMSE_baseline = sqrt(MSE_baseline)
r2_baseline = r2_score(predict.actual, predict.baseline)
print(MSE_baseline,SSE_baseline,RMSE_baseline,r2_baseline)
#Eval Regress 1 
x_1 = df[["bathroom_count", "fireplace_count", "square_feet"]]
y_1= df['home_value']
df_1 = pd.DataFrame({})
df_1["y"] = y_1
ols_model = ols('y_1 ~ x_1', data=df).fit()
df_1['yhat'] = ols_model.predict(x)
df_1['residual'] = df_1['yhat'] - df_1['y']
df_1.head()
df_1['residual^2'] = df_1.residual ** 2
SSE_1 = mean_squared_error(df_1.y, df_1.yhat)*len(df_1)
MSE_1 = mean_squared_error(df_1.y, df_1.yhat)
RMSE_1 = sqrt(MSE_1)
df_1 = pd.DataFrame(np.array(['SSE','MSE','RMSE']), columns=['metric'])
df_1['model_error'] = np.array([SSE_1, MSE_1, RMSE_1])
df_1.model_error = [int(x) for x in df_1.model_error]
df_1['baseline_error'] = np.array([SSE_baseline, MSE_baseline, RMSE_baseline])
df_1['error_delta'] = df_1.model_error - df_1.baseline_error
df_1.baseline_error = [int(x) for x in df_1.baseline_error]
df_1.error_delta = [int(x) for x in df_1.error_delta]
df_1
#Eval Regress 2
x_2 = df[["bathroom_count", "fireplace_count", "year_built"]]
y_2= df['home_value']
df_2 = pd.DataFrame({})
df_2["y"] = y_2
ols_model = ols('y_2 ~ x_2', data=df).fit()
df_2['yhat'] = ols_model.predict(x)
df_2['residual'] = df_2['yhat'] - df_2['y']
df_2.head()
df_2['residual^2'] = df_2.residual ** 2
SSE_2 = mean_squared_error(df_2.y, df_2.yhat)*len(df_2)
MSE_2 = mean_squared_error(df_2.y, df_2.yhat)
RMSE_2 = sqrt(MSE_2)
df_2 = pd.DataFrame(np.array(['SSE','MSE','RMSE']), columns=['metric'])
df_2['model_error'] = np.array([SSE_2, MSE_2, RMSE_2])
df_2.model_error = [int(x) for x in df_2.model_error]
df_2['baseline_error'] = np.array([SSE_baseline, MSE_baseline, RMSE_baseline])
df_2['error_delta'] = df_2.model_error - df_2.baseline_error
df_2.baseline_error = [int(x) for x in df_2.baseline_error]
df_2.error_delta = [int(x) for x in df_2.error_delta]
df_2
#Eval Regress 3 - best
x_3 = df[["bathroom_count", "fireplace_count", "bedroom_count"]]
y_3= df['home_value']
df_3 = pd.DataFrame({})
df_3["y"] = y_3
ols_model = ols('y_3 ~ x_3', data=df).fit()
df_3['yhat'] = ols_model.predict(x)
df_3['residual'] = df_3['yhat'] - df_3['y']
df_3.head()
df_3['residual^2'] = df_3.residual ** 2
SSE_3 = mean_squared_error(df_3.y, df_3.yhat)*len(df_3)
MSE_3 = mean_squared_error(df_3.y, df_3.yhat)
RMSE_3 = sqrt(MSE_3)
df_3 = pd.DataFrame(np.array(['SSE','MSE','RMSE']), columns=['metric'])
df_3['model_error'] = np.array([SSE_3, MSE_3, RMSE_3])
df_3.model_error = [int(x) for x in df_3.model_error]
df_3['baseline_error'] = np.array([SSE_baseline, MSE_baseline, RMSE_baseline])
df_3['error_delta'] = df_3.model_error - df_3.baseline_error
df_3.baseline_error = [int(x) for x in df_3.baseline_error]
df_3.error_delta = [int(x) for x in df_3.error_delta]
df_3


#baseline_model
model = lm1.predict(X_test[["bathroom_count",'fireplace_count', "square_feet"]])
model = model.ravel().reshape(635)
model = [int(x) for x in model]
y_test1 = np.array(y_test).ravel().reshape(635)
best_mod = pd.DataFrame({"predictions" :model, "home_value": y_test1})
best_mod.head()




#test 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
PolynomialFeatures().fit_transform(X_train)
pipeline = Pipeline([("poly", PolynomialFeatures()), ("scale", StandardScaler()), ("lr", LinearRegression())]).fit(X_train, y_train)
from sklearn.metrics import classification_report
preds = pipeline.predict(X_test)
preds = pd.DataFrame(preds)

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(random_state = 0).fit(X_train, y_train)
df_4 = pd.DataFrame([(col, coef) for col, coef in zip(X_train.columns, dt.feature_importances_)], columns=["feature", "importance"]).set_index("feature").sort_values("importance", ascending=False).T
df_4

