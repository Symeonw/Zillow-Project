import stats
import stats_functions
url = f'mysql+pymysql://{user}:{password}@{host}/zillow'
df = pd.read_sql('''select bathroomcnt, bedroomcnt, fireplacecnt, yearbuilt, taxvaluedollarcnt,calculatedfinishedsquarefeet from properties_2017 join propertylandusetype using (propertylandusetypeid) join predictions_2017 using (parcelid) where transactiondate between "2017-05-01" and "2017-06-30" and propertylandusedesc = "Single Family Residential";''', url)
df = df.dropna()
df.columns = ("bathroom_count", "bedroom_count", "fireplace_count", "year_built", "home_value", "square_feet")

df.info()
df.describe()
df.shape
train, test=  train_test_split(df, train_size = .70, random_state = 123)
X_train = train.drop(columns = ["home_value", "year_built", "square_feet", "fireplace_count"])
X_test = test.drop(columns = ["home_value", "year_built", "square_feet", "fireplace_count"])
y_train = train[["home_value"]]
y_test = test[["home_value"]]
X_train_scaled = train_scaled.drop(columns = ["home_value"])
X_test_scaled = test_scaled.drop(columns = ["home_value"])
y_train_scaled = train_scaled[["home_value"]]
y_test_scaled = test_scaled[["home_value"]]
g = sns.PairGrid(train)
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)
#selecting number of features from bedroom_count, bathroom_count, and square_feet
n_features = optimal_number_of_features(X_train,y_train,X_test,y_test)
selected_features = optimal_features(X_train,y_train,n_features)
#running 3 features within regression to form baseline
lm = LinearRegression().fit(X_train, y_train)
lm.intercept_
lm.coef_
y_pred_lm=lm.predict(X_test)
len(y_pred_lm)
y_pred_raveled=y_pred_lm.ravel().reshape(635)
y_pred_raveled = [int(x) for x in y_pred_raveled]
y_test=np.array(y_test).ravel().reshape(635)
pd.DataFrame({'predicted_values':y_pred_raveled,'real_values':y_test})
x = df[["bathroom_count", "bedroom_count", "square_feet"]]
y= df['home_value']
df_regress = pd.DataFrame({})
df_regress["y"] = y
ols_model = ols('y ~ x', data=df_regress).fit()
df_regress['yhat'] = ols_model.predict(x)
df_regress['residual'] = df_regress['yhat'] - df_regress['y']
df_regress.head()
df_regress['residual^2'] = df_regress.residual ** 2
SSE_2 = mean_squared_error(df_regress.y, df_regress.yhat)*len(df_regress)
MSE_2 = mean_squared_error(df_regress.y, df_regress.yhat)
RMSE_2 = sqrt(MSE_2)
df_regress = pd.DataFrame(np.array(['SSE','MSE','RMSE']), columns=['metric'])
df_regress['model_error'] = np.array([SSE_2, MSE_2, RMSE_2])
df_regress['baseline_error'] = np.array([SSE_baseline, MSE_baseline, RMSE_baseline])
df_regress['error_delta'] = df_regress.model_error - df_regress.baseline_error
y_pred_baseline = np.array([y_test.mean()])
y_pred_baseline = np.repeat(y_pred_baseline, len(y_test))
evs = explained_variance_score(y_test, y_pred_baseline)
df_regress, evs
pd.Series(y_pred_baseline)
y_unscaled=y_test
y_lm=y_pred_lm
y_baseline = y_pred_baseline
pd.set_option('display.float_format', lambda x: '%.3f' % x)
df_origin_final = pd.DataFrame({'actual': y_unscaled,
             'lm': y_lm.ravel(),
             'lm_baseline':y_baseline.ravel()})
df_origin_final.head()
%matplotlib inline
df_chart = pd.DataFrame({'actual': y_unscaled,'lm1': y_lm.ravel(),'lm_baseline':y_baseline.ravel()}).melt(id_vars=['actual'],var_name='model',value_name='prediction').pipe((sns.lmplot,'data'),x='actual',y='prediction',hue='model',palette="Set1")
