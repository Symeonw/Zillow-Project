def get_df():
    url = f'mysql+pymysql://{user}:{password}@{host}/zillow'
    df = pd.read_sql('''select bathroomcnt, bedroomcnt, fireplacecnt, yearbuilt, taxvaluedollarcnt,calculatedfinishedsquarefeet from properties_2017 join propertylandusetype using (propertylandusetypeid) join predictions_2017 using (parcelid) where transactiondate between "2017-05-01" and "2017-06-30" and propertylandusedesc = "Single Family Residential";''', url)
    return df