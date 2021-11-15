import pandas as pd 


def load_file(filename):
    data = pd.read_csv(filename)
    data.drop('Province/State', axis=1, inplace=True)
    return data


deaths = load_file("time_series_covid19_deaths_global.csv")
confirmed = load_file("time_series_covid19_confirmed_global.csv")
recovered = load_file("time_series_covid19_recovered_global.csv")


print(deaths.iloc[0:50,:])
print(confirmed.iloc[0:50,300:320])
print(recovered.iloc[0:50,300:320])