import pandas
import numpy as np
from matplotlib import axes, pyplot as plt
import matplotlib

confirmed = pandas.read_csv('time_series_covid19_confirmed_global.csv')
deaths = pandas.read_csv('time_series_covid19_deaths_global.csv')
recovered = pandas.read_csv('time_series_covid19_recovered_global.csv')

n_days = None
n_countries = None

print(confirmed.head())
print(deaths.head())
print(recovered.head())

"""
Rearrange the Canada entries
"""
# indexes for canada's confirmed and deaths
confirmed_idx = confirmed.index[confirmed["Country/Region"]=="Canada"]
deaths_idx = deaths.index[deaths["Country/Region"]=="Canada"]

'''
1. canada confirmed handling
'''
total_confirmed = confirmed.loc[confirmed_idx].drop(columns=["Province/State","Country/Region","Lat","Long"], axis=1).sum()
canada_confirmed_line = pandas.DataFrame(total_confirmed)

# create a new confirmed array where you drop all canada entries except one
confirmed_new = confirmed.drop((confirmed.index[confirmed["Country/Region"]=="Canada"])[1:]).reset_index(drop=True)
print(confirmed_new)
# find the index of the only canada entry
canada_conf_idx = int(confirmed_new.index[confirmed_new["Country/Region"]=="Canada"].to_numpy())

# drop the columns that does not hold data
confirmed_copy = confirmed_new.drop(columns=["Province/State","Country/Region","Lat","Long"], axis=1)
# replace canada row with Canada sum
total_confirmed = pandas.concat([confirmed_copy.iloc[:canada_conf_idx], canada_confirmed_line.transpose(), confirmed_copy.iloc[canada_conf_idx+1:]]).reset_index()
total_confirmed = total_confirmed.drop(columns=['index'], axis=1)

'''
2. canada deaths handling
'''

total_deaths = deaths.loc[deaths_idx].drop(columns=["Province/State","Country/Region","Lat","Long"], axis=1).sum()
canada_deaths_line = pandas.DataFrame(total_deaths)

deaths_new = deaths.drop((deaths.index[deaths["Country/Region"]=="Canada"])[1:]).reset_index(drop=True)
canada_deaths_idx = int(deaths_new.index[deaths_new["Country/Region"]=="Canada"].to_numpy())

deaths_copy = deaths_new.drop(columns=["Province/State","Country/Region","Lat","Long"], axis=1)
total_deaths = pandas.concat([deaths_copy.iloc[:canada_deaths_idx], canada_deaths_line.transpose(), deaths_copy.iloc[canada_deaths_idx+1:]]).reset_index()
total_deaths = total_deaths.drop(columns=['index'], axis=1)


days = []
months = []
year = []
for col in confirmed_new.columns[4:]:
    date = col.split('/')
    days.append(int(date[1]))
    months.append(int(date[0]))
    year.append(int(date[2]))

Country = confirmed_new["Country/Region"].tolist()
Lat = confirmed_new["Lat"].tolist()
Long = confirmed_new["Long"].tolist()

Country1 = deaths_new["Country/Region"].tolist()
Lat1 = deaths_new["Lat"].tolist()
Long1 = deaths_new["Long"].tolist()

Country2 = recovered["Country/Region"].tolist()
Lat2 = recovered["Lat"].tolist()
Long2 = recovered["Long"].tolist()

n_days = len(recovered.columns[4:])
n_countries = len(recovered)

Country = [ele for ele in Country for i in range(n_days)]
Lat = [ele for ele in Lat for i in range(n_days)]
Long = [ele for ele in Long for i in range(n_days)]

days = np.tile(days, n_countries)
months = np.tile(months, n_countries)
year = np.tile(year, n_countries)

confirmed_dates = confirmed.drop(columns=["Province/State","Country/Region","Lat","Long"], axis=1)
deaths_dates = deaths.drop(columns=["Province/State","Country/Region","Lat","Long"], axis=1)
recovered_dates = recovered.drop(columns=["Province/State","Country/Region","Lat","Long"], axis=1)

"""
#Confirmed section
"""

# list to store the confirmed for each day
confirmed_each_day_by_country = []
# for each country
for index, row in total_confirmed.iterrows():
    # convert to list
    temp = row.tolist()
    # calculate differences 
    temp_con = [t - s for s, t in zip(temp, temp[1:])]
    # set negative values to zero
    temp_con = [0 if x < 0 else x for x in temp_con]
    # insert the data from the first recorded day
    temp_con.insert(0, temp[0])
    # append to the country list
    confirmed_each_day_by_country.append(temp_con)

# flatten the list
confirmed_each_day_by_country =[item for sublist in confirmed_each_day_by_country for item in sublist]

"""
#Deaths section
"""

# list to store the deaths for each day
deaths_each_day_by_country = []
# for each country
for index, row in total_deaths.iterrows():
    # convert to list
    temp = row.tolist()
    # calculate differences 
    temp_deaths = [t - s for s, t in zip(temp, temp[1:])]
    # set negative values to zero
    temp_deaths = [0 if x < 0 else x for x in temp_deaths]
    # insert the data from the first recorded day
    temp_deaths.insert(0, temp[0])
    # append to the country list
    deaths_each_day_by_country.append(temp_deaths)

# flatten the list
deaths_each_day_by_country =[item for sublist in deaths_each_day_by_country for item in sublist]

"""
#Recovered section

#NOTE: we have a problem here since recovered file only has 265 countries,
#i think we should be checking for country match at all times
"""

# list to store the recovered for each day
recovered_each_day_by_country = []
# for each country
for index, row in recovered_dates.iterrows():
    # convert to list
    temp = row.tolist()
    # calculate differences 
    temp_rec = [t - s for s, t in zip(temp, temp[1:])]
    # set negative values to zero
    temp_rec = [0 if x < 0 else x for x in temp_rec]
    # insert the data from the first recorded day
    temp_rec.insert(0, temp[0])
    # append to the country list
    recovered_each_day_by_country.append(temp_rec)


# flatten the list
recovered_each_day_by_country =[item for sublist in recovered_each_day_by_country for item in sublist]


print("Length of confirmed: ", len(confirmed_each_day_by_country))
print("Length of deaths: ", len(deaths_each_day_by_country))
print("Length of recovered: ", len(recovered_each_day_by_country))

col_names = ["Country/Region", "Lat", "Long", "Day", "Month", "Year", "Confirmed", "Deaths", "Recovered"]
data = np.column_stack((Country,Lat,Long,days,months,year, confirmed_each_day_by_country, deaths_each_day_by_country, recovered_each_day_by_country))
data = pandas.DataFrame(data, columns = col_names)
print("\n\n")
print(list(data.columns))
print("\n\n")
data.to_csv('data.csv')

data['Year'] = data['Year'].astype('int')
data['Month'] = data['Month'].astype('int')
data['Day'] = data['Day'].astype('int')

data['Confirmed'] = data['Confirmed'].astype('int')
data['Deaths'] = data['Deaths'].astype('int')
data['Recovered'] = data['Recovered'].astype('int')

data_grouped = data.groupby(['Year', 'Month'])[["Confirmed", "Deaths", "Recovered"]].sum()
print(data_grouped)
print("\n")

print(data.dtypes)
"""
plt.figure(figsize=(15, 35))
plt.ylabel("Country")
plt.xlabel("Deaths")
plt.barh(data["Country/Region"], data["Deaths"])

plt.savefig("plot.png", bbox_inches= "tight", dpi = 100)
"""