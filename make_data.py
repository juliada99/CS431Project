import pandas
import numpy as np
confirmed = pandas.read_csv('time_series_covid19_confirmed_global.csv')
deaths = pandas.read_csv('time_series_covid19_deaths_global.csv')
recovered = pandas.read_csv('time_series_covid19_recovered_global.csv')
#lookup = pandas.read_csv('/home/shaineh/Big_Data/csse_covid_19_data/csse_covid_19_time_series/LookUp_Table.csv')

days = []
months = []
year = []
for col in confirmed.columns[4:]:
    date = col.split('/')
    days.append(int(date[1]))
    months.append(int(date[0]))
    year.append(int(date[2]))

Country = confirmed["Country/Region"].tolist()
Lat = confirmed["Lat"].tolist()
Long = confirmed["Long"].tolist()

Country1 = deaths["Country/Region"].tolist()
Lat1 = deaths["Lat"].tolist()
Long1 = deaths["Long"].tolist()

Country2 = recovered["Country/Region"].tolist()
Lat2 = recovered["Lat"].tolist()
Long2 = recovered["Long"].tolist()


Country = [ele for ele in Country for i in range(662)]
Lat = [ele for ele in Lat for i in range(662)]
Long = [ele for ele in Long for i in range(662)]

days = np.tile(days, 280)
months = np.tile(months, 280)
year = np.tile(year, 280)

confirmed_dates = confirmed.drop(columns=["Province/State","Country/Region","Lat","Long"], axis=1)
deaths_dates = deaths.drop(columns=["Province/State","Country/Region","Lat","Long"], axis=1)
recovered_dates = recovered.drop(columns=["Province/State","Country/Region","Lat","Long"], axis=1)

"""
Confirmed section
"""

# list to store the confirmed for each day
confirmed_each_day_by_country = []
# for each country
for index, row in confirmed_dates.iterrows():
    # convert to list
    temp = row.tolist()
    # calculate differences 
    temp_con = [t - s for s, t in zip(temp, temp[1:])]
    # insert the data from the first recorded day
    temp_con.insert(0, temp[0])
    # append to the country list
    confirmed_each_day_by_country.append(temp_con)

# flatten the list
confirmed_each_day_by_country =[item for sublist in confirmed_each_day_by_country for item in sublist]

"""
Deaths section
"""

# list to store the deaths for each day
deaths_each_day_by_country = []
# for each country
for index, row in deaths_dates.iterrows():
    # convert to list
    temp = row.tolist()
    # calculate differences 
    temp_deaths = [t - s for s, t in zip(temp, temp[1:])]
    # insert the data from the first recorded day
    temp_deaths.insert(0, temp[0])
    # append to the country list
    deaths_each_day_by_country.append(temp_deaths)

# flatten the list
deaths_each_day_by_country =[item for sublist in deaths_each_day_by_country for item in sublist]

"""
Recovered section

NOTE: we have a problem here since recovered file only has 265 countries,
i think we should be checking for country match at all times
"""
# list to store the recovered for each day
recovered_each_day_by_country = []
# for each country
for index, row in recovered_dates.iterrows():
    # convert to list
    temp = row.tolist()
    # calculate differences 
    temp_rec = [t - s for s, t in zip(temp, temp[1:])]
    # insert the data from the first recorded day
    temp_rec.insert(0, temp[0])
    # append to the country list
    recovered_each_day_by_country.append(temp_rec)

# flatten the list
recovered_each_day_by_country =[item for sublist in recovered_each_day_by_country for item in sublist]


col_names = ["Country/Region", "Lat", "Long", "Day", "Month", "Year", "Confirmed", "Deaths"]
data = np.column_stack((Country,Lat,Long,days,months,year, confirmed_each_day_by_country, deaths_each_day_by_country))
data = pandas.DataFrame(data, columns = col_names)
data.to_csv('data.csv')
print(data)


