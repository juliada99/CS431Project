import pandas
import numpy as np
confirmed = pandas.read_csv('/home/shaineh/Big_Data/csse_covid_19_data/csse_covid_19_time_series/confirmed_global.csv')
deaths = pandas.read_csv('/home/shaineh/Big_Data/csse_covid_19_data/csse_covid_19_time_series/deaths_global.csv')
recovered = pandas.read_csv('/home/shaineh/Big_Data/csse_covid_19_data/csse_covid_19_time_series/recovered_global.csv')
lookup = pandas.read_csv('/home/shaineh/Big_Data/csse_covid_19_data/csse_covid_19_time_series/LookUp_Table.csv')

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

Country = [ele for ele in Country for i in range(662)]
Lat = [ele for ele in Lat for i in range(662)]
Long = [ele for ele in Long for i in range(662)]

days = np.tile(days, 280)
months = np.tile(months, 280)
year = np.tile(year, 280)

col_names = ["Country/Region", "Lat", "Long", "Day", "Month", "Year"]
data = np.column_stack((Country,Lat,Long,days,months,year))
data = pandas.DataFrame(data, columns = col_names)
data.to_csv('data.csv')
print(data)

