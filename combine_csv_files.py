# Program for loading and combining data files
import pandas as pd 


def load_file(filename):
    data = pd.read_csv(filename)
    data.drop('Province/State', axis=1, inplace=True)
    return data

main_df = pd.DataFrame()

deaths = load_file("time_series_covid19_deaths_global.csv")

for index, row in deaths.iterrows():
    country = row.iloc[0]
    lat = row.iloc[1] #lat
    lon = row.iloc[2] #long

    row = row[3:]
    key_list = ["Country", "Latitude", "Longitude", "Day", "Month", "Year"]

    temp_df = pd.DataFrame(columns=key_list)
    #print(row)
    #break
    # create 662 entries 
    for i, item in enumerate(row.iteritems()):
        #print(item)
        #print(i)
        #print(item[i])

        #print(item[index+1])
        #country_dir = dict.fromkeys(key_list)
        #print(country_dir)
        
        #print(type(item[0])) # date
        val_list = []
        val_list.append(country)
        val_list.append(lat)
        val_list.append(lon)
        array = item[0].split("/")
        month = array[0] 
        day = array[1]
        year = array[2]
        val_list.append(day)
        val_list.append(month)
        val_list.append(year)
        country_dir = dict(zip(key_list, val_list))
        #deaths_that_day = item[index]
        #print(type(item[1])) # number of deaths
        #print(country_dir.keys())
        #print(country_dir.values())
        temp_df = temp_df.append(country_dir, ignore_index=True)
    main_df = main_df.append(temp_df, ignore_index=True)

    print(main_df)
#main_df.to_csv('out.csv')  

    


confirmed = load_file("time_series_covid19_confirmed_global.csv")
recovered = load_file("time_series_covid19_recovered_global.csv")

"""
print(deaths.iloc[0:50,:])
print(confirmed.iloc[0:50,300:320])
print(recovered.iloc[0:50,300:320])
"""