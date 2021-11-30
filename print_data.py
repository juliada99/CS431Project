import pandas
import numpy as np
from matplotlib import axes, pyplot as plt
import matplotlib

confirmed = pandas.read_csv('time_series_covid19_confirmed_global.csv')
deaths = pandas.read_csv('time_series_covid19_deaths_global.csv')
recovered = pandas.read_csv('time_series_covid19_recovered_global.csv')

n_days = None
n_countries = None
print(" ")
print(" ")
print(confirmed.head())
#print(deaths.head())
#print(recovered.head())