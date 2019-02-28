import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression
import os
import pyrebase
try:
    ROOT = os.path.dirname(os.path.realpath(__file__))
except:
    ROOT = os.getcwd()

with open(os.path.join(ROOT, 'firebase_config.json'),'rb') as f:
    config = json.load(f)
config["serviceAccount"] = os.path.join(ROOT,"autoaerie-firebase-adminsdk-nw230-36d33619c0.json")
firebase = pyrebase.initialize_app(config)
db = firebase.database()

thermostat_id = '271sIc5a9G9ZTY6dEDulAjWfgC833gx7'

thermostat_history = db.child('thermostats').child(thermostat_id).child('history').get().val()
tdf = pd.DataFrame(thermostat_history)
tdf['datetime'] = [datetime.utcfromtimestamp(x)-timedelta(hours=5) for x in tdf.timestamp]
tdf['input_on'] = tdf['hvac_state']!='off'
tdf['is_home'] = tdf['away']=='home'
tdf = tdf.set_index('datetime')
tdf_minute = tdf.resample('T').mean().interpolate('zero')

thermostat = db.child('thermostats').child(thermostat_id).child('latest_info').get().val()
weather_key = thermostat['weather_key']
weather_history = db.child('weather').child(weather_key).child('history').get().val()
wdf = pd.DataFrame(weather_history)
wdf['datetime'] = [datetime.utcfromtimestamp(x)-timedelta(hours=5) for x in wdf.timestamp]
wdf = wdf.set_index('datetime')
wdf_minute = wdf.resample('T').mean().interpolate('zero')

df_minute = pd.merge(tdf_minute, wdf_minute, how='left', left_index=True, right_index=True)

# freqs = np.arange(10,361,5)
# scores = []
# for freq in freqs:
freq = 5
freq_str = '%dT'%(freq)
df_hour = df_minute.resample(freq_str).mean()
df_hour['actual_temperature_f_lag'] = df_hour.actual_temperature_f.shift(1)
df_hour['temperature_f_lag'] = df_hour.temperature_f.shift(1)
df_hour = df_hour.dropna()
df_hour['diff_temperature_f'] = df_hour['actual_temperature_f'] - df_hour['actual_temperature_f_lag']
df_hour['delta_temperature_f'] = df_hour['actual_temperature_f_lag'] - df_hour['temperature_f_lag']
df_hour['actual_temperature_f_lag^2'] = np.power(df_hour['actual_temperature_f_lag'],2)

# df_minute['actual_temperature_f_lag_10'] = df_minute.actual_temperature_f.shift(10)
# df_minute = df_minute.dropna()
# df_minute['diff_temperature_f'] = df_minute['actual_temperature_f'] - df_minute['actual_temperature_f_lag_10']

df_hour[['actual_temperature_f','target_temperature_f', 'temperature_f', 'input_on', 'is_home']].plot()
# (55+5*df_hour.input_on).rolling(2,center=True).mean().plot(color='g',label='input_on (1hr MA)')
plt.grid(True)
plt.legend()

plt.show()
