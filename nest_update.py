from nest_auth import NestAuth
import urllib2
import json
import numpy as np
import math
from scipy.optimize import minimize_scalar
from zipcode_lookup import ZipCode
from weather_api import DarkSky
import pytz
from datetime import datetime, timedelta

na = NestAuth()
zc = ZipCode()
ds = DarkSky()

BASE_URL = 'https://developer-api.nest.com/'

headers = {
    'Content-Type': "application/json",
    'Cache-Control': "no-cache",
    'Authorization': 'Bearer ' + na.get_token(),
}

def get_structure_info():
    # get thermostat id
    url = BASE_URL + 'structures/'
    req = urllib2.Request(url, headers=headers)
    res = json.loads(urllib2.urlopen(req).read())
    res = res.itervalues().next()
    return res

def get_thermostat_info(thermostat_id):
    thermo_url = BASE_URL + 'devices/thermostats/' + thermostat_id + '/'
    req = urllib2.Request(thermo_url, headers=headers)
    res = json.loads(urllib2.urlopen(req).read())
    return res

def get_dewpoint(temperature, humidity):
    RH = float(min(100.0, max(1.0, humidity)))
    T = float(temperature - 32)/1.8
    Td = 243.04*(np.log(RH/100)+((17.625*T)/(243.04+T)))/(17.625-np.log(RH/100)-((17.625*T)/(243.04+T)))
    return Td*1.8 + 32

def get_humidity(temperature, dewpoint):
    T = (temperature - 32)/1.8
    TD = (dewpoint - 32)/1.8
    return 100*(np.exp((17.625*TD)/(243.04+TD))/np.exp((17.625*T)/(243.04+T)))

def get_heat_index(temperature, humidity):
    dewpoint = get_dewpoint(temperature, humidity)
    return temperature - 0.9971*np.exp(0.02086*temperature)*(1-np.exp(0.0445*(dewpoint-57.2)))

def invert_heat_index(heat_index, humidity):
    res = minimize_scalar(lambda T: np.square(heat_index - get_heat_index(T, humidity)), tol=0.0001)
    if res.success:
        return res.x
    else:
        return None

def get_desired_heat_index(zipcode, mode, tol=3.0):
    print 'Calculate Target Heat Index'
    lat,lng = zc.get_lat_long(zipcode)
    weather = ds.get_weather(lat, lng, hours=2)

    utcnow = datetime.utcnow()
    utcdate = utcnow.date()
    is_dark = utcnow >= datetime.combine(utcdate, weather['sunset'].time()) or utcnow <= datetime.combine(utcdate, weather['sunrise'].time())
    local_datetime = utcnow.replace(tzinfo=pytz.utc).astimezone(pytz.timezone(weather['timezone']))
    night_hour = 21
    morning_hour = 6 if local_datetime.weekday() < 5 else 7
    is_night = (local_datetime.hour >= night_hour) or (local_datetime.hour <= morning_hour)
    if mode == 'heat':
        sign = 1.0
        baseline = 64.0 if is_night else 68.0
        baseline = baseline - tol
    else:
        sign = -1.0
        baseline = 72.0 if is_night else 76.0
        baseline = baseline + tol

    cloudy_offset = 0 if is_dark else 1*(np.mean(weather['cloudCover'])-0.5)/0.5
    wind_offset = sign*np.mean(weather['windSpeed'])/20.0
    desired = baseline + cloudy_offset + wind_offset
    print ' baseline', baseline
    print ' clouds', cloudy_offset
    print ' wind', wind_offset

    if mode != 'heat':
        outdoor_heatindex = [get_heat_index(t,h) for t,h in zip(weather['temperature'], weather['humidity'])]
        desired = max(desired, np.mean(outdoor_heatindex)-10.0)

    return desired

# set target temperature (F)
def set_temperature(thermostat_id, target_temperature):
    # set thermostat (urllib2 throws an error for the redirect which needs to be followed)
    thermo_url = BASE_URL + 'devices/thermostats/' + thermostat_id + '/'
    success = False
    try:
        req = urllib2.Request(thermo_url, headers=headers, data=json.dumps({'target_temperature_f': target_temperature}))
        req.get_method = lambda: 'PUT'
        res = json.loads(urllib2.urlopen(req).read())
        success = res['target_temperature_f'] == target_temperature
    except urllib2.HTTPError as e:
        if e.code==307:
            req = urllib2.Request(e.headers.dict['location'], headers=headers, data=json.dumps({'target_temperature_f': target_temperature}))
            req.get_method = lambda: 'PUT'
            res = json.loads(urllib2.urlopen(req).read())
            success = res['target_temperature_f'] == target_temperature
    return success

structure = get_structure_info()
thermostat_id = structure['thermostats'][0]
zipcode = structure['postal_code']

thermostat = get_thermostat_info(thermostat_id)
target = get_desired_heat_index(zipcode, thermostat['hvac_mode'])
required = invert_heat_index(target, thermostat['humidity'])
required_int = int(round(required))
print 'Datetime:', datetime.utcnow()
print 'Mode:', thermostat['hvac_mode']
print 'Target Temperature:', thermostat['target_temperature_f']
print 'Actual Temperature:', thermostat['ambient_temperature_f']
print 'Actual Humidity:', thermostat['humidity']
print 'Actual Heat Index:', get_heat_index(thermostat['ambient_temperature_f'], thermostat['humidity'])
print 'Target Heat Index:', target
print 'Required Set:', required, required_int
if required_int != thermostat['target_temperature_f']:
    success = set_temperature(thermostat_id, required_int)
    print 'Set temperature:', success
print ' '
