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
import time
import os
import cPickle as pickle
import pyrebase
import os
import random
import string
from system_model import get_model, eval_model_temperature_after_time
try:
    ROOT = os.path.dirname(os.path.realpath(__file__))
except:
    ROOT = os.getcwd()

with open(os.path.join(ROOT, 'firebase_config.json'),'rb') as f:
    config = json.load(f)
config["serviceAccount"] = os.path.join(ROOT,"autoaerie-firebase-adminsdk-nw230-36d33619c0.json")
firebase = pyrebase.initialize_app(config)
db = firebase.database()

na = NestAuth()
zc = ZipCode()
ds = DarkSky()

NIGHT_HOUR = {
    0: 21,
    1: 21,
    2: 21,
    3: 21,
    4: 21,
    5: 21,
    6: 21,
}
MORNING_HOUR = {
    0: 5.5,
    1: 5.5,
    2: 5.5,
    3: 5.5,
    4: 5.5,
    5: 6.5,
    6: 6.5,
}

HEAT_DAY = 66.0
HEAT_NIGHT = 62.0
COOL_DAY = 78.0
COOL_NIGHT = 74.0

CLOUD_SCALE = 1.0
WIND_SCALE = (1.0/20.0)
MAX_COOL_DIFF = 10.0

BASE_URL = 'https://developer-api.nest.com/'

headers = {
    'Content-Type': "application/json",
    'Cache-Control': "no-cache",
    'Authorization': 'Bearer ' + na.get_token(),
}

weather_pickle = os.path.join(ROOT,'weather_data.pickle')

def get_structure_info():
    # get thermostat id
    url = BASE_URL + 'structures/'
    req = urllib2.Request(url, headers=headers)
    res = json.loads(urllib2.urlopen(req).read())
    res = res.itervalues().next()
    return res

def get_eta_begin(structure_id):
    url = BASE_URL + 'structures/' + structure_id + '/eta_begin'
    req = urllib2.Request(url, headers=headers)
    res = json.loads(urllib2.urlopen(req).read())
    return res

def get_thermostat_info(thermostat_id, weather_key):
    thermo_url = BASE_URL + 'devices/thermostats/' + thermostat_id + '/'
    req = urllib2.Request(thermo_url, headers=headers)
    res = json.loads(urllib2.urlopen(req).read())
    res['weather_key'] = weather_key
    if res['hvac_mode'] == 'eco':
        res['hvac_mode'] = res['previous_hvac_mode']
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

def get_average_high(zipcode):
    today = datetime.utcnow().replace(hour=0,minute=0,second=0,microsecond=0)
    if os.path.exists(weather_pickle):
        with open(weather_pickle,'rb') as f:
            weather_data = pickle.load(f)
    else:
        weather_data = {}
    if zipcode in weather_data and datetime.strptime(weather_data[zipcode]['date'],'%Y-%m-%d') >= today:
        result = weather_data[zipcode]
    else:
        # print 'Pulling historical weather for', zipcode
        lat,lng = zc.get_lat_long(zipcode)
        weather = ds.get_historical(lat, lng, days=14)
        result = {
            'date': today.strftime('%Y-%m-%d')
        }
        for k,v in weather.iteritems():
            if k=='date' or k=='timezone':
                continue
            result[k] = np.mean(v)
        weather_data[zipcode] = result
        with open(weather_pickle,'wb') as f:
            pickle.dump(weather_data, f)
    return result['temperatureMax']

def get_weather_key(zipcode):
    lat,lng = zc.get_lat_long(zipcode)
    key = ('%+.2f_%+.2f'%(lat,lng)).replace('.',',')
    return key

def update_weather(zipcode):
    utcnow = datetime.utcnow()
    key = get_weather_key(zipcode)
    lat,lng = zc.get_lat_long(zipcode)
    latest_weather = db.child('weather').child(key).child('latest').get().val()
    if latest_weather is None or datetime.utcfromtimestamp(latest_weather['timestamp'])+timedelta(minutes=59.5) < utcnow:
        weather = ds.get_weather(lat, lng, hours=1)
        high_temp = get_average_high(zipcode)
        latest_weather = {
            'timestamp': int((utcnow - datetime.utcfromtimestamp(0)).total_seconds()),
            'timezone': weather['timezone'],
            'temperature_f': np.mean(weather['temperature']),
            'dew_point_f': np.mean(weather['dewPoint']),
            'humidity_frac': np.mean(weather['humidity']),
            'cloud_cover_frac': np.mean(weather['cloudCover']),
            'wind_speed_mph': np.mean(weather['windSpeed']),
            'sunrise': int((weather['sunrise'] - datetime.utcfromtimestamp(0)).total_seconds()),
            'sunset': int((weather['sunset'] - datetime.utcfromtimestamp(0)).total_seconds()),
            'average_high_temperature_f': high_temp,
        }
        db.child('weather').child(key).child('latest').set(latest_weather)
        history = db.child('weather').child(key).child('history').get().val()
        if history is None:
            history = []
        oldest = utcnow - timedelta(days=14)
        history = [latest_weather] + [w for w in history if datetime.utcfromtimestamp(w['timestamp'])>=oldest]
        db.child('weather').child(key).child('history').set(history)
    return key, latest_weather

def get_desired_heat_index(weather, mode, indoor_temperature, actual_heat_index, thermostat_id, custom):
    # print 'Calculate Target Heat Index'
    high_temp = get_average_high(zipcode)
    # for k,v in weather.iteritems():
    #     print ' %s: %s'%(k, str(v))

    utcnow = datetime.utcnow()
    utcdate = utcnow.date()
    sunrise = datetime.utcfromtimestamp(weather['sunrise'])
    sunset = datetime.utcfromtimestamp(weather['sunset'])
    is_dark = utcnow >= datetime.combine(utcdate, sunset.time()) or utcnow <= datetime.combine(utcdate, sunrise.time())

    local_datetime = utcnow.replace(tzinfo=pytz.utc).astimezone(pytz.timezone(weather['timezone']))
    local_zero = local_datetime.replace(hour=0, minute=0, second=0, microsecond=0)
    night_threshold = local_zero + timedelta(hours=NIGHT_HOUR[local_datetime.weekday()])
    morning_threshold = local_zero + timedelta(hours=MORNING_HOUR[local_datetime.weekday()])
    is_night = (local_datetime >= night_threshold) or (local_datetime <= morning_threshold)
    if mode == 'heat':
        sign = 1.0
        baseline = HEAT_NIGHT if is_night else HEAT_DAY
        long_term_factor = max(-2.0,min(0, (high_temp-60.0)/30.0))
    else:
        sign = -1.0
        baseline = COOL_NIGHT if is_night else COOL_DAY
        long_term_factor = max(0,min(2, (high_temp-70.0)/10.0))

    cloudy_offset = 0 if is_dark else CLOUD_SCALE*(weather['cloud_cover_frac']-0.5)/0.5
    wind_offset = WIND_SCALE*sign*weather['wind_speed_mph']
    desired = baseline + cloudy_offset + wind_offset + long_term_factor
    # print ' '
    # print ' baseline', baseline
    # print ' clouds', cloudy_offset
    # print ' wind', wind_offset
    # print ' long_term', long_term_factor

    outdoor_heatindex = get_heat_index(weather['temperature_f'],weather['humidity_frac'])
    if mode != 'heat':
        desired = max(desired, outdoor_heatindex-MAX_COOL_DIFF)

    control = {
        'timestamp': int(time.time()),
        'baseline_f': baseline,
        'cloudy_offset_f': cloudy_offset,
        'wind_offset_f': wind_offset,
        'long_term_offset_f': long_term_factor,
        'outdoor_heat_index_f': outdoor_heatindex,
        'actual_heat_index_f': actual_heat_index,
        'target_heat_index_f': desired,
    }

    return desired, control

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

# set away state
def set_away(structure_id, away):
    # set thermostat (urllib2 throws an error for the redirect which needs to be followed)
    url = BASE_URL + 'structures/' + structure_id
    success = False
    data = json.dumps({'away':away})
    try:
        req = urllib2.Request(url, headers=headers, data=data)
        req.get_method = lambda: 'PUT'
        res = json.loads(urllib2.urlopen(req).read())
        success = res['away'] == away
    except urllib2.HTTPError as e:
        if e.code==307:
            req = urllib2.Request(e.headers.dict['location'], headers=headers, data=data)
            req.get_method = lambda: 'PUT'
            res = json.loads(urllib2.urlopen(req).read())
            success = res['away'] == away
    return success

# set eta
def set_eta(structure_id, eta_timestamp, trip_id):
    # set thermostat (urllib2 throws an error for the redirect which needs to be followed)
    url = BASE_URL + 'structures/' + structure_id + '/eta'
    success = False
    eta = datetime.utcfromtimestamp(eta_timestamp)
    eta_begin = (eta - timedelta(minutes=1))
    eta_end = (eta + timedelta(minutes=6))
    if datetime.utcnow() >= eta_begin:
        return True
    data = json.dumps({
        'trip_id': trip_id,
        'estimated_arrival_window_begin': eta_begin.isoformat() + 'Z',
        'estimated_arrival_window_end': eta_end.isoformat() + 'Z',
    })
    print data
    res = None
    try:
        req = urllib2.Request(url, headers=headers, data=data)
        req.get_method = lambda: 'PUT'
        res = json.loads(urllib2.urlopen(req).read())
    except urllib2.HTTPError as e:
        if e.code==307:
            req = urllib2.Request(e.headers.dict['location'], headers=headers, data=data)
            req.get_method = lambda: 'PUT'
            res = json.loads(urllib2.urlopen(req).read())
    success = False
    try:
        success = res['trip_id'] == trip_id
    except:
        pass
    return success


structure = get_structure_info()
thermostat_id = structure['thermostats'][0]
zipcode = structure['postal_code']
weather_key, weather = update_weather(zipcode)

thermostat = get_thermostat_info(thermostat_id, weather_key)
db.child('thermostats').child(thermostat_id).child('latest_info').update(thermostat)
custom = db.child('thermostats').child(thermostat_id).child('custom').get().val()
if custom is None:
    custom = {}
actual_heat_index = get_heat_index(thermostat['ambient_temperature_f'], thermostat['humidity'])
target, control = get_desired_heat_index(weather, thermostat['hvac_mode'], thermostat['ambient_temperature_f'], actual_heat_index, thermostat_id, custom=custom)
required = invert_heat_index(target, thermostat['humidity'])
required_int = int(round(required))
force_temp_away = False
if 'desired_away' in custom:
    write_custom = False
    success = True
    if not custom['set_away']:
        if custom['desired_away'] != structure['away']:
            success = False
            success = set_away(structure['structure_id'], custom['desired_away'])
            if success:
                structure['away'] = custom['desired_away']
                custom['set_away'] = True
                write_custom = True
        else:
            custom['set_away'] = True
            write_custom = True
    if success and custom['desired_away'] == 'away':
        if structure['away'] == 'away':
            if 'desired_eta' in custom and custom['desired_eta'] is not None:
                eta = datetime.utcfromtimestamp(custom['desired_eta'])
                if eta > datetime.utcnow():
                    model = get_model(db, thermostat_id, weather_key, thermostat['hvac_mode'])
                    minutes = min(12*60,(eta - datetime.utcnow()).total_seconds()/60.0)
                    final_temp = eval_model_temperature_after_time(model, thermostat['ambient_temperature_f'], True, weather['temperature_f'], weather['wind_speed_mph'], minutes)
                    print "Predict %.1f degrees in %.1f minutes"%(final_temp, minutes)
                    if final_temp < required-0.5:
                        force_temp_away = True
                    if 'trip_id' not in custom or custom['trip_id'] is None:
                        custom['trip_id'] = 'trip_' + ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(16))
                        write_custom = True
                    success = set_eta(structure['structure_id'], custom['desired_eta'], custom['trip_id'])
                else:
                    custom['desired_away'] = 'home'
                    write_custom = True
                    success = set_away(structure['structure_id'], custom['desired_away'])
                    if success:
                        structure['away'] = custom['desired_away']
        else:
            custom['desired_away'] = structure['away']
            write_custom = True
    if success and custom['desired_away'] != 'away' and 'trip_id' in custom and custom['trip_id'] is not None:
        custom['trip_id'] = None
        custom['desired_eta'] = None
        write_custom = True
    if write_custom:
        db.child('thermostats').child(thermostat_id).child('custom').set(custom)

success = False
if structure['away'] == 'home' or force_temp_away:
    if required_int != thermostat['target_temperature_f'] or force_temp_away:
        success = set_temperature(thermostat_id, required_int)
        if success:
            time.sleep(10)
            thermostat = get_thermostat_info(thermostat_id, weather_key)

control['target_temperature_f'] = required
control['actual_temperature_f'] = thermostat['ambient_temperature_f']
control['away'] = structure['away']
control['set_temperature'] = success
db.child('thermostats').child(thermostat_id).child('control').update(control)

utcnow = datetime.utcnow()
sub = {
    'timestamp': int((utcnow - datetime.utcfromtimestamp(0)).total_seconds()),
    'away': structure['away'],
    'hvac_mode': thermostat['hvac_mode'],
    'hvac_state': thermostat['hvac_state'],
    'target_temperature_f': thermostat['target_temperature_f'],
    'actual_temperature_f': thermostat['ambient_temperature_f'],
    'humidity': thermostat['humidity'],
    'target_heat_index_f': round(control['target_heat_index_f'],1),
    'actual_heat_index_f': round(control['actual_heat_index_f'],1),
}
history = db.child('thermostats').child(thermostat_id).child('history').get().val()
match = False
if history is None:
    history = []
else:
    history = sorted(history, key=lambda x: x['timestamp'])
    match = True
    for k,v in history[-1].iteritems():
        if k=='timestamp':
            continue
        if v != sub[k]:
            match = False
            break
if not match:
    oldest = utcnow - timedelta(days=14)
    history = [w for w in history if datetime.utcfromtimestamp(w['timestamp'])>=oldest] + [sub]
    db.child('thermostats').child(thermostat_id).child('history').set(history)
