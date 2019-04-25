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
from ASHRAE_55_2017 import calculate_set, reverse_set, est_clo
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

DEFAULT_NIGHT_HOUR = {
    0: 20.5,
    1: 20.5,
    2: 20.5,
    3: 20.5,
    4: 20.5,
    5: 21,
    6: 21,
}
DEFAULT_MORNING_HOUR = {
    0: 7.5,
    1: 7,
    2: 7,
    3: 7,
    4: 7,
    5: 7,
    6: 7.5,
}

SOLAR_SCALE = 12.5
WIND_ALPHA_SCALE = 0.05/20.0
PREHEAT_TOL = 1.0

MIN_HEAT_COOL_GAP = 3

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
    return round(temperature - 0.9971*np.exp(0.02086*temperature)*(1-np.exp(0.0445*(dewpoint-57.2))),1)

def invert_heat_index(heat_index, humidity):
    res = minimize_scalar(lambda T: np.square(heat_index - get_heat_index(T, humidity)), tol=0.0001)
    if res.success:
        return round(res.x,1)
    else:
        return None

def get_weather_key(zipcode):
    lat,lng = zc.get_lat_long(zipcode)
    key = ('%+.2f_%+.2f'%(lat,lng)).replace('.',',')
    return key

def update_weather(zipcode):
    alpha = 1.0/(24*7)
    utcnow = datetime.utcnow()
    key = get_weather_key(zipcode)
    lat,lng = zc.get_lat_long(zipcode)
    latest_weather = db.child('weather').child(key).child('latest').get().val()
    if latest_weather is None or datetime.utcfromtimestamp(latest_weather['timestamp'])+timedelta(minutes=59.5) < utcnow:
        weather = ds.get_weather(lat, lng, hours=1)
        temp = np.mean(weather['temperature'])
        if latest_weather is None:
            average_temp = temp
        else:
            average_temp = (1-alpha)*latest_weather['average_high_temperature_f'] + alpha*temp
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
            'average_high_temperature_f': average_temp,
        }
        db.child('weather').child(key).child('latest').set(latest_weather)

        # add to history
        db.child('weather').child(key).child('history2').child(latest_weather['timestamp']).set(latest_weather)

        # clean up old values
        oldest = int((utcnow - timedelta(days=10) - datetime.utcfromtimestamp(0)).total_seconds())
        for _ in xrange(1000):
            try:
                old = db.child('weather').child(key).child('history2').order_by_key().limit_to_first(2).get().val()
            except:
                break
            done = False
            for k in old.keys():
                if int(k) < oldest:
                    db.child('weather').child(key).child('history2').child(k).remove()
                else:
                    done = True
                    break
            if done:
                break

    return key, latest_weather

def calc_neutral_temp_f(outdoor_temp_f):
    # source: Outdoor temperatures and comfort indoors, Michael Humphreys, 1978, https://doi.org/10.1080/09613217808550656
    t0 = 22.0
    tm = 24.0
    a0 = 23.9
    a1 = 0.295
    to = max(0,(outdoor_temp_f-32)/1.8)
    to_shift = to - t0
    res = 1.8*(a0 + a1*(to_shift)*np.exp(-0.5*np.square(to_shift/tm)))+32
    return res

def get_desired_drybulb_temp(weather, thermostat, custom={}):
    average_outdoor_temp = weather['average_high_temperature_f']
    outdoor_temp_f = weather['temperature_f']
    night_hour = custom.get('night_hour',DEFAULT_NIGHT_HOUR)
    morning_hour = custom.get('morning_hour',DEFAULT_MORNING_HOUR)
    utcnow = datetime.utcnow()
    local_datetime = utcnow.replace(tzinfo=pytz.utc).astimezone(pytz.timezone(weather['timezone']))
    local_zero = local_datetime.replace(hour=0, minute=0, second=0, microsecond=0)
    night_threshold = local_zero + timedelta(hours=night_hour[local_datetime.weekday()])
    morning_threshold = local_zero + timedelta(hours=morning_hour[local_datetime.weekday()])
    is_night = (local_datetime >= night_threshold) or (local_datetime <= morning_threshold)
    day_night = 'night' if is_night else 'day'
    base_target_set = calc_neutral_temp_f(average_outdoor_temp)

    utcdate = utcnow.date()
    sunrise = datetime.utcfromtimestamp(weather['sunrise']) + timedelta(hours=1)
    sunset = datetime.utcfromtimestamp(weather['sunset']) - timedelta(hours=1)
    is_dark = utcnow >= datetime.combine(utcdate, sunset.time())\
                or utcnow <= datetime.combine(utcdate, sunrise.time())
    solar = 0 if is_dark else SOLAR_SCALE*(1.0-weather['cloud_cover_frac'])

    alpha = 0.025 + WIND_ALPHA_SCALE*weather['wind_speed_mph']

    clo = 0.5

    actual_temperature_f = thermostat['ambient_temperature_f']
    tr = (1-alpha)*actual_temperature_f + alpha*(outdoor_temp_f + solar)
    actual_set = round(calculate_set(actual_temperature_f, tr, 0.1, thermostat['humidity'], 1.2, clo, 0),1)

    all_ta = []
    all_target_set = []
    for mode in thermostat['hvac_mode'].split('-'):
        target_set = base_target_set
        if mode=='heat':
            night_shift = custom.get('night_heat',0)
            day_shift = custom.get('day_heat',0)
            sign = 1.0
            pre_var = 'transition_preheating'
        else:
            night_shift = custom.get('night_cool',0)
            day_shift = custom.get('day_cool',0)
            sign = -1.0
            pre_var = 'transition_precooling'

        ta_day = round(reverse_set(target_set+day_shift, outdoor_temp_f+solar, alpha, 0.1, thermostat['humidity'], 1.2, clo, 0),1)
        ta_night = round(reverse_set(target_set+night_shift, outdoor_temp_f+solar, alpha, 0.1, thermostat['humidity'], 1.2, clo, 0),1)
        ta_max = sign*max(sign*ta_day, sign*ta_night)
        target_set_max = target_set + sign*max(sign*day_shift, sign*night_shift)
        if pre_var in custom and custom[pre_var]==day_night:
            ta = ta_max
            target_set = target_set_max
        else:
            if pre_var in custom and custom[pre_var]!=day_night:
                _ = custom.pop(pre_var)
                db.child('thermostats').child(thermostat_id).child('custom').set(custom)
            eta = None
            if is_night:
                ta = ta_night
                target_set = target_set + night_shift
                if ta_night*sign < ta_day*sign:
                    if morning_threshold < local_datetime:
                        eta = morning_threshold + timedelta(hours=24)
                    else:
                        eta = morning_threshold
            else:
                ta = ta_day
                target_set = target_set + day_shift
                if ta_day*sign < ta_night*sign:
                    if night_threshold < local_datetime:
                        eta = night_threshold + timedelta(hours=24)
                    else:
                        eta = night_threshold
            if eta is not None:
                model = get_model(db, thermostat_id, weather_key, mode)
                minutes = min(12*60,(eta - local_datetime).total_seconds()/60.0)
                final_temp = eval_model_temperature_after_time(
                                model,
                                thermostat['ambient_temperature_f'],
                                True,
                                weather['temperature_f'],
                                weather['wind_speed_mph'],
                                minutes,
                            )
                if sign*final_temp+PREHEAT_TOL < sign*ta_max:
                    ta = ta_max
                    target_set = target_set_max
                    custom[pre_var] = day_night
                    db.child('thermostats').child(thermostat_id).child('custom').set(custom)
        all_ta.append(ta)
        all_target_set.append(target_set)

    control = {
        'timestamp': int(time.time()),
        'clo': clo,
        'is_dark': is_dark,
        'is_night': is_night,
        'solar': solar,
        'alpha': alpha,
        'actual_heat_index_f': actual_set,
        'base_target_heat_index_f': round(base_target_set,1),
        'target_heat_index_low_f': round(min(all_target_set),1),
        'target_heat_index_high_f': round(max(all_target_set),1),
        'actual_temperature_f': actual_temperature_f,
        'target_temperature_low_f': min(all_ta),
        'target_temperature_high_f': max(all_ta),
    }
    return all_ta, control


# set target temperature (F)
def set_temperature(thermostat_id, target_temperature):
    # set thermostat (urllib2 throws an error for the redirect which needs to be followed)
    thermo_url = BASE_URL + 'devices/thermostats/' + thermostat_id + '/'
    success = False
    try:
        req = urllib2.Request(
            thermo_url,
            headers=headers,
            data=json.dumps({'target_temperature_f': target_temperature}),
        )
        req.get_method = lambda: 'PUT'
        res = json.loads(urllib2.urlopen(req).read())
        success = res['target_temperature_f'] == target_temperature
    except urllib2.HTTPError as e:
        if e.code==307:
            req = urllib2.Request(
                e.headers.dict['location'],
                headers=headers,
                data=json.dumps({'target_temperature_f': target_temperature}),
            )
            req.get_method = lambda: 'PUT'
            res = json.loads(urllib2.urlopen(req).read())
            success = res['target_temperature_f'] == target_temperature
    return success

# set target temperature range (F)
def set_temperature_range(thermostat_id, target_temperature_range):
    # set thermostat (urllib2 throws an error for the redirect which needs to be followed)
    thermo_url = BASE_URL + 'devices/thermostats/' + thermostat_id + '/'
    success = False
    data = json.dumps({
        'target_temperature_low_f': min(target_temperature_range),
        'target_temperature_high_f': max(min(target_temperature_range)+MIN_HEAT_COOL_GAP, max(target_temperature_range)),
    })
    print data
    try:
        req = urllib2.Request(
            thermo_url,
            headers=headers,
            data=data,
        )
        req.get_method = lambda: 'PUT'
        res = json.loads(urllib2.urlopen(req).read())
        success = res['target_temperature_low_f'] == min(target_temperature_range) and res['target_temperature_high_f'] == max(target_temperature_range)
    except urllib2.HTTPError as e:
        if e.code==307:
            req = urllib2.Request(
                e.headers.dict['location'],
                headers=headers,
                data=data,
            )
            req.get_method = lambda: 'PUT'
            res = json.loads(urllib2.urlopen(req).read())
            success = res['target_temperature_low_f'] == min(target_temperature_range) and res['target_temperature_high_f'] == max(target_temperature_range)
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
    # print data
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
required, control = get_desired_drybulb_temp(weather, thermostat, custom=custom)
if len(required) > 0:
    force_temp_away = False
    if 'desired_away' in custom and custom['desired_away']=='away':
        force_temp_away = True
        if 'desired_eta' in custom and custom['desired_eta'] is not None:
            eta = datetime.utcfromtimestamp(custom['desired_eta'])
            if eta > datetime.utcnow():
                if 'preheating' in custom and custom['preheating']:
                    force_temp_away = False
                else:
                    for mode in thermostat['hvac_mode'].split('-'):
                        if mode=='heat':
                            target = min(required)
                            sign = 1.0
                        else:
                            target = max(required)
                            sign = -1.0
                        model = get_model(db, thermostat_id, weather_key, mode)
                        minutes = min(12*60,(eta - datetime.utcnow()).total_seconds()/60.0)
                        final_temp = eval_model_temperature_after_time(
                                        model,
                                        thermostat['ambient_temperature_f'],
                                        True,
                                        weather['temperature_f'],
                                        weather['wind_speed_mph'],
                                        minutes,
                                    )
                        control['away_minutes'] = minutes
                        control['away_final_temperature_f'] = final_temp
                        print "%s: Predict %.1f degrees in %.1f minutes"%(mode, final_temp, minutes)
                        if sign*final_temp+PREHEAT_TOL < sign*target:
                            force_temp_away = False
                            custom['preheating'] = True
                            db.child('thermostats').child(thermostat_id).child('custom').set(custom)
            else:
                force_temp_away = False
                custom['desired_away'] = 'home'
                custom['preheating'] = False
                custom['desired_eta'] = None
                db.child('thermostats').child(thermostat_id).child('custom').set(custom)

    if force_temp_away:
        required = []
        if 'heat' in thermostat['hvac_mode']:
            required.append(thermostat['away_temperature_low_f'])
        elif 'cool' in thermostat['hvac_mode']:
            required.append(thermostat['away_temperature_high_f'])

required_int = [int(round(r)) for r in required]

success = False
if structure['away'] == 'home':
    if len(required_int) == 1:
        required_int = required_int[0]
        if required_int != thermostat['target_temperature_f']:
            success = set_temperature(thermostat_id, required_int)
            if success:
                time.sleep(5)
                thermostat = get_thermostat_info(thermostat_id, weather_key)
    elif len(required_int) == 2:
        if min(required_int) != thermostat['target_temperature_low_f'] or max(required_int) != thermostat['target_temperature_high_f']:
            success = set_temperature_range(thermostat_id, required_int)
            if success:
                time.sleep(5)
                thermostat = get_thermostat_info(thermostat_id, weather_key)

control['target_temperature_low_f'] = min(required)
control['target_temperature_high_f'] = max(required)
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
    'target_temperature_low_f': round(thermostat['target_temperature_low_f'],1),
    'target_temperature_high_f': round(thermostat['target_temperature_high_f'],1),
    'actual_temperature_f': thermostat['ambient_temperature_f'],
    'humidity': thermostat['humidity'],
    'base_target_heat_index_f': round(control['base_target_heat_index_f'],1),
    'target_heat_index_low_f': round(control['target_heat_index_low_f'],1),
    'target_heat_index_high_f': round(control['target_heat_index_high_f'],1),
    'actual_heat_index_f': round(control['actual_heat_index_f'],1),
}

# add to history if changed
try:
    last_history = db.child('thermostats')\
                        .child(thermostat_id)\
                        .child('history2')\
                        .order_by_key()\
                        .limit_to_last(1)\
                        .get().val().itervalues().next()
except:
    last_history = None
match = False
if last_history is None:
    match = False
else:
    match = True
    for k,v in sub.iteritems():
        if k=='timestamp':
            continue
        if k not in last_history:
            match = False
            break
        if v != last_history[k]:
            match = False
            break
if not match:
    db.child('thermostats')\
        .child(thermostat_id)\
        .child('history2')\
        .child(sub['timestamp'])\
        .set(sub)

# clean up old values
oldest = int((utcnow - timedelta(days=10) - datetime.utcfromtimestamp(0)).total_seconds())
for _ in xrange(1000):
    try:
        old = db.child('thermostats')\
                .child(thermostat_id)\
                .child('history2')\
                .order_by_key()\
                .limit_to_first(2)\
                .get().val()
    except:
        break
    if old is None:
        break
    done = False
    for k in old.keys():
        if int(k) < oldest:
            db.child('thermostats').child(thermostat_id).child('history2').child(k).remove()
        else:
            done = True
            break
    if done:
        break
