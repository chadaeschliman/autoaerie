import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.optimize import minimize

# Usage:
# weather_key = thermostat['weather_key']
# model = get_model(db, thermostat_id, weather_key, thermostat['hvac_mode'])

MODEL_FREQ = 5
RETRAIN_FREQ = 7
DEFAULT_INPUT = 0.2
DEFAULT_DELTA = -2e-3
DEFAULT_WIND = -4e-5
CURRENT_MODEL_TYPE = 'input, deltaT, windxdeltaT'

INPUT_LOOKUP = {
    'heat': 'heating',
    'cool': 'cooling',
}

def advance_model(T0, reset_period, input_on, outside_temp, wind_mph, coef):
    T = [T0[0]]
    cnt = 0
    for input, outside, wind in zip(input_on, outside_temp, wind_mph):
        if reset_period is not None and cnt >= reset_period:
            cnt = 0
            T.append(T0[len(T)])
        else:
            delta = T[-1] - outside
            diff_T = coef[0]*input + (coef[1] + coef[2]*wind)*delta
            diff_T = min(100, max(-100, diff_T))
            T.append(T[-1] + diff_T)
        cnt += 1
    return T

def eval(T0, reset_period, input_on, outside_temp, wind_mph, coef):
    err = []
    for offset in xrange(0,reset_period,1):
        temp_T0 = T0[offset:]
        temp_input_on = input_on[offset:]
        temp_outside_temp = outside_temp[offset:]
        temp_wind_mph = wind_mph[offset:]
        temp_T = advance_model(temp_T0, reset_period, temp_input_on, temp_outside_temp, temp_wind_mph, coef)
        err.extend((temp_T0 - temp_T))
    return np.sqrt(np.mean(np.square(np.maximum(-10, np.minimum(10, err)))))

def train_model(db, thermostat_id, weather_key, x0, mode):
    thermostat_history = [v for v in db.child('thermostats').child(thermostat_id).child('history2').order_by_key().get().val().itervalues()]
    if len(thermostat_history) < 24*60//5:
        return None
    tdf = pd.DataFrame(thermostat_history)
    tdf['datetime'] = [datetime.utcfromtimestamp(x) for x in tdf.timestamp]
    tdf['input_on'] = tdf['hvac_state']==INPUT_LOOKUP[mode]
    if tdf.input_on.sum() < 12*60//5:
        return None
    tdf = tdf.set_index('datetime')
    tdf_minute = tdf.resample('T').mean().interpolate('zero')

    weather_history = [v for v in db.child('weather').child(weather_key).child('history2').order_by_key().get().val().itervalues()]
    wdf = pd.DataFrame(weather_history)
    wdf['datetime'] = [datetime.utcfromtimestamp(x) for x in wdf.timestamp]
    wdf = wdf.set_index('datetime')
    wdf_minute = wdf.resample('T').mean().interpolate('zero')

    df_minute = pd.merge(tdf_minute, wdf_minute, how='left', left_index=True, right_index=True)

    freq_str = '%dT'%(MODEL_FREQ)
    df_hour = df_minute.resample(freq_str).mean()
    df_hour = df_hour.dropna()
    y = df_hour.actual_temperature_f.values.ravel()
    input_on = df_hour.input_on.values.ravel()[:-1]
    outside_temp = df_hour.temperature_f.values.ravel()[:-1]
    wind_mph = df_hour.wind_speed_mph.values.ravel()[:-1]
    res = minimize(lambda x: eval(y, (4*60)//MODEL_FREQ, input_on, outside_temp, wind_mph, x), x0, bounds=[(-0.5,0.5)]*len(x0))

    if res.success:
        model = {
            'type': CURRENT_MODEL_TYPE,
            'mode': mode,
            'coef': res.x.tolist(),
            'timestamp': int((datetime.utcnow() - datetime.utcfromtimestamp(0)).total_seconds()),
            'train_days': (tdf.index.max() - tdf.index.min()).total_seconds()/(24.0*60.0*60.0),
            'freq': MODEL_FREQ,
        }
        return model
    else:
        return None

def create_default_model(mode):
    if mode=='heat':
        coef = [DEFAULT_INPUT, DEFAULT_DELTA, DEFAULT_WIND]
    else:
        coef = [-DEFAULT_INPUT, DEFAULT_DELTA, DEFAULT_WIND]
    model = {
        'type': 'default:' + CURRENT_MODEL_TYPE,
        'mode': mode,
        'coef': coef,
        'freq': MODEL_FREQ,
    }
    return model

def get_model(db, thermostat_id, weather_key, mode):
    model = db.child('thermostats').child(thermostat_id).child('model').get().val()
    retrain = True
    existing_model = None
    if model is None:
        model = {}
    elif mode not in model:
        pass
    elif model[mode]['type'] != CURRENT_MODEL_TYPE:
        pass
    else:
        existing_model = model[mode]
        timestamp = datetime.utcfromtimestamp(existing_model['timestamp'])
        if existing_model['train_days'] < 7:
            retrain = timestamp < datetime.utcnow() - timedelta(days=1)
        else:
            retrain = timestamp < datetime.utcnow() - timedelta(days=RETRAIN_FREQ)

    if retrain:
        if existing_model is not None:
            x0 = existing_model['coef']
        elif mode=='heat':
            x0 = [DEFAULT_INPUT, DEFAULT_DELTA, DEFAULT_WIND]
        else:
            x0 = [-DEFAULT_INPUT, DEFAULT_DELTA, DEFAULT_WIND]
        new_model = train_model(db, thermostat_id, weather_key, x0, mode)
        if new_model is None:
            if existing_model is not None:
                return existing_model
            else:
                new_model = create_default_model(mode)
        model[mode] = new_model
        db.child('thermostats').child(thermostat_id).child('model').set(model)
        return new_model
    else:
        return existing_model

def eval_model_temperature_after_time(model, current_temp, input, outdoor_temp, wind_mph, minutes=4*60):
    steps = max(1,int(minutes//model['freq']))
    T = advance_model([current_temp], None, [input]*steps, [outdoor_temp]*steps, [wind_mph]*steps, model['coef'])
    return T[-1]

def eval_model_time_to_temperature(model, target_temp, current_temp, input, outdoor_temp, wind_mph):
    T = []
    steps = int((2*60)//model['freq'])
    if model['mode'] == 'heat':
        sign = 1.0 if input else -1.0
    else:
        sign = -1.0 if input else 1.0
    latest_temp = current_temp
    cnt = 0
    while sign*latest_temp < sign*target_temp:
        cnt += 1
        temp = advance_model([latest_temp], None, [input]*steps, [outdoor_temp]*steps, [wind_mph]*steps, model['coef'])
        T.extend(temp[1:])
        latest_temp = T[-1]
        if cnt >= 4:
            break
    T = [t for t in T if sign*t < sign*target_temp]
    return len(T)*model['freq']
