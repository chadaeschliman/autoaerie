import requests
from zipcode_lookup import ZipCode
from datetime import datetime, timedelta

DARK_SKY_KEY = 'ecec1d84f548d0fc058474873377a0db'

#

class DarkSky:

    def __init__(self):
        pass

    def get_weather(self, lat, lng, hours=3):
        params = {
            'exclude': 'minutely,alerts,flags',
        }
        if hours<2:
            params['exclude'] += ',hourly'
        weather_url = 'https://api.darksky.net/forecast/%s/%f,%f'%(DARK_SKY_KEY, lat, lng)
        r = requests.get(weather_url, params=params)
        if r.status_code != requests.codes.ok:
            print 'Weather error:', r.status_code
            print r.text
            return None

        res = r.json()
        result = {
            'timezone': res['timezone'],
            'sunrise': datetime.utcfromtimestamp(res['daily']['data'][0]['sunriseTime']),
            'sunset': datetime.utcfromtimestamp(res['daily']['data'][0]['sunsetTime']),
            'temperature': [res['currently']['temperature']],
            'dewPoint': [res['currently']['dewPoint']],
            'humidity': [res['currently']['humidity']],
            'cloudCover': [res['currently']['cloudCover']],
            'windSpeed': [res['currently']['windSpeed']],
        }
        for offset in xrange(1,hours):
            sub_res = res['hourly']['data'][offset]
            result['temperature'].append(sub_res['temperature'])
            result['dewPoint'].append(sub_res['dewPoint'])
            result['humidity'].append(sub_res['humidity'])
            result['cloudCover'].append(sub_res['cloudCover'])
            result['windSpeed'].append(sub_res['windSpeed'])
        return result

    def get_all(self, lat, lng):
        weather_url = 'https://api.darksky.net/forecast/%s/%f,%f'%(DARK_SKY_KEY, lat, lng)
        r = requests.get(weather_url)
        if r.status_code != requests.codes.ok:
            print 'Weather error:', r.status_code
            print r.text
            return None

        res = r.json()
        return res

    def get_historical(self, lat, lng, days=14):
        today = datetime.utcnow().replace(hour=0,minute=0,second=0,microsecond=0)
        params = {
            'exclude': 'currently,minutely,hourly,alerts,flags',
        }
        result = {
            'date': [],
            'windSpeed': [],
            'cloudCover': [],
            'humidity': [],
            'temperatureMin': [],
            'temperatureMax': [],
            'apparentTemperatureMin': [],
            'apparentTemperatureMax': [],
        }
        for day in xrange(1,days+1):
            target_date = today + timedelta(days=-day)
            weather_url = 'https://api.darksky.net/forecast/%s/%f,%f,%s'%(DARK_SKY_KEY, lat, lng, target_date.strftime('%s'))
            r = requests.get(weather_url)
            if r.status_code != requests.codes.ok:
                print 'Weather error:', r.status_code
                print r.text
                continue
            res = r.json()
            result['timezone'] = res['timezone']
            sub_res = res['daily']['data'][0]
            result['date'].append(target_date.strftime('%Y-%m-%d'))
            result['windSpeed'].append(sub_res['windSpeed'])
            result['cloudCover'].append(sub_res['cloudCover'])
            result['humidity'].append(sub_res['humidity'])
            result['temperatureMin'].append(sub_res['temperatureMin'])
            result['temperatureMax'].append(sub_res['temperatureMax'])
            result['apparentTemperatureMin'].append(sub_res['apparentTemperatureMin'])
            result['apparentTemperatureMax'].append(sub_res['apparentTemperatureMax'])
        return result
