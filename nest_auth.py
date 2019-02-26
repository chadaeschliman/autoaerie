import requests
from datetime import datetime, timedelta

TIME_BUFFER = 1*60*60

class NestAuth:
    def __init__(self):
        self.token = 'c.jfspuYCTgHLjjbtvgXSo3EZZPGZU1wnvucfNfwnWSP0o0N03g0j3HkFNWLRZTtgsb4IG8E6KdCyL49zCtLKjLrhhX09OLl0EJ3wSLE1pWsVZWqGFOyzX9N8JJLL4tQ5D2cdu5pGRVKv6edpt'
        self.expires = datetime(2028, 11, 13, 18, 17, 11, 625661)

    def __call__(self, r):
        r.headers['Authorization'] = 'Bearer ' + self.get_token()
        return r

    def _update_token(self):
        url = "https://api.home.nest.com/oauth2/access_token"
        payload = {
            'client_id': '4d8f6eaa-0c2d-420a-bf95-2941565d116e',
            'client_secret': 'x0btoxgJAwmvDoIP9NQSqHrKL',
            'grant_type': 'authorization_code',
            'code': '2D7HFUTK',
        }
        headers = {
            'Content-Type': "application/x-www-form-urlencoded",
            'Cache-Control': "no-cache",
            }
        r = requests.request("POST", url, data=payload, headers=headers)
        if r.status_code==requests.codes.ok:
            res = r.json()
            self.token = res['access_token']
            self.expires = datetime.utcnow() + timedelta(seconds=res['expires_in'])
        else:
            print r.text

    def get_token(self):
        if self.token is None or (self.expires - datetime.utcnow()).total_seconds() < TIME_BUFFER:
            self._update_token()
        return self.token
