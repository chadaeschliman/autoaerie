import requests
import pandas as pd
import os
import feather

class ZipCode:
    def __init__(self):
        # from here: http://federalgovernmentzipcodes.us/
        dir = os.path.dirname(os.path.realpath(__file__))
        # self.zip_df = pd.read_csv(os.path.join(dir, 'free-zipcode-database-Primary.csv')).set_index('Zipcode')
        self.zip_df = feather.read_dataframe(os.path.join(dir, 'zipcodes.feather')).set_index('Zipcode')

    def get_lat_long(self, zipcode):
        entry = self.zip_df.loc[int(zipcode)]
        return entry.Lat, entry.Long
