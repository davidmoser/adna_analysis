# Wrapper for the annotations/labels with name constants for interesting columns
# and methods to retrieve those columns as dictionaries mapping id to column value

import pandas as pd
import os

# Replace 'path_to_your_file.txt' with the path to your tab-separated file
file_path = '../data/aadr_v54.1.p1_1240K_public.anno'

age_col = 'Date mean in BP in years before 1950 CE [OxCal mu for a direct radiocarbon date, and average of range for a contextual date]'
lat_col = 'Lat.'
long_col = 'Long.'
id_col = 'Genetic ID'

# Load your data
script_dir = os.path.dirname(os.path.abspath(__file__))
absolute_path = os.path.join(script_dir, file_path)
df = pd.read_csv(absolute_path, sep='\t', quotechar='$', low_memory=False, on_bad_lines='warn', na_values='..')

# Convert latitude and longitude to x and y coordinates
df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
df[long_col] = pd.to_numeric(df[long_col], errors='coerce')

ages = df[age_col].copy()
ids = df[id_col].copy()
lats = df[lat_col].copy()
longs = df[long_col].copy()


def get_ages():
    return dict(zip(ids, ages))


def get_locations():
    return dict(zip(ids, zip(longs, lats)))


def get_labels():
    return list(zip(ages, longs, lats))