# Wrapper for the annotations/labels with name constants for interesting columns
# and methods to retrieve those columns as dictionaries mapping id to column value

import pandas as pd
import os

# Replace 'path_to_your_file.txt' with the path to your tab-separated file
file_path = '../../adna_retrieval_conversion/vcf/v62.0_1240k_public_small.anno'

age_col = 'Date mean in BP in years before 1950 CE'
lat_col = 'Lat.'
long_col = 'Long.'
id_col = 'Genetic ID'

# Load your data
script_dir = os.path.dirname(os.path.abspath(__file__))
absolute_path = os.path.join(script_dir, file_path)
df = pd.read_csv(absolute_path, sep='\t', quotechar='$', low_memory=False, on_bad_lines='warn', na_values='..')

# ID column contains comments after name
id_colname = [col for col in df.columns if col.startswith(id_col)][0]
age_colname = [col for col in df.columns if col.startswith(age_col)][0]

# Convert latitude and longitude to x and y coordinates
df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
df[long_col] = pd.to_numeric(df[long_col], errors='coerce')

ages = df[age_colname].copy()
ids = df[id_colname].copy()
lats = df[lat_col].copy()
longs = df[long_col].copy()


def get_ages():
    return dict(zip(ids, ages))


def get_locations():
    return dict(zip(ids, zip(longs, lats)))


def get_labels():
    return list(zip(ages, longs, lats))