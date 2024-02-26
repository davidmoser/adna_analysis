import pandas as pd

# Replace 'path_to_your_file.txt' with the path to your tab-separated file
file_path = '../data/aadr_v54.1.p1_1240K_public.anno'

age_col = 'Date mean in BP in years before 1950 CE [OxCal mu for a direct radiocarbon date, and average of range for a contextual date]'
lat_col = 'Lat.'
long_col = 'Long.'
id_col = 'Genetic ID'

# Load your data
df = pd.read_csv(file_path, sep='\t', quotechar='$', low_memory=True, on_bad_lines='warn')

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
    return dict(zip(ids, zip(lats, longs)))
