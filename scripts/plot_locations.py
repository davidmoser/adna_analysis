import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.basemap import Basemap

# Replace 'path_to_your_file.txt' with the path to your tab-separated file
file_path = '../../adna_retrieval_conversion/ancestrymap/v62.0_1240k_public.anno'

age_col = 'Date mean in BP in years before 1950 CE [OxCal mu for a direct radiocarbon date, and average of range for a contextual date]'

# Load your data
df = pd.read_csv(file_path, sep='\t', low_memory=False)

# Convert latitude and longitude to x and y coordinates
df['Lat.'] = pd.to_numeric(df['Lat.'], errors='coerce')
df['Long.'] = pd.to_numeric(df['Long.'], errors='coerce')


def plot_locations(df):
    # Determine the oldest age for normalization
    oldest_age = 10000  # df[age_col].max()

    # Normalize ages to range between 0 (most recent) and 1 (oldest)
    ages = df[age_col]
    ages[ages > oldest_age] = oldest_age

    # Initialize Basemap
    plt.figure(figsize=(10, 7))
    m = Basemap(projection='merc', llcrnrlat=-60, urcrnrlat=70, llcrnrlon=-180, urcrnrlon=180, lat_ts=20,
                resolution='c')
    m.drawcoastlines()
    m.drawcountries()
    m.fillcontinents(color='lightgray', lake_color='lightblue')

    # Convert latitude and longitude to x and y coordinates
    x, y = m(df['Long.'].values, df['Lat.'].values)
    print(f"Lowest longitude {min(df['Long.'].values)}, highest longitude {max(df['Long.'].values)}")
    print(f"Lowest latitude {min(df['Lat.'].values)}, highest latitude {max(df['Lat.'].values)}")

    # Plot points with colormap, smaller size, and without edges
    scatter = m.scatter(x, y, c=ages, cmap='coolwarm', edgecolor='none', s=10, linewidth=0)  # Adjust 's' for size

    # Add colorbar with equal height
    cbar = plt.colorbar(scatter, fraction=0.05, pad=0.05, aspect=7)
    cbar.set_label('Sample Age')

    plt.title('Distribution of Samples')
    plt.savefig('../results/adna_locations_on_map_small.png')
    plt.clf()

plot_locations(df)
