# Plot the locations of the specimens on a world map

import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.basemap import Basemap

# Replace 'path_to_your_file.txt' with the path to your tab-separated file
file_path = '../data/aadr_v54.1.p1_1240K_public.anno'

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
    plt.figure(figsize=(50, 20))
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
    m.scatter(x, y, c=ages, cmap='coolwarm', edgecolor='none', s=10,
              linewidth=0)  # Adjust 's' for size

    # Add colorbar
    plt.colorbar(label='Age (Older -> Red, Recent -> Blue)')

    plt.title('World Map with Points Colored by Age')
    # plt.show()
    plt.savefig('../results/adna_locations_on_map.png')
    plt.clf()


def plot_age_histogram(age_array):
    age_array = age_array[age_array < 10000]
    # Create a histogram plot
    plt.figure(figsize=(10, 6))
    plt.hist(age_array, bins=50, color='skyblue', edgecolor='black')
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    # Show the plot
    # plt.show()
    plt.savefig('../results/age_histogram.png')
    plt.clf()


# plot_age_histogram(df[age_col])
plot_locations(df)

