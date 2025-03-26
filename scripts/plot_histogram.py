# Plot a histogram of the age of the specimens

import matplotlib.pyplot as plt
import pandas as pd

# Replace 'path_to_your_file.txt' with the path to your tab-separated file
file_path = '../../adna_retrieval_conversion/ancestrymap/v62.0_1240k_public.anno'

age_col = 'Date mean in BP in years before 1950 CE [OxCal mu for a direct radiocarbon date, and average of range for a contextual date]'

# Load your data
df = pd.read_csv(file_path, sep='\t', low_memory=False)


def plot_age_histogram(age_array):
    # Filter age array to include ages up to 100,000 years
    age_array = age_array[0 < age_array]
    age_array = age_array[age_array <= 10000]

    # Create a histogram plot with 100 bins, each 1000 years wide
    plt.figure(figsize=(10, 6))
    plt.hist(age_array, bins=range(0, 10000, 100), color='skyblue', edgecolor='black')
    plt.title('Age Distribution')
    plt.xlabel('Age (years before present)')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    # Show the plot
    # plt.show()
    plt.savefig('../results/age_histogram.png')
    plt.clf()


# Call the histogram plotting function with your data
plot_age_histogram(df[age_col])
