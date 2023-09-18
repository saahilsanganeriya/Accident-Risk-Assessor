import csv
from matplotlib import pyplot as plt

FEATURE_TO_PLOT = 'Vehicle Model' # Change to plot a different feature
FEATURE_TO_COMPARE = 'Injury Severity' # Change to the feature you want to compare to
FIELD_TO_COMPARE = 'POSSIBLE INJURY' # Change to the field you want in the feature
FILE_PATH = 'C:/Users/Eric Lim/Desktop/CS-4641-Project/data/output.csv' # Your file path
LABEL_ROTATION = 85 # Rotation of the labels on the graphs (so they don't overlap)

# This is a slight change to Manuels bar graph finder, so that you can plot based on another given field.
# It is quite scuffed but use it if it helps

class FrequencyFinder:

    def __init__(self, feature=FEATURE_TO_PLOT, file_path=FILE_PATH, compare_field=FEATURE_TO_COMPARE, compare_feature=FIELD_TO_COMPARE):

        self.frequencies = dict()

        with open(file_path) as file:
            reader = csv.DictReader(file)

            for row in reader:
                prop = row[feature] 
                compare = row[compare_field]
                if compare not in 'Fill field': # change this to whatever field in feature you want to get
                    if prop in self.frequencies:
                        self.frequencies[prop] += 1
                    else:
                        self.frequencies[prop] = 0
    
    def plot(self, rotation=LABEL_ROTATION):
        filtered_frequencies = {key: value for key, value in self.frequencies.items() if value > 1000} # This determines the minimum amount of datapoints needed to be plotted

        sorted_frequencies = dict(sorted(filtered_frequencies.items(), key=lambda item: item[0][-4:])) # Sorts data points by number/letter
        sorted_keys = sorted(sorted_frequencies.keys())
        sorted_values = [sorted_frequencies[k] for k in sorted_keys]

        plt.bar(sorted_keys, sorted_values)
        plt.xticks(rotation=rotation)
        plt.tight_layout()
        plt.show()
  

if __name__ == '__main__':
    ff = FrequencyFinder()
    ff.plot()