import csv
from matplotlib import pyplot as plt

FEATURE_TO_PLOT = 'Route Type' # Change to plot a different feature
FILE_PATH = '/media/mroglan/TOSHIBA EXT/Crash_Reporting_-_Drivers_Data.csv' # Your file path
LABEL_ROTATION = 45 # Rotation of the labels on the graphs (so they don't overlap)

class FrequencyFinder:

    def __init__(self, feature=FEATURE_TO_PLOT, file_path=FILE_PATH):

        self.frequencies = dict()

        with open(file_path) as file:
            reader = csv.DictReader(file)

            for row in reader:
                prop = row[feature] 
                if prop in self.frequencies:
                    self.frequencies[prop] += 1
                else:
                    self.frequencies[prop] = 0
    
    def plot(self, rotation=LABEL_ROTATION):
        plt.bar(self.frequencies.keys(), self.frequencies.values())
        plt.xticks(rotation=rotation)
        plt.tight_layout()
        plt.show()
    

if __name__ == '__main__':
    
    ff = FrequencyFinder()
    ff.plot()