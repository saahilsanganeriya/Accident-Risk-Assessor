import pandas as pd
import numpy as np
import random

# Load the dataset
data = pd.read_csv("normalized_numerical_data.csv")

# Calculate the number of negative samples to generate
num_positive_samples = len(data)
num_negative_samples = num_positive_samples * 3

# Create a container for the negative samples
negative_samples = []

# Set random seed for reproducibility
random.seed(42)

print("Generating negative samples...")

# Generate negative samples
while len(negative_samples) < num_negative_samples:
    print(len(negative_samples))
    # Randomly select two accident records
    random_samples = data.sample(n=2)
    # Use the first sample accident record to use as a base for an altered negative sample
    positive_sample = random_samples.iloc[0]
    # Use the second sample accident record to use its data to replace data of positive_sample
    new_data = random_samples.iloc[1]

    # Randomly alter: the road segment, the hour of the day, or the day of the year
    feature_choice = random.choice(["location", "time", "date"])

    if feature_choice == "location":
        positive_sample["Latitude"] = new_data["Latitude"]
        positive_sample["Longitude"] = new_data["Longitude"]
    elif feature_choice == "time":
        positive_sample["time of day in minutes"] = new_data["time of day in minutes"]
    elif feature_choice == "date":
        positive_sample["day of the week"] = new_data["day of the week"]
        positive_sample["day of the year"] = new_data["day of the year"]
        positive_sample["week of the year"] = new_data["week of the year"]
        positive_sample["month"] = new_data["month"]
        positive_sample["year"] = new_data["year"]


    # Check if the new sample isn't within the accident records
    if not ((data["Latitude"] == positive_sample["Latitude"]) &
            (data["Longitude"] == positive_sample["Longitude"]) &
            (data["time of day in minutes"] == positive_sample["time of day in minutes"]) &
            (data["day of the week"] == positive_sample["day of the week"]) &
            (data["day of the year"] == positive_sample["day of the year"]) &
            (data["week of the year"] == positive_sample["week of the year"]) &
            (data["month"] == positive_sample["month"]) &
            (data["year"] == positive_sample["year"])).any():
        # Add to the list of negative samples
        positive_sample['accident binary'] = 0
        negative_sample = positive_sample.copy()
        negative_samples.append(negative_sample)

print("Done generating negative samples.")
print(negative_samples)
# Convert the list of negative samples into a DataFrame
negative_samples_df = pd.DataFrame(negative_samples)

# Combine the original dataset with the generated non-accident data
balanced_data = pd.concat([negative_samples_df, data], ignore_index=True)
balanced_data.sort_values(by=['year', 'day of the year', 'time of day in minutes'], inplace=True)
# Save the balanced data to a new CSV file
balanced_data.to_csv("balanced_data.csv", index=False)
