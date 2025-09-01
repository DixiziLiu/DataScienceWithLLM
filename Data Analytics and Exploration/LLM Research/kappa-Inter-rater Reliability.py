#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 11:30:56 2024

Cohen's Kappa Inter-rater Reliability'

@author: dixiziliu
"""

from sklearn.metrics import cohen_kappa_score
import pandas as pd

# Define the set of labels from both rater 1 and 2
labels = [
    'Recognition: Positive', 
    'Proficiency: Positive', 
    'Courtesy/Respect: Positive', 
    'Involvement: Positive', 
    'Wait time: Positive', 
    'Presence: Positive', 
    'Guest Accommodation: Positive'
]

# Rater 1's labels
rater1 = {
    'Recognition: Positive': 1,
    'Proficiency: Positive': 1,
    'Courtesy/Respect: Positive': 1,
    'Involvement: Positive': 1,
    'Wait time: Positive': 1,
    'Presence: Positive': 0,
    'Guest Accommodation: Positive': 0
}

# Rater 2's labels
rater2 = {
    'Recognition: Positive': 0,
    'Proficiency: Positive': 0,
    'Courtesy/Respect: Positive': 0,
    'Involvement: Positive': 0,
    'Wait time: Positive': 0,
    'Presence: Positive': 1,
    'Guest Accommodation: Positive': 1
}

# Create dataframes for rater labels
rater1_labels = pd.Series(rater1, index=labels)
rater2_labels = pd.Series(rater2, index=labels)

# Calculate Cohen's Kappa
kappa_score = cohen_kappa_score(rater1_labels, rater2_labels)

print(f"Cohen's Kappa: {kappa_score}")















## Next Comment ##

# Define the set of possible labels including only the "Negative" categories
labels = [
    'Recognition: Negative', 
    'Proficiency: Negative', 
    'Sad: Negative'
]

# Rater 1's labels (with negative sentiments)
rater1 = {
    'Recognition: Negative': 1,
    'Proficiency: Negative': 1,
    'Sad: Negative': 1
}

# Rater 2's labels (no labels assigned)
rater2 = {
    'Recognition: Negative': 0,
    'Proficiency: Negative': 0,
    'Sad: Negative': 0
}

# Create dataframes for rater labels
rater1_labels = pd.Series(rater1, index=labels)
rater2_labels = pd.Series(rater2, index=labels)

# Calculate Cohen's Kappa
kappa_score = cohen_kappa_score(rater1_labels, rater2_labels)

print(f"Cohen's Kappa: {kappa_score}")


## Next Comment ##

# Define the set of actual labels from both raters
labels = [
    'Recognition: Positive', 
    'Proficiency: Positive', 
    'Nurse Nurse Aide-Courtesy/Respect: Positive', 
    'Responsiveness: Positive', 
    'Involvement: Positive', 
    'Nurse Nurse Aide-Emotional Support: Positive', 
    'Guest Accommodation: Positive', 
    'Communication: Positive'
]

# Rater 1's labels (assign 1 for labels used by rater 1, 0 otherwise)
rater1 = {
    'Recognition: Positive': 1,
    'Proficiency: Positive': 1,
    'Nurse Nurse Aide-Courtesy/Respect: Positive': 1,
    'Responsiveness: Positive': 1,
    'Involvement: Positive': 1,
    'Nurse Nurse Aide-Emotional Support: Positive': 1,
    'Guest Accommodation: Positive': 0,
    'Communication: Positive': 0
}

# Rater 2's labels (assign 1 for labels used by rater 2, 0 otherwise)
rater2 = {
    'Recognition: Positive': 0,
    'Proficiency: Positive': 0,
    'Nurse Nurse Aide-Courtesy/Respect: Positive': 0,
    'Responsiveness: Positive': 0,
    'Involvement: Positive': 0,
    'Nurse Nurse Aide-Emotional Support: Positive': 0,
    'Guest Accommodation: Positive': 1,
    'Communication: Positive': 1
}

# Create dataframes for rater labels
rater1_labels = pd.Series(rater1, index=labels)
rater2_labels = pd.Series(rater2, index=labels)

# Calculate Cohen's Kappa
kappa_score = cohen_kappa_score(rater1_labels, rater2_labels)

print(f"Cohen's Kappa: {kappa_score}")


## Next One ##

# Define the set of actual labels from both raters (positive and negative labels)
labels = [
    'Medication: Negative',
    'Communication: Negative',
    'Nurse Nurse Aide-Emotional Support: Positive',
    'Discharge and Follow Up: Negative',
    'Doctor-Communication: Positive',
    'Nurse Nurse Aide-Communication: Positive',
    'Proficiency: Negative',
    'Wait Time: Negative'
]

# Rater 1's labels (assign 1 for labels used by rater 1, 0 otherwise)
rater1_labels = {
    'Medication: Negative': 1,
    'Communication: Negative': 1,
    'Nurse Nurse Aide-Emotional Support: Positive': 1,
    'Discharge and Follow Up: Negative': 1,
    'Doctor-Communication: Positive': 1,
    'Nurse Nurse Aide-Communication: Positive': 1,
    'Proficiency: Negative': 1,
    'Wait Time: Negative': 1
}

# Rater 2's labels (assign 1 for labels used by rater 2, 0 otherwise)
rater2_labels = {
    'Medication: Negative': 1,
    'Communication: Negative': 1,
    'Nurse Nurse Aide-Emotional Support: Positive': 0,
    'Discharge and Follow Up: Negative': 0,
    'Doctor-Communication: Positive': 0,
    'Nurse Nurse Aide-Communication: Positive': 0,
    'Proficiency: Negative': 0,
    'Wait Time: Negative': 1
}

# Create dataframes for rater labels
rater1_series = pd.Series(rater1_labels, index=labels)
rater2_series = pd.Series(rater2_labels, index=labels)

# Calculate Cohen's Kappa
kappa_score = cohen_kappa_score(rater1_series, rater2_series)

print(f"Cohen's Kappa: {kappa_score}")



### Another method for another comment ###

# Step 1: Extract the labels from both raters
rater1_labels = ["Discharge and Follow Up: Negative", "Amenities: Negative", "Guest Accommodation: Negative"]
rater2_labels = ["discharge and follow up: negative", "presence: negative", "guest accommodation: negative"]

# Step 2: Normalize the labels by converting to lowercase and fixing potential spelling variations
rater1_labels = [label.lower() for label in rater1_labels]
rater2_labels = [label.lower() for label in rater2_labels]

# Step 3: Define a common set of labels (as given by the user)
common_labels = ["discharge and follow up: negative", "amenities: negative", "guest accommodation: negative", "presence: negative"]

# Step 4: Create binary agreement matrix
def create_binary_vector(labels, common_labels):
    # Create a binary vector, 1 if the label is present, 0 otherwise
    return [1 if label in labels else 0 for label in common_labels]

rater1_vector = create_binary_vector(rater1_labels, common_labels)
rater2_vector = create_binary_vector(rater2_labels, common_labels)

# Step 5: Calculate Cohen's Kappa Score
kappa_score = cohen_kappa_score(rater1_vector, rater2_vector)

print(f"Cohen's Kappa Score: {kappa_score}")



### More efficient method

# Labels from Rater 1 and Rater 2 as seen in the image
rater1_labels = ["Amenities: Positive", "Recognition: Positive", "Collective Team-Communication: Positive", 
                 "Collective Team-Courtesy/Respect: Positive", "Proficiency: Positive", "Testing: Positive"]

rater2_labels = ["courtesty/respect: positive", "testing: positive", "commication: positive"]

# Step 2: Normalize the labels by converting to lowercase and fixing potential spelling variations
rater1_labels = [label.lower() for label in rater1_labels]
rater2_labels = [label.lower() for label in rater2_labels]

# Extract the unique set of labels from both raters
all_labels = list(set(rater1_labels + rater2_labels))

# Create the agreement matrix (1 if the label is present, 0 if not)
rater1_binary = [1 if label in rater1_labels else 0 for label in all_labels]
rater2_binary = [1 if label in rater2_labels else 0 for label in all_labels]

# Calculate Cohen's Kappa
kappa = cohen_kappa_score(rater1_binary, rater2_binary)
print(f"Cohen's Kappa score: {kappa}")





