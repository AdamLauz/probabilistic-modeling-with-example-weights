import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
N = 30000
data = pd.DataFrame({
    'country': np.random.choice(['US', 'Germany', 'France', 'India'], size=N),
    'parents_education': np.random.choice(['no degree', 'undergraduate', 'graduate', 'phd'], size=N)
})

# Encode categorical variables
label_encoders = {}
for column in ['parents_education']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le


# Define parents_factor and psycho_factor
def parents_factor(parents_education):
    return (parents_education + 1) * 0.5


def psycho_factor(psychometric_score):
    return psychometric_score / 100


# Generate education_level based on parents_factor and psycho_factor
education_levels = ['no degree', 'undergraduate', 'graduate', 'phd']

mean_scores = [50, 60, 70, 80]  # Mean psychometric score for each education level
std_devs = [10, 8, 6, 4]  # Standard deviation for each education level


def assign_psycho_score(row):
    # Get the index of the education level
    level_index = row['parents_education']

    # Sample psychometric score from Gaussian distribution
    score = np.random.normal(mean_scores[level_index], std_devs[level_index])

    # Ensure the score is within the range of 0 to 100
    score = max(0, min(score, 100))

    return score


def assign_education_level(row):
    p_factor = parents_factor(row['parents_education'])
    ps_factor = psycho_factor(row['psychometric_score'])
    combined_factor = p_factor * ps_factor

    probabilities = np.exp(combined_factor) / np.sum(np.exp(combined_factor))

    # Sample education level based on probabilities
    education_level = np.random.choice(education_levels, p=probabilities)

    return education_level


data['psychometric_score'] = data.apply(assign_psycho_score, axis=1)
data['education_level'] = data.apply(assign_education_level, axis=1)
data['education_level'] = label_encoders['parents_education'].fit_transform(data['education_level'])

# Save LabelEncoders
with open('./datasets/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)


# Define weight function (example: based on country)
def weight_function(row):
    if row['country'] == 'US':
        return 0.6
    elif row['country'] == 'Germany':
        return 0.3
    elif row['country'] == 'France':
        return 0.5
    else:
        return 0.4


# Sample the data based on weights
def sample_data(data):
    sampled_data = data.sample(frac=1, weights='weight', random_state=42)
    return sampled_data


if __name__ == "__main__":
    # Apply weight function
    data['weight'] = data.apply(weight_function, axis=1)

    # Normalize weights so the minimum weight is 1
    min_weight = data['weight'].min()
    data['weight'] = data['weight'] / min_weight

    # Display the first few rows of the dataset
    print(data.head())

    sampled_data = sample_data(data)

    # Separate features and target
    X = sampled_data[['psychometric_score', 'parents_education']]
    y = sampled_data['education_level']
    weights = sampled_data['weight']

    # Create directory if it does not exist
    os.makedirs('./datasets', exist_ok=True)

    # Save the datasets to CSV files
    X.to_csv('./datasets/X.csv', index=False)
    y.to_csv('./datasets/y.csv', index=False)
    weights.to_csv('./datasets/weights.csv', index=False)

