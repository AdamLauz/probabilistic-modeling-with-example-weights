
# Probabilistic Modeling with Example Weights

This repository contains code for generating synthetic data, performing probabilistic modeling, and evaluating the model with example weights.

## Description

The code consists of several components:

1. **Data Generation**: Synthetic data is generated to simulate educational attainment based on parents' education levels and psychometric scores.

2. **Weighted Sampling**: The generated data is sampled based on example weights assigned to each data point.

3. **Model Training**: A TensorFlow model is trained using the sampled data to predict education levels based on psychometric scores and parents' education levels.

4. **Correction of Predictions**: Predictions made by the model are corrected based on the sampling probability function used initially.

## File Structure

- `prepare_dataset.py`: Python script for generating synthetic data, sampling, and preparing datasets for training.
- `prepare_model.py`: Python script for training the TensorFlow model using the prepared datasets.


## Usage

1. **Data Generation**: Run `prepare_dataset.py` to generate synthetic data and prepare datasets.

```bash
python prepare_dataset.py
```

2. **Model Training**: Run `train_model.py` to train the TensorFlow model using the prepared datasets.

```bash
python train_model.py
```

3. **Model Evaluation**: Run `evaluate_model.py` to evaluate the trained model and correct predictions based on example weights.

```bash
python evaluate_model.py
```

## Requirements

- Python 3.x
- TensorFlow
- NumPy
- Pandas
- scikit-learn

## Author

[Your Name]

---

Feel free to customize the README according to your specific project details and requirements. Let me know if you need further assistance!
