# SONAR Rock vs Mine Classification

A machine learning project that classifies sonar signals as **Rock** or **Mine** using **Logistic Regression**.  
The main focus of this repository is the **machine learning workflow**: data processing, model training, evaluation, prediction, and model packaging for practical use.

A lightweight **Streamlit dashboard** is included only to demonstrate the trained model in an interactive way.

---

## Project Focus

This project is primarily about **machine learning**, not frontend development.

It demonstrates:
- supervised classification using sonar signal data
- preprocessing and dataset analysis with Pandas
- training a **Logistic Regression** model with scikit-learn
- evaluating training and test performance
- building a predictive system for new sonar inputs
- exporting the trained model with Joblib
- using the saved model in a simple Streamlit app

---

## Problem Statement

The goal is to classify sonar returns into one of two categories:

- **R** → Rock
- **M** → Mine

Each sample contains **60 numeric sonar feature values**, and the model learns patterns that help distinguish rocks from mines.

This is a classic binary classification problem and is a strong beginner-to-intermediate machine learning portfolio project because it shows:
- structured tabular data handling
- train/test splitting
- model training and evaluation
- prediction on unseen data
- converting notebook-style work into a usable application

---

## Machine Learning Workflow

### 1. Importing Dependencies

```python
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

Purpose of the main libraries:
- **NumPy** → array handling
- **Pandas** → data loading and preprocessing
- **Joblib** → saving and loading the trained model
- **train_test_split** → splitting data into training and testing sets
- **LogisticRegression** → classification model
- **accuracy_score** → evaluating model performance

---

### 2. Data Collection and Data Processing

The sonar dataset is loaded into a Pandas DataFrame:

```python
sonar_data = pd.read_csv('/content/Copy of sonar data.csv', header=None)
```

Typical analysis steps used:

```python
sonar_data.head()
sonar_data.shape
sonar_data.describe()
sonar_data[60].value_counts()
sonar_data.groupby(60).mean()
```

These steps help understand:
- dataset size
- feature distribution
- class balance
- statistical properties
- average feature differences between rock and mine samples

---

### 3. Separating Features and Labels

```python
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]
```

- `X` contains the 60 sonar features
- `Y` contains the target labels: `R` or `M`

---

### 4. Training and Test Split

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.1, stratify=Y, random_state=1
)
```

Why this matters:
- **90%** of the data is used for training
- **10%** is used for testing
- `stratify=Y` keeps the class distribution balanced in both sets
- `random_state=1` makes results reproducible

---

### 5. Model Training

The model used is **Logistic Regression**:

```python
model = LogisticRegression()
model.fit(X_train, Y_train)
```

Why Logistic Regression?
- simple and effective baseline for binary classification
- easy to interpret
- widely used in machine learning practice
- good choice for a structured tabular dataset like sonar signals

---

### 6. Model Evaluation

Training accuracy:

```python
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print("Accuracy on training data: ", training_data_accuracy)
```

Testing accuracy:

```python
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print("Accuracy on test data: ", test_data_accuracy)
```

This evaluates:
- how well the model fits the training data
- how well it generalizes to unseen data


---

### 7. Making a Predictive System

A sample prediction pipeline:

```python
input_data = (
    0.0131,0.0387,0.0329,0.0078,0.0721,0.1341,0.1626,0.1902,0.2610,0.3193,
    0.3468,0.3738,0.3055,0.1926,0.1385,0.2122,0.2758,0.4576,0.6487,0.7154,
    0.8010,0.7924,0.8793,1.0000,0.9865,0.9474,0.9474,0.9315,0.8326,0.6213,
    0.3772,0.2822,0.2042,0.2190,0.2223,0.1327,0.0521,0.0618,0.1416,0.1460,
    0.0846,0.1055,0.1639,0.1916,0.2085,0.2335,0.1964,0.1300,0.0633,0.0183,
    0.0137,0.0150,0.0076,0.0032,0.0037,0.0071,0.0040,0.0009,0.0015,0.0085
)

input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if prediction[0] == 'R':
    print("The object is a Rock")
else:
    print("The object is a Mine")
```

This step shows how the trained model can be used to classify a new sonar observation.

---

### 8. Saving the Trained Model

```python
joblib.dump(model, 'model.pkl')
```

The trained model is saved as `model.pkl` and later used in the dashboard application.

This is an important portfolio detail because it shows model persistence and real-world usability.

---

## Included Dashboard

The repository also contains a **Streamlit dashboard** that loads the saved model and allows users to:
- enter 60 sonar values
- run predictions interactively
- view confidence scores
- see history and analytics
- view centered images for rock and mine results

This interface is included as a **demonstration layer** for the trained ML model.

---

## Tech Stack

### Machine Learning
- **Python**
- **NumPy**
- **Pandas**
- **scikit-learn**
- **Joblib**

### App / Visualization
- **Streamlit**
- **Plotly**

---

## Project Structure

```bash
Sonar rock vs mine/
│
├── assets/
│   ├── logo.png
│   ├── mine.jpg
│   └── rock.jpg
│
├── frontend/
│   └── sonar_ai_dashboard.py
│
├── ml/
│   └── model.pkl
│
├── requirements.txt
└── README.md
```

---

## Key Skills Demonstrated

This project highlights the following skills:

- binary classification
- machine learning model training
- train/test split methodology
- feature-label separation
- model evaluation using accuracy
- predictive system design
- model serialization with Joblib
- Python-based ML application building
- translating notebook work into a usable project

---

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Create a virtual environment

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit dashboard

```bash
streamlit run frontend/sonar_ai_dashboard.py
```

## Author

**Nasiha**  
Computer Science / Machine Learning / Python Developer

GitHub: `https://github.com/Nasiha-MH`  
LinkedIn: `https://www.linkedin.com/in/nasiha-mh/`

---

## License

This project is for educational and portfolio use.
