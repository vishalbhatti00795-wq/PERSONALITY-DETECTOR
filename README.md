# 🧠 Personality Detector

> A Machine Learning web application that predicts whether a person is **Introvert, Ambivert, or Extrovert** based on psychological, social, behavioral, and lifestyle traits.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-purple?logo=pandas)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-blue?logo=numpy)](https://numpy.org/)

---

## 📌 Overview

**Personality Detector** is a supervised Machine Learning project built using **Logistic Regression**.

The system takes numerical ratings representing different aspects of a person's behavior and personality, processes those inputs using the same scaling procedure used during model training, and predicts one of three personality categories:

* 🟣 **Introvert**
* 🔵 **Ambivert**
* 🟢 **Extrovert**

The trained Machine Learning model is integrated into an interactive **Streamlit** web application where users can rate their characteristics from **0 to 10** and receive a prediction along with probability scores.

---

## 🎯 Project Objective

The primary objective of this project is to demonstrate an end-to-end Machine Learning workflow:

1. Load and understand a dataset
2. Perform Exploratory Data Analysis
3. Check data quality
4. Encode the target variable
5. Select relevant features
6. Standardize numerical features
7. Split the dataset into training and testing sets
8. Train a Logistic Regression classifier
9. Evaluate model performance
10. Save the trained model and scaler
11. Deploy the model through a Streamlit interface

This project demonstrates how a Machine Learning model can move from **raw data → preprocessing → training → evaluation → deployment**.

---

## 📊 Dataset

The project uses a synthetic personality dataset named:

```text
personality_synthetic_dataset.csv
```

The dataset contains:

* **20,000 records**
* **30 columns**
* 1 target column
* 29 numerical feature columns

The target variable is:

```text
personality_type
```

with three possible classes:

```text
Introvert
Ambivert
Extrovert
```

The dataset contains behavioral and lifestyle attributes such as:

* Social Energy
* Alone Time Preference
* Talkativeness
* Deep Reflection
* Group Comfort
* Party Liking
* Listening Skill
* Empathy
* Creativity
* Organization
* Leadership
* Risk Taking
* Public Speaking Comfort
* Curiosity
* Routine Preference
* Excitement Seeking
* Friendliness
* Emotional Stability
* Planning
* Spontaneity
* Adventurousness
* Reading Habit
* Sports Interest
* Online Social Usage
* Travel Desire
* Gadget Usage
* Collaborative Work Style
* Decision Speed
* Stress Handling

## The notebook confirms that all 20,000 records contain non-null values and that there are no duplicate rows.

## 🔍 Exploratory Data Analysis

The notebook performs several data exploration steps, including:

* Inspecting the first few records
* Checking dataset dimensions
* Checking missing values
* Checking duplicate records
* Inspecting column names and data types
* Generating descriptive statistics
* Examining personality-class distribution
* Visualizing feature distributions

The dataset contains **29 numerical variables and 1 categorical target variable**.

---

## 🧹 Data Preprocessing

### 1. Target Encoding

Since Machine Learning algorithms require numerical target values, the `personality_type` column is converted into numerical labels using `LabelEncoder`.

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data['personality_type'] = le.fit_transform(data['personality_type'])
```

The resulting mapping used by the application is:

| Encoded Value | Personality |
| ------------: | ----------- |
|             0 | Ambivert    |
|             1 | Extrovert   |
|             2 | Introvert   |

---

### 2. Feature Selection

Three variables were intentionally excluded from the final model:

```text
creativity
emotional_stability
stress_handling
```

The target variable `personality_type` was also separated from the feature matrix.

```python
X = data.drop(
    columns=[
        'personality_type',
        'emotional_stability',
        'stress_handling',
        'creativity'
    ]
)

y = data['personality_type']
```

This results in **26 input features** being used by the trained model.

---

### 3. Feature Scaling

The numerical features are standardized using `StandardScaler`.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

Standardization transforms the features so that they have a mean close to **0** and a standard deviation of **1**.

---

## 🤖 Machine Learning Model

### Logistic Regression

The project uses **Logistic Regression** as its classification algorithm.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
```

Although its name contains "Regression", Logistic Regression is commonly used for **classification problems**.

In this project, it performs **multiclass classification** across:

```text
Ambivert
Extrovert
Introvert
```

The model is trained using the standardized feature set.

---

## 🧪 Train-Test Split

The dataset is divided into:

* **80% Training Data**
* **20% Testing Data**

using `train_test_split` with `random_state=42`.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)
```

With 20,000 records, this produces:

* **16,000 training samples**
* **4,000 testing samples**

---

## 📈 Model Performance

The trained Logistic Regression model achieved:

### Accuracy

**99.75%**

```text
Accuracy: 0.9975
```

The classification report evaluated **4,000 test samples** and showed approximately 1.00 precision, recall, and F1-score for all three encoded classes.

| Metric       |     Result |
| ------------ | ---------: |
| Accuracy     | **99.75%** |
| Test Samples |  **4,000** |
| Classes      |      **3** |
| Precision    |      ~1.00 |
| Recall       |      ~1.00 |
| F1-Score     |      ~1.00 |

> **Note:** The dataset is synthetic, so the reported performance should not be interpreted as evidence that the model can accurately determine real-world psychological personality types. The model is primarily a Machine Learning demonstration project.

---

## 🌐 Streamlit Web Application

The trained model is deployed through a custom **Streamlit** interface.

Users can rate themselves from **0 to 10** across different categories.

### Social & Communication

* Social Energy
* Talkativeness
* Group Comfort
* Party Liking
* Listening Skill
* Empathy
* Alone Time Preference
* Public Speaking Comfort
* Friendliness
* Online Social Usage
* Collaborative Work Style

### Thinking & Personality

* Deep Reflection
* Organization
* Leadership
* Curiosity
* Routine Preference
* Risk Taking
* Excitement Seeking
* Planning
* Spontaneity
* Adventurousness

### Interests & Lifestyle

* Creativity
* Reading Habit
* Sports Interest
* Travel Desire
* Gadget Usage

## The application provides an interactive interface for entering these values.

## 🔮 Prediction Pipeline

When the user clicks **"Detect My Personality"**, the application follows this pipeline:

```text
User Input
    ↓
Numerical Feature Vector
    ↓
StandardScaler
    ↓
Logistic Regression Model
    ↓
Personality Prediction
    ↓
Prediction Probabilities
    ↓
Streamlit Result
```

The input is transformed using the saved scaler before being passed to the trained model.

The application then displays:

* Predicted personality type
* Model confidence
* Probability for each personality class
* Short interpretation of the predicted personality

---

## 📊 Prediction Output

Example output:

```text
Your predicted personality type is

        Ambivert

Model confidence: 87.42%
```

The application also provides a probability breakdown:

```text
Ambivert    87.42%
Extrovert    8.31%
Introvert    4.27%
```

This allows users to see not only the predicted class but also the model's probability distribution across the three classes.

---

## 💾 Saved Model Files

The trained Machine Learning artifacts are saved using Python's `pickle` module:

```text
personality_model.pkl
scaler.pkl
```

### `personality_model.pkl`

Contains the trained Logistic Regression model.

### `scaler.pkl`

Contains the fitted `StandardScaler` used during training.

## Both files are loaded by the Streamlit application when it starts.

## 📁 Project Structure

```text
Personality-Detector/
│
├── app.py
│
├── logistic_Regression.ipynb
│
├── personality_model.pkl
│
├── scaler.pkl
│
├── personality_synthetic_dataset.csv
│
├── requirements.txt
│
└── README.md
```

### File Description

| File                                | Description                                           |
| ----------------------------------- | ----------------------------------------------------- |
| `app.py`                            | Streamlit web application                             |
| `logistic_Regression.ipynb`         | Data analysis, preprocessing, training and evaluation |
| `personality_model.pkl`             | Trained Logistic Regression model                     |
| `scaler.pkl`                        | Fitted StandardScaler                                 |
| `personality_synthetic_dataset.csv` | Dataset used for model training                       |
| `requirements.txt`                  | Python dependencies                                   |
| `README.md`                         | Project documentation                                 |

---

## ⚙️ Technologies Used

### Programming Language

* Python

### Data Analysis

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* Logistic Regression
* LabelEncoder
* StandardScaler
* Train-Test Split
* Classification Metrics

### Deployment / UI

* Streamlit

### Model Serialization

* Pickle

---

## 🚀 How to Run Locally

### 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd Personality-Detector
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Make Sure Model Files Exist

Ensure the following files are in the same directory as `app.py`:

```text
personality_model.pkl
scaler.pkl
```

The application expects these files in the same folder when loading the trained artifacts.

### 5. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🧠 What I Learned From This Project

This project helped reinforce several important Machine Learning concepts:

* Understanding classification problems
* Exploratory Data Analysis
* Data quality checking
* Target encoding
* Feature selection
* Feature scaling
* Train-test splitting
* Logistic Regression
* Multiclass classification
* Model evaluation
* Accuracy, precision, recall and F1-score
* Model serialization with Pickle
* Building a Machine Learning application with Streamlit
* Connecting a trained ML model to a user-facing interface

---

## 🔮 Future Improvements

Some potential improvements for future versions include:

* [ ] Compare Logistic Regression with Random Forest, SVM, XGBoost and other classifiers
* [ ] Perform hyperparameter tuning
* [ ] Add cross-validation
* [ ] Add feature importance / model interpretability
* [ ] Improve the personality questionnaire
* [ ] Add more detailed personality insights
* [ ] Add prediction history
* [ ] Add downloadable prediction reports
* [ ] Deploy the application publicly
* [ ] Use a larger and more representative real-world dataset
* [ ] Build an API using FastAPI or Flask
* [ ] Containerize the application using Docker

---

## ⚠️ Disclaimer

This project is intended for **educational and demonstration purposes only**.

The predictions generated by this application should **not** be considered a psychological diagnosis, professional personality assessment, or scientifically validated evaluation of an individual's personality.

The training dataset used in this project is synthetic, and therefore the model's high test accuracy does not necessarily translate to real-world personality prediction performance.

---

## ⭐ Project Highlights

* **20,000** training records
* **26** features used by the final model
* **3** personality classes
* **99.75%** test accuracy
* Interactive Streamlit interface
* Real-time prediction
* Probability-based output
* Saved and reusable ML model
* End-to-end Machine Learning workflow

---

## 👨‍💻 Author

**Vishal**

Built as a Machine Learning project to practice the complete workflow from **data analysis and model training to deployment**.

---

## ⭐ Support

If you found this project interesting or useful, consider giving the repository a **star** on GitHub.

```text
⭐ Star this repository if you found it useful!
```
