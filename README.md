# NLP with Disaster Tweets - Kaggle Classification Challenge

This repository contains my solution for the [NLP with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started/data) Kaggle competition.  
The objective is to classify tweets into disaster-related (`1`) or not (`0`) — a binary text classification task.

---

## 📁 Project Structure

```

project-root/


├── 📄 Task.ipynb                # Main notebook: full ML pipeline from preprocessing to submission.
├── 📊 train.csv                 # Training dataset (tweets + target)
├── 🧪 test.csv                  # Test dataset (tweets without target)
├── 📝 sample_submission.csv     # Template for Kaggle submission.
└── 🚀 submission1.csv           # Submission from TfidfVectorizer + XGBClassifier
└── 🚀 submission2.csv           # Submission from TfidfVectorizer + LogisticRegression
└── 📜 README.md                 # Project documentation.

```


---

## 💻 Technologies Used

- **Language**: Python 3.x  
- **Environment**: Jupyter Notebook


## 📦 Libraries

- `pandas`, `numpy` – Data manipulation  
- `matplotlib`, `seaborn` – Visualization  
- `scikit-learn` – Preprocessing & modeling (`LogisticRegression`, `LabelEncoder`, `classification_report` and `TfidfVectorizer` - for text vectorization)  
- `GridSearchCV` – Hyperparameter tuning  
- `xgboost` – Gradient boosting (`XGBClassifier`)

---

## 📊 Dataset Description

Competition: [NLP with Disaster Tweets – Kaggle](https://www.kaggle.com/competitions/nlp-getting-started/data)

- `train.csv`:  Training data including id, keyword, location, text, and the target label.
- `test.csv`: Test data with the same columns as the training data, but without the target column.
- `sample_submission.csv`: Format required for submitting predictions

---

## 🔁 Workflow Summary

The `Task.ipynb` notebook implements the complete ML pipeline:

### 1. 📥 Data Loading & Preprocessing
- Loaded `train.csv` and `test.csv`.
- EDA ` Check class distribution, look for missing values, explore some tweets, visualize word frequencies.
- Feature Engineering` Convert text into numerical features using TfidfVectorizer.


### 2. 🧠 Modeling & Tuning

Three classification models were built and tuned using `TfidfVectorizer`:

| Model                  | Method                  | Output File       |
|------------------------|-------------------------|-------------------|
| `XGBClassifier`        | Pipeline + GridSearchCV | `submission1.csv` |
| `LogisticRegression`   | Pipeline + GridSearchCV | `submission2.csv` |


### 3. 📤 Prediction & Submission

Final predictions for each model were saved:

- `submission1.csv` → XGBClassifier
- `submission2.csv` → LogisticRegression

---

## 📈 Results Summary

| Model                  | Accuracy |    Output File    |
|------------------------|----------|-------------------|
| XGBClassifier          | 0.87     | `submission1.csv` |
| LogisticRegression     | 0.80     | `submission2.csv` |

---

## ⚙️ Installation

To install all required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost




