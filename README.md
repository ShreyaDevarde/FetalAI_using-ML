# FetalAI: Using Machine Learning to Predict and Monitor Fetal Health

## 📌 Project Overview
FetalAI is a Machine Learning + Flask web application designed to predict fetal health status using cardiotocography (CTG) features.

The model classifies fetal health into:

- Normal (1)
- Suspect (2)
- Pathological (3)

This system helps support:

- Early intervention
- Remote monitoring
- Risk prediction during pregnancy
---
## 🚀 Features

- Fetal health prediction using Machine Learning
- Multiple model comparison
- Imbalance handling using SMOTE
- Random Forest best model selection
- Flask web integration
- User-friendly prediction interface
---
## 🧠 Tech Stack

### Machine Learning
- Python
- NumPy
- Pandas
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib
- Seaborn

### Web
- Flask
- HTML
- CSS
---
## 📂 Project Structure

```text
fetal_health/
│
├── flask/
│   ├── static/
│   │   └── style.css
│   │
│   ├── templates/
│   │   ├── index.html
│   │   ├── inspect.html
│   │   └── output.html
│   │
│   ├── app.py
│   ├── fetal_health.pkl
│
├── |--datadet/
        |--fetal_health.csv
├── fetal_health.ipynb
└── README.md
```
---
## 📊 Dataset
Source:
Kaggle Fetal Health Classification Dataset
https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification
---
## ⚙️ Workflow

### 1. Data Collection
- Load fetal_health.csv

### 2. Data Preparation
- Check missing values
- Handle imbalance using SMOTE
- Feature scaling using MinMaxScaler

### 3. Visual Analysis
- Histograms
- Scatter plots
- Correlation heatmap

### 4. Feature Selection

Selected Features:

- 'accelerations',
- 'prolongued_decelerations',
- 'abnormal_short_term_variability',
- 'percentage_of_time_with_abnormal_long_term_variability',
- 'mean_value_of_long_term_variability',
- 'histogram_mode',
- 'histogram_median',
- 'histogram_variance',
---
## 🤖 Models Trained
- Random Forest Classifier
- Decision Tree Classifier
- Logistic Regression
- K Neighbors Classifier
---

## 📈 Model Performance

| Model | Accuracy |
|-------|----------|
| Random Forest | 94.98% |
| Decision Tree | 90.75% |
| KNN | 87.14% |
| Logistic Regression | 80.56% |

Best Model:
Random Forest Classifier
---
## 💾 Save Model

```python
pickle.dump(RF_model, open("fetal_health_final.pkl","wb"))
```
## ▶️ Run Flask App
### Install dependencies
```bash
pip install flask pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
```
### Run application
```bash
python app.py
```
Open:
```text
http://127.0.0.1:5000
```
## 🖥 Web Flow
User Input
↓
Model Prediction
↓
Prediction Output

## Sample Predictions

- Normal
- Suspect
- Pathological

## Future Improvements

- Real-time fetal monitoring
- Cloud deployment
- Confidence probability scores
- Patient history database
- Deep learning integration

## 👩‍💻 Author
Developed as part of FetalAI Machine Learning Project.
