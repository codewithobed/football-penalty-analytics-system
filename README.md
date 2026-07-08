# Football Penalty Analytics System

A Streamlit-based machine learning dashboard for predicting football penalty kick outcomes and supporting coach/analyst decision-making.

## Features

- Upload CSV or Excel football penalty datasets
- Clean and inspect penalty data
- Explore football-specific visualisations
- Train and compare machine learning models
- Evaluate performance with accuracy, precision, recall, F1-score, ROC-AUC and confusion matrix
- Predict penalty outcome as Goal or Miss
- Display prediction confidence score
- Explain model decisions using feature importance and local explanation
- Download model comparison and prediction reports

## Project Structure

```text
football-penalty-analytics-system/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── eda.py
│   ├── training.py
│   ├── evaluation.py
│   └── prediction.py
├── data/
├── models/
├── reports/
├── docs/
├── notebooks/
└── images/
```

## How to Run

```powershell
pip install -r requirements.txt
streamlit run app.py
```

## Dissertation Context

This dashboard forms the software artefact for an MSc project on predicting football penalty kick outcomes using machine learning for sports performance analysis.
