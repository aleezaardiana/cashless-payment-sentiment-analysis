# cashless-payment-sentiment-analysis
Sentiment analysis and predictive modeling of public opinions on cashless payment systems in Indonesia using Multinomial Naive Bayes and SVM.

# Cashless Payment Sentiment Analysis & Prediction

## Project Overview
This project focuses on analyzing public sentiment toward cashless payment systems in Indonesia using data collected from X (Twitter). It applies Natural Language Processing (NLP) and Machine Learning techniques to classify public opinions into Positive, Neutral, and Negative categories.

The project also compares the performance of Multinomial Naive Bayes and Support Vector Machine (SVM) and implements the best-performing model into an interactive Streamlit application.

## Objectives
- Analyze public opinions toward cashless payment systems in Indonesia.
- Classify opinions into Positive, Neutral, and Negative sentiments.
- Compare the performance of Multinomial Naive Bayes and SVM.
- Identify public sentiment distribution and frequently discussed words.
- Deploy the best-performing model into an interactive web application.

## Dataset
- Source: X (Twitter)
- Data collection: Web scraping using Apify
- Total collected tweets: 8,685
- Topic: Cashless payment systems in Indonesia
- Sentiment classes: Positive, Neutral, and Negative

## Tools & Technologies
- Python
- Pandas
- NLTK
- Scikit-learn
- IndoBERT
- TF-IDF
- SMOTE
- Streamlit
- Apify

## Methodology
1. Data Collection
2. Data Cleaning and Preprocessing
3. Sentiment Labeling using IndoBERT
4. Train-Test Split (80:20)
5. Feature Extraction using TF-IDF
6. Class Imbalance Handling using SMOTE
7. Model Training (Multinomial Naive Bayes & SVM)
8. Model Evaluation
9. Streamlit Deployment

## Model Performance

| Evaluation Metric | Naive Bayes | SVM |
|---|---:|---:|
| Accuracy | 72.87% | 78.57% |
| Precision | 76.86% | 77.44% |
| Recall | 72.87% | 78.57% |
| F1-Score | 74.04% | 76.72% |

SVM achieved the best overall performance with 78.57% accuracy and 76.72% F1-score.

## Key Findings
- Neutral sentiment dominated public opinions.
- Positive and Negative sentiments had relatively balanced proportions.
- SVM outperformed Multinomial Naive Bayes based on weighted evaluation metrics.
- The best-performing model was deployed using Streamlit.

## Streamlit Application
The application consists of five main pages:
- About System
- Upload Dataset
- Statistics & Model Evaluation
- Visualization (EDA)
- Try It Now (Real-Time Prediction)

## Project Structure
- `app/` – Streamlit application
- `notebooks/` – Data analysis and modeling notebooks
- `src/` – Supporting Python scripts
- `results/` – Model evaluation and analysis outputs
- `screenshots/` – Application screenshots

## Author
[Your Name]

**Field:** Data Analytics | Machine Learning | Natural Language Processing
