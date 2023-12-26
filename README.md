# engr-ALDA-Fall2023-P20
**Sentiment Analysis Of Customer Reviews**

_P20: Prabhudatta Mishra, Jay Joshi, Janani Pradeep, Yi Hu_

The exploration of sentiment in user-generated content carries substantial implications for businesses and individuals. Businesses stand to gain insights into customer satisfaction, while users can enhance decision-making by considering sentiments expressed by others. Our project aspires to offer valuable insights and practical tools, contributing to navigating the expanding landscape of online reviews.



## Project Structure

### Cleaning Data

- **Script:** `cleaning.py`
  - **Description:** Cleans the raw data provided in `Data.json` and saves the cleaned data as `Data_cleaned.csv`.

### Word2Vec Implementation

- **Scripts:**
  - `word2vec XGboost.py`
    - **Description:** Implements Word2Vec with XGBoost for sentiment analysis.
  - `word2vec Logistic.py`
    - **Description:** Implements Word2Vec with logistic regression.
  - `word2vec SVM.py`
    - **Description:** Implements Word2Vec with Support Vector Machines (SVM) for sentiment analysis.

### Sentence Transformers

- **Scripts:**
  - `transformer XGboost.py`
    - **Description:** Implements sentence transformers with XGBoost.
  - `transformer Logistic.py`
    - **Description:** Implements sentence transformers with logistic regression.
  - `transformer SVM.py`
    - **Description:** Implements sentence transformers with SVM.

### TF-IDF Implementation

- **Scripts:**
  - `TFIDF XGboost.py`
    - **Description:** Implements TF-IDF with XGBoost.
  - `TFIDF SVM.py`
    - **Description:** Implements TF-IDF with SVM.
  - `TFIDF NB.py`
    - **Description:** Implements TF-IDF with Naive Bayes.

### Bag of Words (BoW) Implementation

- **Scripts:**
  - `BOG XGBOOST.py`
    - **Description:** Implements Bag of Words with XGBoost.
  - `BOG SVM.py`
    - **Description:** Implements Bag of Words with SVM.
  - `BOG Logistic.py`
    - **Description:** Implements Bag of Words with logistic regression.

### Accuracy Comparison

- **Script:** `accuracy.py`
  - **Description:** Compares the accuracy of all implemented techniques.

### Visualization

- **Script:** `draw.py`
  - **Description:** Draws accuracy comparisons for better visualization.

## Getting Started

To set up the project, install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```
