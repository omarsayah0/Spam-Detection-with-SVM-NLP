# SMS Spam Detection using Support Vector Machine (SVM)

## About
This project implements a Support Vector Machine (SVM) classifier to distinguish between spam and ham (non spam) text messages.
The dataset is preprocessed, vectorized using TF-IDF, and optimized through GridSearchCV to achieve the best hyperparameters for accurate classification.

---

## Files
- `support-vector-machine.py` → Python script implementing the model.
- `spam-ham.csv` → Dataset contains a collection of SMS messages labeled as either spam or ham

---

## Steps Included

### 1️⃣ Data Preprocessing
- Loaded the dataset spam-ham.csv using pandas.
- Dropped unnecessary columns (Unnamed: 2, Unnamed: 3, Unnamed: 4).
- Renamed columns to:
- label → message type (ham or spam)
- text → message content
- Text Cleaning Steps, each message is processed through:
  -   Converting to lowercase.
  -   Removing all non-alphabetical characters using `re`.
  -   Removing English stopwords using `stopwords` from `nltk`.
  -   Applying Porter Stemming to normalize words (e.g., “running” → “run”) using `PorterStemmer` from `nltk`
- Applied TF-IDF Vectorization using `TfidfVectorizer` from `sklearn` to transform cleaned text into numerical features.

---

### 2️⃣ Model Training
- Built an **Support Vector Classifier** model using `svm` from `sklearn`.  
- Performed hyperparameter tuning with GridSearchCV to identify the optimal parameters for each model.
  
1- For the Decision Tree, tuned parameters such as criterion, max_depth, and min_samples_split.

2- For the Random Forest, optimized parameters including n_estimators, max_depth, criterion, and min_samples_split.

---


## How to Run

1- Install Dependencies:
  ```bash
pip install pandas nltk seaborn matplotlib scikit-learn
```
Also make sure to download NLTK stopwords:
 ```python
import nltk
nltk.download('stopwords')
  ```

2-Run :

  ```bash
python support-vector-machine.py
```

3- Model Evaluation :

<p align="center">
<img width="1860" height="718" alt="image" src="https://github.com/user-attachments/assets/852c496a-2975-487a-89b2-441ba8945dca" />
</p>

- The figure above shows three key evaluation metrics for the Support Vector Machine (SVM) classifier used in the spam detection task:

  1️⃣ Classification Report (Left)

      Displays precision, recall, and F1-score for both classes — ham (0) and spam (1).

      The model achieves very strong results across all metrics, with an overall accuracy of 98%.

      Precision of 0.99 for spam messages means that almost all emails predicted as spam were actually spam.

      Recall of 1.00 for ham messages indicates the model correctly identified nearly all legitimate (non-spam) emails.

  2️⃣ Confusion Matrix (Middle)

      Shows the actual versus predicted labels.

      Out of 1115 total messages,

      965 ham messages were correctly classified,

      130 spam messages were correctly detected,

      Only 20 total misclassifications occurred (1 ham mislabeled as spam, and 19 spam mislabeled as ham).

      This confirms that the SVM model rarely confuses spam with non-spam emails.

  3️⃣ ROC Curve (Right)

      The Receiver Operating Characteristic (ROC) curve illustrates the model’s ability to separate the two classes.

      The Area Under the Curve (AUC) = 0.98, indicating excellent discriminative performance, the model can almost perfectly distinguish between spam and ham messages.



 ## Author
  
  Omar Alethamat

  LinkedIn : https://www.linkedin.com/in/omar-alethamat-8a4757314/

  ## License

  This project is licensed under the MIT License — feel free to use, modify, and share with attribution.
