import pandas as pd
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import seaborn as sns
import matplotlib.pyplot as plt
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve

def load_data():
    data = pd.read_csv("spam-ham.csv", encoding='ISO-8859-1')
    data = data.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'])
    data.columns = ['label' , 'text']
    y = data['label']
    x = data['text']
    x = pd.DataFrame(x)
    y = pd.DataFrame(y)
    return (x, y)

def clean_text(text):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    words = [stemmer.stem(w) for w in words]
    text = ' '.join(words)
    return(text)

def split_data(x, y):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size = 0.2, random_state = 42, stratify=y
    )
    return(x_train, x_test, y_train, y_test)

def set_data(x, y):
    x['text'] = x['text'].apply(clean_text)
    trans = TfidfVectorizer()
    x = trans.fit_transform(x['text'])
    y['label'] = y['label'].map({'ham':0, 'spam':1})
    y = y['label'].values
    return(x, y)

def set_model(x_train, y_train ):
    param = {
        'C':[0.1, 1, 10],
        'kernel':['linear', 'rbf'],
        'gamma':['scale', 'auto']
    }
    model = GridSearchCV(
        SVC(probability =  True, random_state = 42),
        param_grid = param,
        cv = 5,
        scoring = 'accuracy',
        n_jobs = -1
    )
    model.fit(x_train, y_train)
    return(model)

def evaluate_model(model, x_test, y_test, y_pred):
    report = classification_report(y_test, y_pred, output_dict = True)
    report = pd.DataFrame(report)
    _, axes = plt.subplots(1, 3, figsize=(12, 5))
    sns.heatmap(report.iloc[: -1 , :-2] , annot= True , fmt=".2f" , cmap="Blues" ,ax=axes[0])
    axes[0].set_title("Classification Report")
    axes[0].set_aspect(1)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Ham','Spam'])
    disp.plot(cmap='Blues', ax=axes[1])
    axes[1].set_title("Confusion Matrix")
    plt.tight_layout()
    fpr ,tpr, _ = roc_curve(y_test, model.predict_proba(x_test)[:, 1])
    axes[2].plot(fpr, tpr, label=f"SVM AUC = {roc_auc_score(y_test, model.predict_proba(x_test)[:, 1]):.2f}")
    axes[2].set_xlabel("False Positive Rate")
    axes[2].set_ylabel("True Positive Rate")
    axes[2].set_title("ROC Curve")
    axes[2].legend()
    axes[2].set_aspect(1)
    plt.show()

def main():
    x, y = load_data()
    x, y = set_data(x, y)
    x_train, x_test, y_train, y_test = split_data(x, y)
    model = set_model(x_train, y_train)
    y_pred = model.predict(x_test)
    evaluate_model(model, x_test, y_test, y_pred)

if __name__ == "__main__":
    main()