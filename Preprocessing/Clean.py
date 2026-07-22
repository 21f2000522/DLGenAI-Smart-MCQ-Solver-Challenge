import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

train = pd.read_csv("/kaggle/input/competitions/smart-mcq-solver-challenge/train.csv")
test = pd.read_csv("/kaggle/input/competitions/smart-mcq-solver-challenge/test.csv")

train_text = (
    train["prompt"].fillna("") + " " +
    train["A"].fillna("") + " " +
    train["B"].fillna("") + " " +
    train["C"].fillna("") + " " +
    train["D"].fillna("") + " " +
    train["E"].fillna("")
)

test_text = (
    test["prompt"].fillna("") + " " +
    test["A"].fillna("") + " " +
    test["B"].fillna("") + " " +
    test["C"].fillna("") + " " +
    test["D"].fillna("") + " " +
    test["E"].fillna("")
)
