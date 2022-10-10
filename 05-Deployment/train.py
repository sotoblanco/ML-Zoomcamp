# %%
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

# parameters

C = 1.0
n_splits = 5
output_file = f'model_C={C}.bin'

##################


df = pd.read_csv("https://raw.githubusercontent.com/alexeygrigorev/datasets/master/AER_credit_card_data.csv")


df["card"] = (df["card"]=="yes").astype('int8')
df["owner"] = (df["owner"]=="yes").astype('int8')
df["selfemp"] = (df["selfemp"]=="yes").astype('int8')


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train["card"].values
y_val = df_val["card"].values
y_test = df_test["card"].values

del df_train["card"]
del df_val["card"]

features = ["reports", "age", "income", "share", "expenditure",
 "dependents", "months", "majorcards", "active", "owner", "selfemp"]


dv = DictVectorizer(sparse=False)

train_dict = df_train[features].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
model.fit(X_train, y_train)

val_dicts = df_val[features].to_dict(orient='records')
X_val = dv.transform(val_dicts)

y_pred = model.predict_proba(X_val)[:, 1]

# Training

def train(df_train, y_train, C=1.0):
    dicts = df_train[features].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(solver='liblinear', C=C, max_iter=1000)
    model.fit(X_train, y_train)

    return dv, model

def predict(df, dv, model):
    dicts = df[features].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred

# validation

scores = []

print(f"Doing validation with C={C}")


kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)
fold = 0
for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.card.values
    y_val = df_val.card.values

    dv, model = train(df_train, y_train, C=C)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

    print('auc on fold {fold} is {auc}')
    fold += 1

print("Validation results:")
print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))

dv, model = train(df_full_train, df_full_train.card.values, C=1.0)
y_pred = predict(df_test, dv, model)

y_test = df_test.card.values
auc = roc_auc_score(y_test, y_pred)

print(f'auc={auc}')

# Save the model

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'The model is save to {output_file}')

