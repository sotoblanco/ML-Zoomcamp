import pickle

model_file = 'model1.bin'
dv_file = 'dv.bin'

with open(model_file, 'rb') as model_in:
    model = pickle.load(model_in)

with open(dv_file, 'rb') as dv_in:
    dv = pickle.load(dv_in)

def predict(customer):

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0,1]
    card = y_pred >= 0.5

    result = {
        'card_probability': float(y_pred),
        'card': bool(card)
    }

    return result

if __name__=="__main__":
    print(predict({"reports": 0, "share": 0.001694, "expenditure": 0.12, "owner": "yes"}))
    