import pickle
import numpy as np
import pandas as pd

from flask import Flask
from flask import request

app = Flask(__name__)

loaded_model = pickle.load(open('churn_model.pkl', 'rb'))

@app.route('/predict_churn')
def predict_churn():
    '''
    receives inputs for a single prediction as parameters and
    predict if the user churn
    :return: class label as string (example: “0” / “1”)
    '''
    is_male = request.args.get('is_male')
    num_inters = request.args.get('num_inters')
    late_on_payment = request.args.get('late_on_payment')
    age = request.args.get('age')
    years_in_contract = request.args.get('years_in_contract')

    df = pd.DataFrame([[is_male, num_inters, late_on_payment, age, years_in_contract]], \
                      columns=['is_male', 'num_inters', 'late_on_payment', 'age', 'years_in_contract'])

    y_pred = loaded_model.predict(df)
    #print(df)
    #print(y_pred[0])
    return str(y_pred[0])


def test_func(model):
    '''
    test the model if it predicts exactly the same in comparison with the saved prdictions
    :param model
    :return: True or False
    '''
    X_test = pd.read_csv('X_test.csv')
    y_pred = np.loadtxt('preds.csv')

    y_pred_on_server = model.predict(X_test)
    print(np.array_equal(y_pred, y_pred_on_server))
    return np.array_equal(y_pred, y_pred_on_server)


if __name__ == "__main__":
    '''
    run the predicting server
    '''
    app.run(host='0.0.0.0' , port=8080)