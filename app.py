import pickle
from flask import Flask,request,jsonify,render_template,redirect,url_for
import numpy as np
import pandas as pd

app=Flask(__name__)
regmodel=pickle.load(open('regmodel.pkl','rb'))
scaler=pickle.load(open('scaler.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    newdata=scaler.transform(np.array(list(data.values())).reshape(1,-1))
    print(newdata)
    output=regmodel.predict(newdata)
    print(output[0])
    return jsonify(output[0])
if __name__=='__main__':
    app.run(host='0.0.0.0',port=5000,debug=True)