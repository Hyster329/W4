from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

#load the data set
iris = load_iris()
X = iris.data
y = iris.target

# 20% for testing and 80%for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier
# random forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

import pickle
pickle.dump(model, open('model.pickle','wb'))

from flask import Flask,jsonify,request
app=Flask(__name__)
@app.route('/',methods=['GET','POST'])
def home():
    if(request.method == 'GET'):
        data = 'Hello World!'
        return jsonify({'data':data})
@app.route('/predict/')
def class_predict():
    model=pickle.load(open('model.pickle','rb'))
    sepal_length=request.args.get("sepal_length")
    sepal_width=request.args.get('sepal_width')
    petal_length=request.args.get('petal_length')
    petal_width=request.args.get('petal_width')
    test_df=pd.DataFrame({'sepal_length':[sepal_length],'sepal_width':[sepal_width],'petal_length':[petal_length],'petal_width':[petal_width]})
    pred_class=model.predict(test_df)
    return jsonify({'Species':str(pred_class)})
if __name__=='__main__':
    app.run(debug=True)

