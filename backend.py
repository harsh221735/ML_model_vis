from flask import Flask, request, jsonify
#from networkx import display
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Gradient Descent Implementation
class gradient_descent:
    def __init__(self,x_train,y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.B = np.zeros((x_train.shape[1],1))
    def fit(self,epochs,lr,no_of_batch):                        #we getting best accuracy at this value of lr
        num = self.x_train.shape[0] // no_of_batch
        slopes_array = []
        for j in range(epochs):
            val1 = 0
            for i in range(no_of_batch):   # it was rigth
                val2 = val1 + num 
                batch_x = self.x_train[val1:val2]   #here i only did for x , i have to do also for y
                batch_y = self.y_train[val1:val2]
                val1 = val2 
                #here batch becomes std_xtrain
                slope = (np.dot(np.dot(batch_x.T, batch_x), self.B) - np.dot(batch_x.T, batch_y))/num
                self.B = self.B - lr*slope
                slopes_array.append(self.B)
        return self.B,slopes_array
    def predict(self,x_data):
        y_pred = np.dot(x_data,self.B)    #just interchanging position of two matrices
        return (y_pred)


GLOBAL_DF = None


@app.route('/dataset', methods=['POST'])
def upload_dataset():
    global GLOBAL_DF

    mode = request.form.get("mode", "default")

    if mode == "default":
        X, y = make_regression(
            n_samples=100,
            n_features=1,
            noise=20,
            bias=50,
            random_state=42
        )
        GLOBAL_DF = pd.DataFrame({
            "input": X.flatten(),
            "output": y
        })
    else:
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file uploaded"}), 400

        GLOBAL_DF = pd.read_csv(file)
    if GLOBAL_DF.shape[1] < 2 or GLOBAL_DF.shape[1] > 2:
        return jsonify({"message": "Dataset must have at least two columns"}), 400
    else:
    # Handle missing values
        for col in GLOBAL_DF.columns:
            if GLOBAL_DF[col].isnull().sum() > 0:
                if GLOBAL_DF[col].dtype in ['float64', 'int64']:
                    GLOBAL_DF[col].fillna(GLOBAL_DF[col].mean(), inplace=True)
                else:
                    GLOBAL_DF[col].fillna(GLOBAL_DF[col].mode()[0], inplace=True)

        return jsonify({
            "message": "Dataset loaded successfully",
            "shape": GLOBAL_DF.shape
        }), 200

#  Handle missing values    
'''for i in GLOBAL_DF.columns:
    if GLOBAL_DF[i].isnull().sum() > 0:
        if GLOBAL_DF[i].dtype == 'float64' or GLOBAL_DF[i].dtype == 'int64':
            GLOBAL_DF[i] = GLOBAL_DF[i].fillna(np.mean(GLOBAL_DF[i]))
        else:  
            GLOBAL_DF[i] = GLOBAL_DF[i].fillna(GLOBAL_DF[i].mode().iloc[0])'''


@app.route('/train', methods=['POST'])
def train():
    global GLOBAL_DF
    # splitting, reshaping and adding 1 for intercept in X
    x_train,x_test,y_train,y_test = train_test_split(GLOBAL_DF.iloc[:,0],GLOBAL_DF.iloc[:,1])
    x_train = x_train.values.reshape(-1,1)
    x_test = x_test.values.reshape(-1,1)
    y_train = y_train.values.reshape(-1,1)
    y_test = y_test.values.reshape(-1,1)
    x_train = np.insert(x_train,0,1,axis = 1)
    x_test = np.insert(x_test,0,1,axis = 1)
    

    if GLOBAL_DF is None:
        return jsonify({"error": "Dataset not loaded"}), 400

    data = request.json   # ✅ ONLY hyperparameters

    mode = data.get("mode", "batch")

    gd = gradient_descent(x_train,y_train)
    if mode == 'batch':
        lr = float(data.get("lr", 0.01))
        epochs = int(data.get("epochs", 10))
        #batch_size = int(data.get("batch_size", 1))
        B,slopes_array = gd.fit(epochs,lr,1)
        y_pred = gd.predict(x_test)
    elif mode == 'mini-batch':
        lr = float(data.get("lr", 0.01))
        epochs = int(data.get("epochs", 10))
        batch_size = int(data.get("batch_size", 5))
        B,slopes_array = gd.fit(epochs,lr,batch_size)
        y_pred = gd.predict(x_test)
    elif mode == 'stochastic':
        lr = float(data.get("lr", 0.01))
        epochs = int(data.get("epochs", 10))
        #batch_size = int(data.get("batch_size", 1))
        B,slopes_array = gd.fit(epochs,lr,x_train.shape[0])
        y_pred = gd.predict(x_test)
    return jsonify({
        "message": "Model trained successfully",
        "coefficients": B.flatten().tolist(),
        "slopes_array": [s.flatten().tolist() for s in slopes_array]
    }), 200
if __name__ == '__main__':
    app.run(debug=True,port=5001)

    