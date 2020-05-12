import pandas as pd
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)
model = joblib.load(open('iris_trained_model.pkl', 'rb'))
# knn_from_joblib = joblib.load('filename.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    # final_features = [np.array(int_features)]
    final_features = pd.DataFrame(int_features).T
    final_features.columns = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
    # SepalLengthCm
    # SepalWidthCm
    # PetalLengthCm
    # PetalWidthCm
    print('df',final_features)
    prediction = model.predict(final_features)
    output = prediction[0]

    return render_template('index.html', prediction_text=' Iris Flower category is {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
