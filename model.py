import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.externals import joblib
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('Iris.csv')
#input faetures
X = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm',
       'PetalWidthCm']]

#output target
encoder = LabelEncoder()
y = df[['Species']]

#train-test-split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.5)

#model
model=neighbors.KNeighborsClassifier()
model.fit(X_train, y_train)
# Save the model as a pickle in a file
joblib.dump(model, 'iris_trained_model.pkl')



#load model
with open('iris_trained_model.pkl', 'rb') as f:
       model = joblib.load(f)

#prediction
print(X_test)
int_features = [6.1,3.0,4.9,1.8]
# int_features = [x for x in request.form.values()]

final_features = pd.DataFrame(int_features).T
print('df',final_features)
final_features.columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
print('df',final_features)

predictions=model.predict(final_features)
print(predictions)
# print(accuracy_score(y_test,predictions)*100)

