import sklearn
from sklearn import linear_model, preprocessing
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

data = pd.read_csv("car.data")
print(data.head())
label = preprocessing.LabelEncoder()
buying = label.fit_transform(list(data["buying"]))
maint = label.fit_transform(list(data["maint"]))
door = label.fit_transform(list(data["door"]))
persons = label.fit_transform(list(data["persons"]))
lug_boot = label.fit_transform(list(data["lug_boot"]))
safety = label.fit_transform(list(data["safety"]))
Class = label.fit_transform(list(data["class"]))

prerdict = "class"

x = list(zip(buying,maint,door,persons,lug_boot,safety))
y = list(Class)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

model = KNeighborsClassifier(n_neighbors=8)

model.fit(x_train,y_train)
accuracy = model.score(x_test,y_test)
print(f"Accuracy = {accuracy*100}%")

test = 0
prediction = model.predict(x_test)
names =["Inaccurate","Accurate","Good","Excellent"]

for _ in range(len(x_test)):
    #print(f"Prediction = {prediction[_]} , Data = {x_test[_]} , Actual_Data = {y_test[_]}")
    print(f"Prediction = {names[prediction[_]]} , Data = {x_test[_]} , Actual_Data = {names[y_test[_]]}")
    #n = model.kneighbors([x_test[_]] , 9 , True)
    #print(f"N = {n}")
    
y_test_names = [names[i] for i in y_test]
pred_names = [names[i] for i in prediction]

plt.figure(figsize=(10, 6))
plt.plot(range(len(x_test)), y_test_names, label="Actual Data", marker="o")
plt.plot(range(len(x_test)), pred_names, label="Predictions", marker="x")
plt.xlabel("Car Evaluation Test Index")
plt.ylabel("Class")
plt.title("KNN Classification: Actual vs Prediction")
plt.legend()
plt.show()