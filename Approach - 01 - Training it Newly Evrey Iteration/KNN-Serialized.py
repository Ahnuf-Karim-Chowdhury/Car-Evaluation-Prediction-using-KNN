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

x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(Class)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

model = KNeighborsClassifier(n_neighbors=8)

model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)
print(f"Accuracy = {accuracy*100}%")

names = ["Inaccurate", "Accurate", "Good", "Excellent"]

label_map = {"Inaccurate": 0, "Accurate": 1, "Good": 2, "Excellent": 3}

prediction = model.predict(x_test)

for i in range(len(x_test)):
    print(f"Prediction = {names[prediction[i]]} , Data = {x_test[i]} , Actual_Data = {names[y_test[i]]}")

y_test_mapped = [label_map[names[i]] for i in y_test]
pred_mapped = [label_map[names[i]] for i in prediction]

# Plotting the data
plt.figure(figsize=(10, 6))

plt.plot(range(len(x_test)), y_test_mapped, label="Actual Data", marker="o", linestyle="--", color="blue")
plt.plot(range(len(x_test)), pred_mapped, label="Predictions", marker="x", linestyle="-", color="green")

plt.xlabel("Car Evaluation Test Index")
plt.ylabel("Class (0=Inaccurate, 1=Accurate, 2=Good, 3=Excellent)")
plt.title("KNN Classification: Actual vs Prediction")
plt.legend()
plt.yticks([0, 1, 2, 3], ["Inaccurate", "Accurate", "Good", "Excellent"])
plt.grid(True)
plt.show()
