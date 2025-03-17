import pickle
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing

data = pd.read_csv("car.data")
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

# Load the trained model
with open("knn_model.pickle", "rb") as f:
    model = pickle.load(f)

accuracy = model.score(x, y)
print(f"Accuracy = {accuracy*100}%")

prediction = model.predict(x)
names = ["Inaccurate", "Accurate", "Good", "Excellent"]

for _ in range(len(x)):
    print(f"Prediction = {names[prediction[_]]} , Data = {x[_]} , Actual_Data = {names[y[_]]}")

y_test_names = [names[i] for i in y]
pred_names = [names[i] for i in prediction]

plt.figure(figsize=(10, 6))
plt.plot(range(len(x)), y_test_names, label="Actual Data", marker="o")
plt.plot(range(len(x)), pred_names, label="Predictions", marker="x")
plt.xlabel("Car Evaluation Test Index")
plt.ylabel("Class")
plt.title("KNN Classification: Actual vs Prediction")
plt.legend()
plt.show()
