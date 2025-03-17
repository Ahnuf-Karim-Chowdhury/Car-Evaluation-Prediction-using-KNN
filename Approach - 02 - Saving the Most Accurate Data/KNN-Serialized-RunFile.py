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

# Convert class labels to numeric values for plotting
class_mapping = {"Inaccurate": 0, "Accurate": 1, "Good": 2, "Excellent": 3}

y_test_numeric = [class_mapping[names[i]] for i in y]
pred_numeric = [class_mapping[names[i]] for i in prediction]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(range(len(x)), y_test_numeric, label="Actual Data", marker="o")
plt.plot(range(len(x)), pred_numeric, label="Predictions", marker="x")
plt.xlabel("Car Evaluation Test Index")
plt.ylabel("Class (0: Inaccurate, 1: Accurate, 2: Good, 3: Excellent)")
plt.title("KNN Classification: Actual vs Prediction")
plt.yticks([0, 1, 2, 3], ["Inaccurate", "Accurate", "Good", "Excellent"])
plt.legend()
plt.show()
