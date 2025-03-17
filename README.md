# Car Evaluation Prediction using KNN

---

This project uses the **K-Nearest Neighbors (KNN)** algorithm to classify cars based on various features such as buying price, maintenance cost, number of doors, seating capacity, luggage space, and safety ratings. The dataset is preprocessed, and the model is trained to predict the car's classification.

---

## Approach - 01 - Training it Newly Every Iteration

--- 

## KNN.py aka Normal KNN Prediction Data Evaluation Overview

### **1. Importing Libraries**
- **scikit-learn**: Used for machine learning functionalities like KNN classification and data preprocessing.
- **pandas & numpy**: Used for data handling and numerical operations.
- **matplotlib**: Used to visualize predictions.

### **2. Data Preprocessing**
- The dataset (`car.data`) is read using **pandas**.
- Categorical values are converted into numerical values using `LabelEncoder`.
- Features (X) and target labels (Y) are extracted.
- Data is split into training and testing sets using `train_test_split`.

### **3. Model Training**
- **KNeighborsClassifier** with `n_neighbors=8` is used to train the model.
- The model is fit using training data (`x_train, y_train`).
- Accuracy of the model is calculated and printed.

### **4. Predictions and Visualization**
- The model makes predictions on `x_test`.
- Predictions are compared with actual values and printed in human-readable form.
- A plot is generated to compare actual vs predicted classifications.

## KNN Classification Prediction Data Evaluation : Actual vs Prediction

![KNN Classification: Actual vs Prediction](https://github.com/Ahnuf-Karim-Chowdhury/Car-Evaluation-Prediction-using-KNN/blob/main/Approach%20-%2001%20-%20Training%20it%20Newly%20Evrey%20Iteration/Result%20Prediction/Prediction%20Data%20Evaluation.png?raw=true)

This visualization represents the **comparison between actual car evaluation classifications and the predictions made by the K-Nearest Neighbors (KNN) algorithm**.

### **Explanation of the Plot**
- **X-Axis:** Represents the test index of the car evaluation dataset.
- **Y-Axis:** Represents the classification labels of the cars, such as `Good`, `Inaccurate`, `Excellent`, and `Accurate`.
- The **blue dots** represent the actual classifications.
- The **orange crosses and lines** represent the predictions made by the KNN model.
- The lines connecting the points indicate differences between actual and predicted classifications.

### **How It Works**
1. **Data Preprocessing:**  
   - The dataset is preprocessed, converting categorical values into numerical values.
   - Features and labels are extracted and split into training and testing datasets.

2. **Model Training & Prediction:**  
   - The KNN model is trained using `n_neighbors=8`.
   - The trained model makes predictions on the test dataset.

3. **Visualization:**  
   - The actual and predicted values are plotted to visually assess the performance of the model.
   - Differences between actual and predicted classifications highlight the model's accuracy.

This graph helps to **identify mismatches between actual and predicted classifications**, allowing us to evaluate the effectiveness of the KNN model.


---

## KNN-Serialized.py aka Serialized KNN Prediction Data Evaluation Overview

This script is an optimized version of `KNN.py` with additional enhancements.

### **1. Serialization and Mapping**
- Class names (`Inaccurate`, `Accurate`, `Good`, `Excellent`) are mapped to numerical values for better visualization.
- A dictionary `label_map` is used to maintain class mappings.

### **2. Model Training and Evaluation**
- The training process remains the same as `KNN.py`.
- Predictions are converted into readable class names.

### **3. Improved Visualization**
- `plt.plot` is used with different colors and line styles to improve clarity.
- Y-axis labels are mapped to class names for better readability.
- Grid lines are added to enhance visualization.

## KNN Classification Serialized Prediction Data Evaluation : Actual vs Prediction

![KNN Classification: Actual vs Prediction](https://github.com/Ahnuf-Karim-Chowdhury/Car-Evaluation-Prediction-using-KNN/blob/main/Approach%20-%2001%20-%20Training%20it%20Newly%20Evrey%20Iteration/Result%20Prediction/Serialized%20Prediction%20Data%20Evaluation.png?raw=true)

This visualization represents the **comparison between actual car evaluation classifications and the predictions made by the K-Nearest Neighbors (KNN) algorithm**.

### **Explanation of the Plot**
- **X-Axis:** Represents the test index of the car evaluation dataset.
- **Y-Axis:** Represents the classification labels of the cars, such as `Good`, `Inaccurate`, `Excellent`, and `Accurate`.
- The **blue dots** represent the actual classifications.
- The **green crosses and lines** represent the predictions made by the KNN model.
- The lines connecting the points indicate differences between actual and predicted classifications.

### **How It Works**
1. **Dataset Preparation:**  
   - The dataset is preprocessed, and categorical features are converted into numerical values.  
   - This ensures the data is compatible with the KNN algorithm.

2. **Model Training and Prediction:**  
   - The KNN model is trained using a predefined number of neighbors (`n_neighbors=8`).  
   - Predictions are made on the test dataset.

3. **Result Evaluation:**  
   - Predictions are compared with actual classifications to determine accuracy.  
   - Any discrepancies are noted and analyzed.

4. **Visualization:**  
   - Actual values are plotted as **blue dots**, representing the correct classifications.  
   - Predicted values are shown as **green crosses**, with lines connecting them to the actual values.  
   - This helps in identifying where predictions align or differ from the actual classifications.

This graph serves as a **visual representation of the model's performance**, highlighting areas where the KNN algorithm performs well and where it could potentially be improved.


---

## Key Differences Between `Normal-KNN(KNN.py)` and `Serialized-KNN(KNN-Serialized.py)`
| Feature | `KNN.py` | `KNN-Serialized.py` |
|---------|---------|---------------------|
| Prediction Format | Uses list of names for display | Uses a dictionary for mapping class names |
| Visualization | Basic line plot | Enhanced plot with colors, labels, and grid |
| Serialization | No explicit mapping | Uses `label_map` for structured mapping |

Both scripts efficiently classify cars based on KNN, with `KNN-Serialized.py` offering a more structured approach and improved visualization.
