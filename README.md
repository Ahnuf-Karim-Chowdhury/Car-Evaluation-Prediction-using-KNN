# Car Evaluation Prediction using KNN

---

This project uses the **K-Nearest Neighbors (KNN)** algorithm to classify cars based on various features such as buying price, maintenance cost, number of doors, seating capacity, luggage space, and safety ratings. The dataset is preprocessed, and the model is trained to predict the car's classification.

---

## Approach - 01 - Training it Newly Every Iteration

--- 
The Approach - 01 is a method where a model is trained from scratch in each iteration, without retaining any learned information from previous runs. This means the model starts with random weights every time and learns as if it's encountering the data for the first time. This approach is useful for testing consistency, avoiding bias from previous training, and ensuring fair comparisons in experimental setups.

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

## KNN Classification Prediction Data Evaluation : Serialized Actual vs Prediction

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
However, Approach - 01 it can be computationally expensive since it disregards past learning and requires full retraining each time.


---

## Approach - 02 - Saving the Most Accurate Data

--- 

## KNN-Training.py

This script is responsible for **training the KNN model** and saving it for later use.

### How It Works:
1. **Data Preprocessing:**
   - The `car.data` dataset is loaded using `pandas`.
   - Categorical values (e.g., "buying," "maint," "door") are converted into numerical values using `LabelEncoder` from `sklearn.preprocessing`.
   - Features (X) are extracted by zipping relevant attributes such as "buying," "maint," and "safety."
   - Labels (Y) are assigned from the "class" column.

2. **Training the KNN Model:**
   - The data is split into training and testing sets using `train_test_split` with a test size of 10%.
   - A KNN classifier (`KNeighborsClassifier`) is initialized with `n_neighbors=8`.
   - The model is trained using the training set (`x_train` and `y_train`).

3. **Model Persistence:**
   - The trained model is serialized and saved to a file (`knn_model.pickle`) using Python's `pickle` module.
   - This ensures the model can be reused without retraining.

4. **Accuracy Calculation:**
   - The accuracy of the trained model is computed using the testing set.

## KNN Classification Prediction Graph: Actual vs Prediction

This graph provides a **visual comparison** between the actual classifications of car evaluations and the predictions made by the K-Nearest Neighbors (KNN) algorithm.

![KNN Classification: Actual vs Prediction](https://github.com/Ahnuf-Karim-Chowdhury/Car-Evaluation-Prediction-using-KNN/blob/main/Approach%20-%2002%20-%20Saving%20the%20Most%20Accurate%20Data/Result%20Predictions/Predicted%20Data%20Graphical%20Representation.png?raw=true)

---

### **How the Graph Works**
1. **X-Axis: Car Evaluation Test Index**
   - Represents individual instances or test cases in the dataset.
   - Each index corresponds to a specific car evaluation entry.

2. **Y-Axis: Class**
   - Represents the classification categories for car evaluations.
   - Classes include:
     - **Good**
     - **Inaccurate**
     - **Excellent**
     - **Accurate**

3. **Data Points:**
   - **Blue circles** represent the **actual classifications** of the cars in the dataset.
   - **Orange crosses** represent the **predicted classifications** made by the KNN model.

4. **Connections:**
   - Lines connect the data points (actual vs. predicted) for each test index.
   - Alignment between the circles and crosses indicates **accurate predictions**.
   - Discrepancies between them highlight **prediction errors** by the model.

---

### **Explanation of the Plot**
- **Purpose:** The plot evaluates the performance of the KNN model by comparing its predictions with actual classifications from the dataset.
- **Observations:**
  - A high degree of overlap between the blue circles and orange crosses signifies that the model has made correct predictions.
  - Mismatches indicate where the model's predictions differ from the actual classifications, pinpointing areas for improvement.

---

### **Graph Utility**
This visualization helps:
- Assess the overall accuracy of the KNN model.
- Identify specific test cases where the model struggles to provide correct predictions.
- Highlight trends or patterns in the dataset that might affect the model’s performance.

By analyzing this graph, developers can fine-tune the model and address discrepancies for better accuracy in future iterations.


---

## KNN-RunFile.py

This script **evaluates the saved KNN model** and visualizes its predictions.

### How It Works:
1. **Loading Data and Model:**
   - The `car.data` dataset is preprocessed in the same way as in the training script.
   - The serialized model (`knn_model.pickle`) is loaded using `pickle`.

2. **Prediction and Accuracy:**
   - The model predicts classifications based on the feature set (X).
   - Accuracy is calculated by comparing the predicted classifications with the actual values.

3. **Visualization:**
   - **Matplotlib** is used to plot a comparison between actual classifications and predicted classifications.
   - Actual values are plotted as circles (`marker="o"`), while predicted values are plotted as crosses (`marker="x"`).
   - The graph provides a clear visual representation of how well the model performs.

4. **Result Output:**
   - The script outputs detailed comparisons, including predicted and actual labels for each instance.

---

## KNN-Serialized-RunFile.py

This script is an **enhanced version** of `KNN-RunFile.py` with improved visualization and additional data transformations.

### How It Works:
1. **Class Label Mapping:**
   - A dictionary is used to map class names (`"Inaccurate", "Accurate", "Good", "Excellent"`) to numerical values (`0, 1, 2, 3`) for better visualization.

2. **Visualization Enhancements:**
   - The Y-axis uses numeric class values, with corresponding labels added as `yticks`.
   - Enhanced graph styling makes the comparison between actual and predicted values more intuitive.
   - Gridlines improve readability of the plot.

3. **Result Evaluation:**
   - Numeric and textual comparisons between actual and predicted values are displayed for easy interpretation.

---

## KNN Classification Prediction Graph: Serialized Actual vs Prediction

This graph provides a **visual comparison** between the serialized actual classifications of car evaluations and the predictions made by the K-Nearest Neighbors (KNN) algorithm.

![KNN Classification: Serialized Actual vs Prediction](https://github.com/Ahnuf-Karim-Chowdhury/Car-Evaluation-Prediction-using-KNN/blob/main/Approach%20-%2002%20-%20Saving%20the%20Most%20Accurate%20Data/Result%20Predictions/Seriialized%20Predicted%20Data%20Graphical%20Representation.png?raw=true)

---

### **How the Graph Works**
1. **X-Axis: Car Evaluation Test Index**
   - Represents the individual instances or test cases in the dataset.
   - Each index corresponds to a specific car evaluation entry.

2. **Y-Axis: Serialized Class**
   - Represents the serialized classification categories for car evaluations.
   - Classes are mapped to numerical values for better visualization:
     - `0`: **Inaccurate**
     - `1`: **Accurate**
     - `2`: **Good**
     - `3`: **Excellent**

3. **Data Points:**
   - **Blue circles** represent the **serialized actual classifications** of the cars in the dataset.
   - **Orange crosses** represent the **serialized predictions** made by the KNN model.

4. **Connections:**
   - Lines connect the data points (actual vs. predicted) for each test index.
   - Alignment between the circles and crosses indicates **accurate predictions**.
   - Discrepancies between them highlight **prediction errors** by the model.

---

### **Explanation of the Plot**
- **Purpose:** The plot evaluates the performance of the KNN model by comparing its serialized predictions with serialized actual classifications from the dataset.
- **Observations:**
  - The numerical mapping of the classifications makes it easier to identify trends and patterns.
  - Alignment between blue circles (actual) and orange crosses (predictions) signifies correct classifications by the model.
  - Deviations between the two indicate areas where the model predictions are inaccurate.

---

### **Graph Utility**
This visualization helps:
- Assess the performance and accuracy of the serialized KNN model.
- Highlight specific test cases where there is a mismatch between actual and predicted classifications.
- Identify areas where the model can be refined or improved for better predictions.

By leveraging the serialized mapping, this plot offers a **structured approach to analyzing the KNN model's prediction accuracy**, making it easier to interpret results and identify discrepancies.

---

### **Comparison with the Non-Serialized Graph**

#### **Serialized vs. Non-Serialized**
| Feature                   | Non-Serialized Graph                      | Serialized Graph                       |
|---------------------------|-------------------------------------------|----------------------------------------|
| **Y-Axis Representation** | Class labels in text format (e.g., "Good", "Inaccurate") | Numeric mapping (0: Inaccurate, 1: Accurate, etc.) |
| **Visualization Clarity** | Visual comparison of predictions and actuals, but less structured | Clear, structured numeric scale aiding analysis |
| **Interpretation**        | Requires manual matching of text labels  | Numeric serialization simplifies trend identification |
| **Graph Customization**   | Basic plot with minimal features         | Enhanced visualization with mapped classes |

#### **Why the Serialized Graph is Better**
1. **Structured Representation:**  
   The serialized graph uses numeric mappings for class labels, making it easier to identify patterns and trends.
   
2. **Enhanced Visualization:**  
   With clearer axes and mapped values, discrepancies between actual and predicted classifications are easier to spot.

3. **Simplified Analysis:**  
   Numeric serialization eliminates ambiguity when interpreting the Y-axis, making the graph more user-friendly.

4. **Improved Readability:**  
   The serialized graph includes structured ticks and labels, ensuring a cleaner visual representation, especially for large datasets.

By moving from textual class labels to serialized numeric mappings, the serialized graph offers a more **professional, structured, and insightful evaluation of the KNN model's performance**.


---

## Summary of Features

| Script                  | Purpose                          | Key Features                                                |
|-------------------------|----------------------------------|------------------------------------------------------------|
| **KNN-Training.py**      | Trains the KNN model             | Data preprocessing, model training, and saving the model.   |
| **KNN-RunFile.py**       | Evaluates the saved model        | Predictions, accuracy calculation, and basic visualization. |
| **KNN-Serialized-RunFile.py** | Enhanced model evaluation      | Visualization with numeric mapping and improved styling.    |

This project demonstrates the use of **KNN for classification**, covering the complete workflow from data preprocessing to model training, evaluation, and visualization.





