Artificial Neural Network for Customer Churn Prediction

This project uses an Artificial Neural Network (ANN) to predict customer churn using a bank dataset (`Churn_Modelling.csv`). The model is implemented using TensorFlow/Keras in Python via a Jupyter notebook.

Dataset

The dataset contains information on customer demographics, bank balance, activity, and whether they exited the bank or not.

 File: Churn_Modelling.csv
 Target: Exited (0 = stayed, 1 = churned)
 Features include:
   CreditScore
   Geography
   Gender
   Age
   Tenure
   Balance
   NumOfProducts
   HasCrCard
   IsActiveMember
   EstimatedSalary

Technologies Used

 Python
 NumPy, Pandas
 TensorFlow & Keras
 Scikit-learn
 Matplotlib & Seaborn
 Jupyter Notebook

Files

ANN.ipynb: Jupyter notebook containing preprocessing, model creation, training, and evaluation.
Churn_Modelling.csv: Dataset used for training/testing.

## ⚙️ How to Run

1. Clone this repo:
bash
   git clone https://github.com/your-username/ann-churn-prediction.git
   cd ann-churn-prediction

2. Open Jupyter Notebook:
bash
   jupyter notebook
   
3. Run the `ANN.ipynb` notebook step-by-step.

Model Summary

* Input: 11 features (after encoding categorical data)
* Hidden Layers: 2 Dense layers with ReLU
* Output Layer: 1 neuron with sigmoid (binary classification)
* Optimizer: Adam
* Loss: Binary Crossentropy
* Metrics: Accuracy

Evaluation

* The model is evaluated using a confusion matrix and accuracy score.
* Training history (loss and accuracy) is visualized for performance tracking.
