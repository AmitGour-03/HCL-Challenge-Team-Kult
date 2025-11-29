## HCL-Challenge-Team-Kult
The repository will have the project implementation

Usecase: Customer Churn Prediction

**Problem Statement**
Predict whether a customer will churn based on usage pattern, demographics, and engagement

# Project Overview:
This project predicts whether a bank customer will churn (leave the bank).

It uses machine learning and deep learning to analyze customer behavior.

The aim is to help the bank identify customers at risk and take preventive actions.

The model is trained on historical data and predicts churn probability for new customers.

# Libraries and Frameworks Used:
**TensorFlow / Keras** – for building and training the neural network model.

**Scikit-Learn** – for preprocessing, encoding, scaling, and calculating evaluation metrics.

**Pandas** – used for data handling, cleaning, and DataFrame operations.

**NumPy** – used for numerical computations and array manipulation.

**Matplotlib** – for visualizations for understanding data patterns.

**Streamlit** – used for building an interactive web app for model prediction.

**IPykernel** – enables running Jupyter notebooks smoothly and managing virtual environments.


# Data Description:
Dataset contains 10,000 rows and 11 columns.

Key columns include:
CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts,
HasCrCard, IsActiveMember, EstimatedSalary, and the target column Exited.

The target variable `Exited` represents whether the customer churned (1 = yes, 0 = no).

Data consists of customer demographics, account details, and activity patterns.


# Data Preprocessing and Feature Engineering: - Amit Gour
Categorical features (Geography, Gender) converted using OneHotEncoder and Label Encoder respectively.

Numerical columns scaled using StandardScaler for uniform value ranges.

Dataset divided into training, validation, and testing subsets.

Checked for missing values, duplicates, and inconsistencies or irrelevant feature removal before training.

Saved preprocessing files (scaler.pkl and onehot_encoder_geo.pkl and label_encoder_gender.pkl) for future predictions so that we don't need to run it again from start just do the prediction from trained model.


# Model: - K Sri Varshini
Model is a Deep Neural Network - (Artificial NEural Network) built using TensorFlow/Keras.

Uses multiple Dense layers with ReLU activation for feature learning.

Last layer uses Sigmoid activation to output churn probability.

Optimized using Adam optimizer and trained using Binary Crossentropy loss.


# Training: - Atishay Jain
Training performed on the preprocessed training dataset.

Weights updated through backpropagation for multiple epochs.

Training accuracy and loss monitored to ensure learning consistency.

Encoders and scalers applied to maintain proper input format.


# Validation:
Validation dataset used during training to measure performance of model after training.

Helps to compare training vs validation performance to avoid overfitting.

Hyperparameters such as epochs, batch size, and learning rate adjusted based on results.

# Testing
Final model evaluation done using a separate unseen test dataset.

Testing accuracy represents the real-world performance.

Ensures the model’s predictions are not biased toward training data.

# Evaluation Metrics - Faiyaz 
Accuracy: percentage of correct predictions.

Precision: how many predicted-churn customers truly churned.

Recall: how many actual churn customers the model captured.

F1 Score: harmonic mean of precision and recall (balanced measure).

# Output:
Model outputs a probability between 0 and 1.

Values near 1 = high chance the customer will churn.

Values near 0 = customer will likely stay.

Predictions can be done using real time UI by Streamlit web app.


# Challenges:
Handling the mix of categorical and numerical features.

Avoiding model overfitting when training deep networks.

Ensuring data scaling/encoding is consistent during prediction.

Dealing with class imbalance between churned and non-churned customers.

Hyperparameter tuning required multiple experiments.


# Deliverables: - Sreevas TM
**Trained model**:
model.h5 – trained deep learning model.

**Trained Dumped pickle files for new data input**
scaler.pkl, onehot_encoder_geo.pkl, label_encoder_gender.pkl

**Preprocessed and cleaned dataset**
Jupyter Notebook with complete workflow (uses IPykernel).

**User Interface by Streamlit** 
UI by Streamlit web app framework for real-time prediction.
