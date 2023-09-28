import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# Load the data
data = pd.read_csv('credit_risk_dataset.csv')

# Handle categorical variables using one-hot encoding
data = pd.get_dummies(data, columns=['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file'], drop_first=True)

# Split data into features and target
X = data.drop('loan_status', axis=1)
y = data['loan_status']

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# GUI setup
window = tk.Tk()
window.title("Credit Risk Assessment")

# Function to train the model
def train_model():
    global X, y, model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    messagebox.showinfo("Training Completed", "Model has been trained successfully.")

# Function to show feature importance
def show_feature_importance():
    global model
    importances = model.feature_importances_
    features = data.drop('loan_status', axis=1).columns
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    messagebox.showinfo("Feature Importance", importance_df.to_string())

# Function to predict test data
def predict_test_data():
    try:
        income = float(entry_income.get())
        loan_amnt = float(entry_loan_amnt.get())

        # Check if income is greater than loan amount
        if income > loan_amnt:
            result_label.config(text="Predicted Loan Status: YES")
        else:
            result_label.config(text="Predicted Loan Status: NO")

    except ValueError as e:
        result_label.config(text=f"Error: {e}")


def plot_graphs():
    global X, y, model

    # Split the data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Create bar charts
    metrics = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
    values = [accuracy, f1, precision, recall]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(metrics, values, color=['blue', 'green', 'red', 'purple'])
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Values')
    ax.set_title('Model Evaluation Metrics')

    # Embed the chart in the GUI
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().pack()

# Train Model Button
train_button = ttk.Button(window, text="Train Model", command=train_model)
train_button.pack()

plot_button = ttk.Button(window, text="Plot Evaluation Metrics", command=plot_graphs)
plot_button.pack()


# Show Feature Importance Button
feature_button = ttk.Button(window, text="Show Feature Importance", command=show_feature_importance)
feature_button.pack()

# Test Data Entry Fields
label_age = ttk.Label(window, text="Age:")
label_age.pack()
entry_age = ttk.Entry(window)
entry_age.pack()

label_income = ttk.Label(window, text="Income:")
label_income.pack()
entry_income = ttk.Entry(window)
entry_income.pack()

label_emp_length = ttk.Label(window, text="Employment Length:")
label_emp_length.pack()
entry_emp_length = ttk.Entry(window)
entry_emp_length.pack()

label_loan_amnt = ttk.Label(window, text="Loan Amount:")
label_loan_amnt.pack()
entry_loan_amnt = ttk.Entry(window)
entry_loan_amnt.pack()

# Add similar entries for home ownership, loan intent, loan grade, and default
label_home_ownership = ttk.Label(window, text="Home Ownership:")
label_home_ownership.pack()
combo_home_ownership = ttk.Combobox(window, values=['RENT', 'OWN', 'OTHER'])
combo_home_ownership.pack()

label_intent = ttk.Label(window, text="Loan Intent:")
label_intent.pack()
combo_intent = ttk.Combobox(window, values=['VENTURE', 'EDUCATION', 'MEDICAL', 'PERSONAL', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'])
combo_intent.pack()

label_grade = ttk.Label(window, text="Loan Grade:")
label_grade.pack()
combo_grade = ttk.Combobox(window, values=['A', 'B', 'C', 'D', 'E', 'F', 'G'])
combo_grade.pack()

label_default = ttk.Label(window, text="Default on File:")
label_default.pack()
combo_default = ttk.Combobox(window, values=['Y', 'N'])
combo_default.pack()

# Predict Button
predict_button = ttk.Button(window, text="Predict", command=predict_test_data)
predict_button.pack()

# Result Label
result_label = ttk.Label(window, text="")
result_label.pack()

window.mainloop()
