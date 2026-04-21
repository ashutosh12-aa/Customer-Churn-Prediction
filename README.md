# 📊 Customer Churn Prediction System

## 🚀 Project Overview

The **Customer Churn Prediction System** is a Machine Learning-based web application that predicts whether a customer is likely to leave (churn) a telecom service.

This project not only predicts churn but also provides **insights, visualizations, and reasons** behind the prediction, making it highly useful for business decision-making.

---

## 🎯 Objectives

* Predict customer churn using Machine Learning
* Provide a user-friendly web interface using Streamlit
* Visualize prediction results with charts and risk indicators
* Explain key factors influencing churn

---

## 🧠 Key Features

✔️ Customer churn prediction (Yes/No)
✔️ 📈 Model accuracy display
✔️ 🎯 Customer Risk Meter (visual indicator)
✔️ 📊 Churn probability pie chart
✔️ 📉 Feature importance graph
✔️ 🧠 Top 3 reasons for churn (explainable AI)
✔️ 🌐 Interactive Streamlit dashboard

---

## 🛠️ Technologies Used

| Category         | Tools                        |
| ---------------- | ---------------------------- |
| Programming      | Python                       |
| Data Analysis    | Pandas, NumPy                |
| Machine Learning | Scikit-learn (Random Forest) |
| Visualization    | Matplotlib                   |
| Web App          | Streamlit                    |

---

## 📂 Project Structure

```
Customer_Churn_Prediction/
│
├── app.py                  # Streamlit web application
├── main.py                 # Model training script
├── model.pkl               # Trained ML model
├── scaler.pkl              # Data scaler
├── accuracy.pkl            # Model accuracy
├── features.pkl            # Feature names
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset
└── README.md               # Project documentation
```

---

## 📊 Dataset Information

* Dataset: **Telco Customer Churn Dataset**
* Contains customer details like:

  * Demographics
  * Account information
  * Services subscribed
* Target variable: **Churn (Yes/No)**

---

## ⚙️ How It Works

1. Data preprocessing (handling missing values, encoding)
2. Feature scaling using StandardScaler
3. Model training using Random Forest Classifier
4. Model evaluation using accuracy score
5. Deployment using Streamlit for real-time predictions

---

## ▶️ How to Run the Project

### Step 1: Install dependencies

```
pip install pandas numpy scikit-learn streamlit matplotlib
```

### Step 2: Train the model

```
python main.py
```

### Step 3: Run the web app

```
streamlit run app.py
```

---

## 📸 Application Preview

The app includes:

* Input form for customer details
* Prediction result (High Risk / Low Risk)
* Risk meter visualization
* Feature importance graph
* Top reasons affecting churn

---

## 📈 Model Details

* Algorithm: **Random Forest Classifier**
* Evaluation Metric: **Accuracy**
* Handles both numerical and categorical data effectively

---

## 🧠 Explainability (Important Feature)

The system highlights:

* Top 3 features influencing churn
* Human-readable explanations (e.g., high charges, short tenure)

---

## 🔮 Future Improvements

* Add more ML models (XGBoost, Logistic Regression)
* Deploy on cloud (Streamlit Cloud / Heroku)
* Add real-time database integration
* Improve UI with advanced dashboards

---

## 👨‍💻 Author

**Ashutosh**
GitHub: https://github.com/ashutosh12-aa

---

## ⭐ Conclusion

This project demonstrates how Machine Learning can be used to:

* Predict customer behavior
* Improve retention strategies
* Provide actionable insights for businesses

---

⭐ *If you like this project, don’t forget to star the repository!*
