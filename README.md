# 💰 AI Salary Predictor

An interactive web application that predicts salaries based on user input such as job title, experience, education, location, skills, and more. Built with Streamlit and trained using machine learning models on real-world salary data.

---

## 🚀 Features

- **Salary Prediction**: Predict your salary based on selected profile attributes and skills.
- **Career Growth Projection**: Visualize how your salary might grow over time with new skills and roles.
- **Top Paying Roles**: Explore the highest paying job titles and salary trends by company size.
- **Confidence Visualization**: View prediction confidence intervals and peer comparisons.

---

## 🧠 Machine Learning

- Uses models like **Random Forest Regressor** and **Gradient Boosting Regressor**.
- Data is preprocessed with pipelines including scaling, one-hot encoding, and skill-based feature engineering.
- Trained on structured salary data.

---

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Backend/Modeling**: Scikit-learn, Pandas, NumPy
- **Visualization**: Plotly
- **Serialization**: Joblib

---

## 📁 Project Structure

```
├── data/
│   └── salary_data.csv          
├── models/
│   ├── column_order.json        
│   └── salary_predictor.pkl     
├── scripts/
│   ├── app.py                   
│   ├── generate_data.py         
│   └── train_model.py           
├── README.md
├── requirements.txt
```

---

## ⚙️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/KrrishTayal/Employee-Salary-Prediction.git
cd Employee-Salary-Prediction
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the app:**
```bash
streamlit run scripts/app.py
```

---

## 📝 Usage

1. Open the app in your browser (usually at `http://localhost:8501`).
2. Navigate between pages using the sidebar:
   - "Salary Prediction" for instant estimation.
   - "Career Growth Projection" to see long-term projections.
   - "Top Paying Roles" to explore high-paying job titles.
3. Enter details like age, experience, job title, education, and check relevant skills.
4. Hit **Predict Salary** and get results!

---

## 📊 Sample Output

- **Salary Value** with gauge chart
- **Peer Comparison Box Plot**
- **Salary Timeline** for growth projections
- **Top Roles Bar Chart**

---

## 📌 Notes

- This app is for educational and demonstration purposes.
- It does not guarantee real-world salaries.

---

