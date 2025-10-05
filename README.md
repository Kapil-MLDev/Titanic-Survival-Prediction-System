# Titanic Passenger Survival Prediction Using Machine Learning 🚢

A machine learning project to predict passenger survival on the Titanic based on historical data. The project includes a trained model, comprehensive data analysis, interactive visualizations, and a feature-rich Streamlit web application for user-friendly predictions.

---

## 📌 Project Overview

This project leverages machine learning algorithms to predict Titanic passenger survival based on multiple features including passenger class, gender, age, family relationships, fare, and embarkation port. The dataset contains **891 entries** representing passengers aboard the RMS Titanic during its tragic maiden voyage in 1912.

- The model was trained using **scikit-learn**
- Features comprehensive **data exploration** and **interactive visualizations**
- Deployed as a multi-page **Streamlit** application with real-time predictions
- Includes detailed **model performance evaluation** with metrics and visualizations

---

## 🚀 Features

- **Data Exploration**: 
  - Dataset overview with dimensions, data types, and statistical summaries
  - Missing value analysis and identification
  - Interactive filtering by passenger class, sex, survival status, and age range
  - Downloadable filtered data in CSV format

- **Interactive Visualizations**: 
  - Survival rate by passenger class (bar chart)
  - Age distribution by survival status (histogram)
  - Survival analysis by sex and passenger class (grouped bar chart)
  - Ticket fare distribution by survival (box plot)
  - Feature correlation heatmap

- **Machine Learning Prediction**: 
  - User-friendly input form with help text for all features
  - Real-time survival probability prediction
  - Probability distribution visualization
  - Detailed interpretation of prediction results
  - Input summary table

- **Model Performance Evaluation**:
  - Performance metrics (Accuracy, Precision, Recall, F1-Score)
  - Confusion matrix with interactive heatmap
  - ROC curve with AUC score
  - Feature importance visualization
  - Model comparison table across multiple algorithms
  - Performance analysis by passenger class
  - Downloadable performance report

- **User Interface**:
  - Sidebar navigation for easy access to all sections
  - Responsive design with custom CSS styling
  - Loading states and error handling
  - Comprehensive documentation and help text throughout

---

## 🛠️ Technologies Used

- **Machine Learning**: Random Forest Classifier, Logistic Regression, Support Vector Machine (`scikit-learn`)
- **Programming Language**: Python 3.9+
- **Libraries**:
  - `scikit-learn`: For building and training ML models
  - `pandas`: For data manipulation and analysis
  - `numpy`: For numerical computations
  - `plotly`: For interactive visualizations
  - `matplotlib`: For static plots
  - `seaborn`: For statistical visualizations
  - `streamlit`: For creating the multi-page interactive web application
- **Development Environment**: Jupyter Notebook / Google Colab (model training), Anaconda Navigator
- **Version Control**: GitHub

---

## 📂 Dataset

- **Filename**: `titanic_data.csv`
- **Source**: Kaggle Titanic Dataset
- **Description**: Contains passenger information including survival status, class, demographics, family relationships, fare, and embarkation port
- **Size**: 891 passengers with 11 features
- **Features Used for Prediction**:
  - **Pclass**: Passenger class (1st, 2nd, 3rd)
  - **Sex**: Gender (male, female)
  - **Age**: Age in years
  - **SibSp**: Number of siblings/spouses aboard
  - **Parch**: Number of parents/children aboard
  - **Fare**: Ticket fare in pounds
  - **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

---

## 📊 Model Performance

- **Algorithm**: Random Forest Classifier
- **Accuracy**: 82%
- **Precision**: 80%
- **Recall**: 75%
- **F1-Score**: 77%
- **AUC-ROC**: 0.85
- **Training/Test Split**: 80% / 20%

---

## 📁 Project Structure

```
titanic-survival-prediction/
│
├── app.py                          # Main Streamlit application (multi-page)
├── best_model.pkl                  # Trained ML model (pickle file)
├── titanic_data.csv                # Titanic dataset
├── model_metrics.pkl               # Model performance metrics
├── create_model_metrics.py         # Script to generate metrics
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

## 🔧 Installation & Setup

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/titanic-survival-prediction.git
cd titanic-survival-prediction
```

### Step 2: Create Virtual Environment
```bash
# Using conda
conda create -n titanic_app python=3.9
conda activate titanic_app

# OR using venv
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Required packages:**
```
streamlit==1.28.0
pandas==2.1.0
numpy==1.26.4
scikit-learn==1.3.0
plotly==5.17.0
matplotlib==3.7.2
seaborn==0.12.2
```

### Step 4: Generate Model Metrics (First Time Only)
```bash
python create_model_metrics.py
```

---

## ▶️ How to Run the Project

```bash
streamlit run app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`

---

## 💡 How to Use the Application

1. **Home Page**: View project overview, key statistics, and navigation guide
2. **Data Exploration**: Explore the dataset, apply filters, and download filtered data
3. **Visualizations**: Analyze interactive charts to understand survival patterns
4. **Make Prediction**: 
   - Enter passenger details (class, sex, age, family size, fare, embarkation port)
   - Click "🔮 Predict Survival"
   - View survival probability and detailed interpretation
5. **Model Performance**: Evaluate model accuracy, view confusion matrix, ROC curve, and feature importance

---

## 📈 Key Insights from Data Analysis

- **First-class passengers** had significantly higher survival rates (63%) compared to third-class (24%)
- **Females** had dramatically higher survival rates (74%) than males (19%)
- **Children** (under 18) were prioritized during evacuation
- **Higher fares** correlated with better survival rates due to cabin location
- **Sex** and **Passenger Class** are the most influential features for prediction

---

## 🎯 Future Enhancements

- Add more ML models (XGBoost, Neural Networks)
- Implement ensemble methods for improved accuracy
- Add SHAP values for enhanced model interpretability
- Deploy to cloud platforms (Heroku, AWS, Streamlit Cloud)
- Add user authentication and prediction history
- Implement real-time model retraining functionality

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👤 Author

**S.Kapila Deshapriya**
- LinkedIn: [https://www.linkedin.com/in/kapila-Deshapriya/]
---

## 🙏 Acknowledgments

- Kaggle for providing the Titanic dataset
- Streamlit team for the amazing web framework
- scikit-learn developers for comprehensive ML tools
- The data science community for inspiration and support

---

## 📚 References

- [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Plotly Python Documentation](https://plotly.com/python/)

---

**Made with ❤️ and Python**

*"Analyzing history to predict outcomes - A tribute to the passengers of the Titanic"*
