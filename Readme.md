# Customer Churn Prediction 

<video src="https://github.com/user-attachments/assets/18ac39f7-6342-43e8-a6cd-feb53c2873ae" width="352" height="500"></video>

A comprehensive machine learning project to predict customer churn using various classification algorithms. This project demonstrates end-to-end ML pipeline development including exploratory data analysis, feature engineering, model training, evaluation, and deployment-ready predictions.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Key Features](#key-features)
- [Results & Insights](#results--insights)
- [Future Improvements](#future-improvements)
- [License](#license)

## 🔍 Overview

Customer churn prediction is critical for businesses to retain customers and reduce revenue loss. This project builds predictive models to identify customers likely to churn, enabling proactive retention strategies.

**Key Highlights:**
- 5 machine learning algorithms compared
- Comprehensive feature engineering
- Cross-validated model selection
- ROC AUC: **86.92%** on test set
- Production-ready pipeline saved for deployment

## 📊 Dataset

**Size:** 10,000 customer records  
**Features:** 12 attributes including demographic, account, and behavioral data  
**Target:** Binary classification (Churn: Yes/No)  
**Class Distribution:** 79.63% non-churn, 20.37% churn

### Features:
| Feature | Type | Description |
|---------|------|-------------|
| `customer_id` | Integer | Unique customer identifier |
| `credit_score` | Integer | Customer credit score (350-850) |
| `country` | Categorical | Customer location (France, Spain, Germany) |
| `gender` | Categorical | Male/Female |
| `age` | Integer | Customer age (18-92) |
| `tenure` | Integer | Years with the company (0-10) |
| `balance` | Float | Account balance |
| `products_number` | Integer | Number of products held (1-4) |
| `credit_card` | Binary | Has credit card (0/1) |
| `active_member` | Binary | Active membership status (0/1) |
| `estimated_salary` | Float | Estimated annual salary |
| `churn` | Binary | **Target variable** (0=stayed, 1=churned) |

## 🗂️ Project Structure

```
customer-churn-prediction/
│
├── customer_data.csv              # Raw dataset
├── churn_prediction.ipynb         # Main Jupyter notebook
├── best_churn_pipeline.pkl        # Saved model pipeline
├── README.md                      # Project documentation
│
└── outputs/                       # Generated visualizations
    ├── distribution_plots/
    ├── correlation_heatmap.png
    ├── confusion_matrix.png
    └── feature_importance.png
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- Jupyter Notebook

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Required Libraries
```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
joblib>=1.2.0
```

## 💻 Usage

### 1. Run the Complete Pipeline

```bash
jupyter notebook churn_prediction.ipynb
```

### 2. Load Pre-trained Model for Predictions

```python
import joblib
import pandas as pd

# Load saved pipeline
pipeline = joblib.load('best_churn_pipeline.pkl')

# Prepare new customer data
new_customer = {
    'credit_score': 650,
    'country': 'France',
    'gender': 'Male',
    'age': 40,
    'tenure': 3,
    'balance': 50000.0,
    'products_number': 2,
    'credit_card': 1,
    'active_member': 1,
    'estimated_salary': 60000.0,
}

# Predict
prediction = pipeline.predict(pd.DataFrame([new_customer]))
probability = pipeline.predict_proba(pd.DataFrame([new_customer]))[0, 1]

print(f"Churn Prediction: {prediction[0]}")
print(f"Churn Probability: {probability:.2%}")
```

## 📈 Model Performance

### Models Compared:
| Model | Cross-Val ROC AUC | Std Dev |
|-------|-------------------|---------|
| **Gradient Boosting** ⭐ | **0.8628** | 0.0097 |
| Random Forest | 0.8486 | 0.0130 |
| AdaBoost | 0.8462 | 0.0133 |
| SVC | 0.8351 | 0.0104 |
| Logistic Regression | 0.7877 | 0.0244 |

### Best Model (Gradient Boosting) - Test Set Results:
```
Accuracy:  86.80%
Precision: 78.04%
Recall:    48.89%
F1-Score:  60.12%
ROC AUC:   86.92%
```

### Confusion Matrix:
```
              Predicted
              No    Yes
Actual No    1529   64
       Yes   208   199
```

## ✨ Key Features

### Feature Engineering:
1. **Balance per Product** - Average balance across products
2. **Salary to Balance Ratio** - Financial health indicator
3. **Age Groups** - Categorized age brackets (<25, 25-34, 35-44, 45-54, 55-64, 65+)
4. **Tenure Buckets** - Customer loyalty segments (0, 1-2, 3-5, 6-10, 10+ years)
5. **High Balance Flag** - Customers above 75th percentile

### Top 5 Predictive Features:
1. **Age** (32.8%) - Older customers more likely to churn
2. **Number of Products** (26.6%) - Multiple products reduce churn
3. **Balance per Product** (6.3%)
4. **Account Balance** (5.7%)
5. **Active Member Status** (5.3%)

## 🔬 Results & Insights

### Key Findings:

**Demographics:**
- Customers aged 45+ show significantly higher churn rates
- Gender has minimal impact on churn prediction
- Germany-based customers churn more than France/Spain

**Account Behavior:**
- Customers with **only 1 product** have the highest churn risk
- **Inactive members** are 2x more likely to churn
- High account balance alone doesn't prevent churn

**Business Implications:**
- **Target older customers** with retention campaigns
- **Cross-sell additional products** to single-product customers
- **Re-engage inactive members** through personalized outreach
- **Monitor Germany operations** for service quality issues

### Visualizations:
The project includes comprehensive EDA with:
- Distribution plots by churn status
- Correlation heatmaps
- Pairwise feature relationships
- Confusion matrices
- Feature importance charts

## 📝 Notes

- The model prioritizes **precision over recall** to minimize false positives in retention campaigns
- All categorical features are one-hot encoded
- Numerical features are standardized using StandardScaler
- Pipeline ensures consistent preprocessing for new predictions

## 📧 Contact

**Your Name**  
Email: your.email@example.com  
LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com)  
GitHub: [@yourusername](https://github.com/yourusername)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

⭐ **Star this repo** if you find it helpful!

**Built with** ❤️ **using Python, scikit-learn, and pandas**
