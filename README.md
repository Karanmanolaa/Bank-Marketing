#  Overview

The goal of this project is to predict whether a bank customer will subscribe to a term deposit after a marketing campaign. The model is trained using the Bank Marketing dataset from Kaggle, downloaded via the Kaggle API (henriqueyamahata/bank-marketing). The dataset contains approximately 41,188 records with 21 features describing customer demographics, financial information, and campaign interactions.

After performing data cleaning, preprocessing, and exploratory analysis, several machine learning models were trained and evaluated, including Decision Trees, Random Forest, and XGBoost. Hyperparameter tuning was performed using GridSearchCV, and the final model was selected based on evaluation metrics such as precision, recall, F1-score, and overall accuracy.

The trained model is served through a Flask API, containerized using Docker, and deployed on Render for live predictions.

Live application:

#### [https://bank-marketing-oqbz.onrender.com/](https://bank-marketing-oqbz.onrender.com)

Beyond the core machine learning workflow, this project also focuses on building a reproducible and production-style ML pipeline. The training process is tracked with MLflow, data and pipeline stages are managed using DVC, and a CI/CD pipeline with GitHub Actions automatically runs linting, training, testing, and Docker image builds to ensure the system remains reliable and reproducible.

#  Technical Aspects

This project was built using **Python** and several key data science and deployment tools.  
Below are the main technical components involved in each phase of the project:

###  Data Analysis & Modeling
- **Libraries:** pandas, NumPy, scikit-learn, Matplotlib,Xgboost, joblib 
- **Algorithm:** Random Forest Classifier, Decision tree, XG Boost 
- **Model Tuning:** GridSearchCV for hyperparameter optimization 
- **Pipeline:** Combined preprocessing and model into a single workflow using `ColumnTransformer` and `Pipeline`
- **Evaluation Metrics:** accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

#  MLOps & Engineering

To make the project reproducible and closer to a real production workflow, I integrated several MLOps tools and practices into the pipeline.

- **Experiment Tracking — MLflow**

MLflow is used to track model experiments, parameters, and evaluation metrics across different training runs.

- **Data & Pipeline Versioning — DVC**
  
DVC manages the dataset and training pipeline so that changes in data or code trigger reproducible training runs.

- **Backend API — Flask**
  
The trained model is served through a Flask API that exposes a prediction endpoint.

- **Containerization — Docker**
  
Docker is used to package the application and its dependencies into a consistent runtime environment.

- **Continuous Integration — GitHub Actions**
  
A CI pipeline automatically runs linting (flake8), pipeline execution (dvc repro), API tests (pytest), and Docker build checks whenever new code is pushed.

- **Deployment — Render**
  
The Dockerized Flask application is deployed on Render to provide a live prediction service.

#  Dataset Information

The dataset used in this project is the Bank Marketing dataset from Kaggle, originally sourced from a Portuguese bank’s marketing campaigns.
It contains details about customers, their personal and financial attributes, and the outcome of previous marketing contacts.
I downloaded it directly using the Kaggle API command below:
!kaggle datasets download -d henriqueyamahata/bank-marketing
The target column y indicates whether the customer subscribed to a term deposit (yes or no).

#  Deployment

Framework: Flask (REST API for predictions)  
Containerization: Docker (environment consistency)  
Hosting Platform: Render 

 
# Visualisation 


![Correlation Heatmap](assets/Corr_heatmap.png)
![Education Distribution](assets/Dist_Education.png)
![Job Distribution](assets/Dist_job.png)
![Marital Status Distribution](assets/Dist_marital_status.png)
![Contact Type](assets/Dist_contact.png)
![Housing Loan Distribution](assets/Dist_of_housing.png)
![Loan Distribution](assets/Distribution_of_loan.png)
![Subscription Rate by Job](assets/subs_rate_of_job.png)
![Subscription Rate by Month](assets/subs_rate_by_month.png)
![Day of Week Distribution](assets/Dist_of_day_of_week.png)
![Month Distribution](assets/Dist_month.png)
