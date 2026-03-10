# importing libraries
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

# mlflow model signature
from mlflow.models.signature import infer_signature

# sklearn utilities
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# evaluation metrics
from sklearn.metrics import accuracy_score, roc_auc_score

# machine learning models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# setting mlflow experiment
mlflow.set_experiment("bank_marketing_model")


# loading dataset
df = pd.read_csv("data/bank-additional-full.csv", sep=";")
df.columns = df.columns.str.strip()

# target variable
y = df["y"].map({"yes": 1, "no": 0}).astype(int)

# feature matrix
X = df.drop("y", axis=1)


# identifying categorical and numerical columns
cat_cols = X.select_dtypes(include="object").columns
num_cols = X.select_dtypes(exclude="object").columns


# preprocessing pipeline
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)


# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# models dictionary
models = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
    ),
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=3
    ),
}


# tracking best model
best_model = None
best_score = 0


# training models loop
for model_name, model in models.items():

    # starting mlflow run
    with mlflow.start_run(run_name=model_name):

        # creating pipeline
        pipe = Pipeline([("preprocess", preprocess), ("model", model)])

        # training model
        pipe.fit(X_train, y_train)

        # predictions
        train_preds = pipe.predict(X_train)
        test_preds = pipe.predict(X_test)

        # evaluation metrics
        train_acc = accuracy_score(y_train, train_preds)
        test_acc = accuracy_score(y_test, test_preds)

        # probability predictions
        test_proba = pipe.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, test_proba)

        # logging model parameters
        mlflow.log_param("model_type", model_name)

        # logging metrics
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("roc_auc", roc_auc)

        # creating model signature
        signature = infer_signature(X_train, test_preds)

        # logging model to mlflow
        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            signature=signature,
            input_example=X_train.iloc[:5],
        )

        # selecting best model
        if test_acc > best_score:
            best_score = test_acc
            best_model = pipe


# saving best model
joblib.dump(best_model, "models/model.joblib")

print("Best model saved")
