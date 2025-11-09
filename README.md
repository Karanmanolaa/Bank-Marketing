# 1️⃣ Overview

The goal of this project is to predict whether a bank customer will subscribe to a term deposit after a marketing campaign.
I used the Bank Marketing dataset from Kaggle, downloaded using the Kaggle API (henriqueyamahata/bank-marketing), which contains around 41,188 records and 21 features about customer demographics and campaign details.

After cleaning and exploring the data, I trained a Random Forest Classifier that achieved strong predictive accuracy.
Finally, I built a Flask web app, containerized it with Docker, and deployed it on Render for live predictions:

# https://bank-marketing-oqbz.onrender.com/
