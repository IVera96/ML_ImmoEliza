
# Problem:
The real estate company "ImmoEliza" asked to develop a machine learning model that can predict property prices in Belgium's real estate market.
To address this challenge, I created a Deep Neural Network (DNN) model.

# Installation:
pandas
numpy
MinMaxScaler from sklearn.preprocessing
Sequential, load_model from tensorflow.keras.models
Input, Dense, Dropout from tensorflow.keras.layers
Adam from tensorflow.keras.optimizers
IsolationForest from sklearn.ensemble
keras_tuner as kt
r2_score, mean_absolute_percentage_error, mean_absolute_error, mean_squared_error from sklearn.metrics
tensorflow as tf
adjust_row, remove_outliers, accuracy, smape from functions

# Data Overview:
The original dataset contains 16 columns and 1,663 rows, including 5 categorical features. There are no missing values in the data.
For preprocessing the categorical features, Label Encoding was applied.
To handle outliers in the dataset, I used the Isolation Forest algorithm.
Two new engineered features were added using external data:
- Salary_prov_med: the median salary by province.
- Prix_m2_prov: the average price per square meter by province.

Feature Importance:
Using SHAP (Shapley Additive Explanations), I was able to identify the most important features influencing the model. Based on the SHAP analysis, the top five most influential features are:
- Price
- Province
- Salary_prov_med
- Prix_m2_prov
- Municipality
- Living_Area

# Model Development:
Initially, the DNN model's performance was not satisfactory due to having too few layers and an insufficient number of neurons. To improve the model, I present two versions:
1. The first model uses 6 layers.
2. The second model leverages Bayesian Optimization to tune the hyperparameters for better training performance.

# Data Augmentation:
Despite improvements to the model, the dataset was still lacking in size and diversity for optimal training. To enhance the dataset, I used The Synthetic Data Vault (SDV), which allows for the generation of synthetic data.
For this, I developed a function called adjust_row. This function performs the following:
- Adjusts the living area and price of a property with a random variation.
- Simulates a small random change in the living area (between -10% and +10%) and recalculates the adjusted price based on:
  1. The province's average price per square meter (weighted at 20%).
  2. The current property's price (weighted at 80%).

# Model Performance Analysis:
Let's compare the results of different models trained. For evaluation, I used the Mean Absolute Error (MAE) metric. The following settings were used for training:

1. DNN with 6 Layers we analyze the results of the model without outliers. The following settings were used:
- Epochs: 200
- Fit patience: 10 for early stopping
- Batch_size: 16
  
- Using SDV for all features:
MAE:  62448.73 
- Without SDV for all features:
MAE: 83571.69 
-  Without SDV 5 features: 
MAE:  72840.60
-  Using SDV 5 features:
MAE:  73393.24 

2. DNN with Bayesian Optimization:
In this section, we analyze the results of the model with and without outliers. The following settings were used:
- Max trials: 30 (for hyperparameter tuning)
- Epochs: 20 (for Bayesian optimization)
- Epochs: 200 (for final training)
- Fit patience: 5 (for early stopping)

With Outliers:
DNN with Bayesian Optimization:
- For the 5 SHAP-selected features, with a batch size of 16:
MAE: 78896.54
- For the 5 SHAP-selected features, with a batch size of 32:
MAE: 79325.51
- For the 5 SHAP-selected features with SDV, with a batch size of 16:
MAE: 69,105.98

Without Outliers:
DNN with Bayesian Optimization and SDV:
- Using all features, batch size 16:
MAE: 65,380.18
- Using all features, batch size 32:
MAE: 65,005.93
- Using the 5 SHAP-selected features, batch size 32:
MAE: 76,250.22

# Conclusion:
From the model performance analysis, we can draw the following conclusions:
1. DNN with 6 Layers:
The use of Synthetic Data Vault (SDV) to augment the data significantly improved the model's performance, reducing the MAE from 83,571 (without SDV) to 62448.73 (with SDV). This indicates that synthetic data can be a valuable tool for improving the model's predictive accuracy.
2. DNN with Bayesian Optimization:
Unfortunately, we are unable to directly compare the results with and without outliers for all features or the 5 selected features, as the model consistently performed better without outliers. In particular, the best performance was observed with a batch size of 16. However, the lowest MAE (65,005.93) was achieved when using a batch size of 32.

Effect of Outliers:
The presence of outliers negatively impacted the model's performance, as evidenced by the higher MAE values when outliers were included in the dataset. Removing outliers led to better results, particularly when combined with SDV for data augmentation.

Feature Selection:
The 5 SHAP-selected features consistently performed well across different configurations, indicating their strong influence on the model’s predictions even if the model performs better with all the data.

How could you improve this result?
- Feature Engineering: Create new features, such as more detailed economic data.
- Advanced Data Augmentation: Explore other data augmentation techniques like GANs.
- Hyperparameter Tuning: Further optimize parameters with techniques like grid search.
- Ensemble Models: Combine DNN with other models to improve performance.

Which part of the process has the most impact on the results?
- Data Augmentation (SDV): Generating synthetic data significantly improved generalization.
- Outlier Handling: Removing outliers led to better results.
- Feature Selection: Using SHAP to select important features boosted accuracy.

How should you divide your time working on this kind of project?
- Data Preprocessing: Data cleaning and synthetic data generation.
- Model Development: Model creation and parameter optimization.
- Feature Engineering: Experimenting with new features.
- Model Evaluation (10%): Validation and testing.
- Documentation (5%): Documentation and model interpretability.

Try to illustrate your model if it's possible to make it interpretable:
- Use SHAP to visualize feature importance.
  

