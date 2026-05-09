# Ecommerce Return Prediction App

This project is an AI/ML-powered ecommerce return prediction app built using Python, scikit-learn, and Streamlit.

The app predicts whether an ecommerce order is likely to be returned based on order, product, customer, payment, delivery, and date-related features. It also shows the return probability, risk level, and a business recommendation to help ecommerce teams reduce avoidable returns.

---

## Project Objective

Ecommerce returns can create major operational and financial costs for online businesses. This project aims to predict the likelihood of an order being returned so that businesses can identify risky orders early and take preventive action.

The app answers the question:

> Is this ecommerce order likely to be returned?

---

## Machine Learning Problem Type

This is a supervised machine learning binary classification problem.

The target variable is:

```text
is_returned

where
0 = Not Returned
1 = Returned
```
---------
## Dataset

The project uses a synthetic ecommerce sales dataset:
synthetic_ecommerce_sales_2025.csv

from kaggle URL: https://www.kaggle.com/datasets/emirhanakku/synthetic-e-commerce-sales-dataset-2025

The dataset contains order-level ecommerce information such as:

- Product category
- Product price
- Quantity
- Region
- Payment method
- Delivery days
- Customer rating
- Discount percentage
- Order date
- Return status

-----
## Exploratory Data Analysis Summary

The dataset contains:
- 100,000 rows
- 13 original columns
- No missing values
- No duplicate rows

The target variable `is_returned` is imbalanced, i.e. 
- Not Returned: 93.94%
- Returned: 6.06%

Because of this imbalance, accuracy alone is not enough to judge model performance. Precision, recall, F1-score, and confusion matrix were also considered.

**Insight**: Fashion showed a higher return rate compared to other product categories.

-----------
## Model Training

The data was split into:

> 80% training data
> 
> 20% testing data

The split was done using stratification so that the returned/not-returned ratio remained consistent in both train and test datasets.

Categorical columns were encoded using one-hot encoding, and numerical columns were scaled using StandardScaler.

The following supervised machine learning models were tested:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier

-------
## Model Evaluation Comparison 

| Model               | Accuracy | Precision |  Recall | F1 Score |
| ------------------- | -------: | --------: | ------: | -------: |
| Random Forest       |  0.82730 |   0.11478 | 0.27558 |  0.16206 |
| Logistic Regression |  0.82345 |   0.11208 | 0.27640 |  0.15949 |
| Decision Tree       |  0.72610 |   0.08679 | 0.36964 |  0.14057 |
| Gradient Boosting   |  0.93940 |   0.00000 | 0.00000 |  0.00000 |

Although Gradient Boosting had the highest accuracy, it predicted all orders as not returned and failed to identify returned orders. Therefore, it was rejected. 

So **final model selected was: Random Forest Classifier** as *Random Forest achieved the best F1-score and precision among the tested models while maintaining similar recall to Logistic Regression.*

------
## Streamlit App features: 

The Streamlit app allows users to enter order details such as:

Product category
Region
Payment method
Product price
Quantity
Delivery days
Customer rating
Discount percentage
Order year
Order month
Order day of week

**The app then displays:**
- Prediction: Likely to be Returned / Not Likely to be Returned
- Return probability
- Risk level: Low / Medium / High
- Business recommendation based on the Risk level 

**View the app here:** url 

------
## Limitations

This project uses a synthetic dataset, so the results should be treated as a learning and demonstration project rather than a production-ready e-commerce return prediction system.

The model performance is affected by class imbalance because returned orders represent only a small percentage of the dataset.

For real-world deployment, the model should be improved using:
- Real ecommerce transaction data
- More customer behavior features
- Return reason data
- Product review data
- Historical customer return patterns
