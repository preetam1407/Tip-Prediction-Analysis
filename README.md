# Tip Prediction Analysis

Welcome to the **Tip Prediction Analysis** project! This repository aims to enhance restaurants' understanding of tipping behavior by developing predictive models based on customer billing and demographic details. By leveraging various regression techniques, we identify significant factors impacting tip amounts and provide actionable insights for restaurant management.

---

## Table of Contents

- [Introduction](#introduction)
- [Dataset Overview](#dataset-overview)
- [Project Objectives](#project-objectives)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Modeling Techniques](#modeling-techniques)
  - [1. Linear Regression](#1-linear-regression)
  - [2. Ridge Regression](#2-ridge-regression)
  - [3. Lasso Regression](#3-lasso-regression)
  - [4. Decision Tree Regression](#4-decision-tree-regression)
  - [5. Random Forest Regression](#5-random-forest-regression)
  - [6. Support Vector Regression (SVR)](#6-support-vector-regression-svr)
  - [7. K-Nearest Neighbors (KNN)](#7-k-nearest-neighbors-knn)
- [Model Evaluation and Results](#model-evaluation-and-results)
- [Significant Factors Impacting Tip Amounts](#significant-factors-impacting-tip-amounts)
- [Insights for Management](#insights-for-management)
- [Conclusion](#conclusion)
- [Getting Started](#getting-started)
---

## Introduction

Understanding tipping behavior is crucial for restaurants aiming to optimize operations and enhance customer satisfaction. This project explores various regression techniques to predict tip amounts, identify significant influencing factors, and provide actionable insights for management.

## Dataset Overview

The dataset contains information about customers' bills and demographic details:

| total_bill | tip  | sex    | smoker | day | time   | size |
|------------|------|--------|--------|-----|--------|------|
| 16.99      | 1.01 | Female | No     | Sun | Dinner | 2    |
| 10.34      | 1.66 | Male   | No     | Sun | Dinner | 3    |
| 21.01      | 3.50 | Male   | No     | Sun | Dinner | 3    |
| ...        | ...  | ...    | ...    | ... | ...    | ...  |

**Features:**

- `total_bill`: Total bill amount (including tip).
- `tip`: Tip amount.
- `sex`: Gender of the bill payer.
- `smoker`: Whether the party had smokers.
- `day`: Day of the week.
- `time`: Time of day (`Lunch` or `Dinner`).
- `size`: Number of people in the party.

## Project Objectives

1. **Identify Significant Factors**: Determine which factors significantly impact tip amounts.
2. **Prediction Accuracy**: Build and evaluate models to forecast tips effectively.
3. **Insights for Management**: Provide actionable insights to improve customer service strategies and revenue management.

## Exploratory Data Analysis (EDA)

We performed an in-depth EDA to understand the data and relationships between variables:

- **Scatter Plots**: Visualized relationships between `total_bill`, `tip`, and other features.
- **Pair Plots**: Explored interactions between multiple features.
- **Correlation Matrix**: Identified strong correlations between variables.
- **Statistical Tests**: Conducted Rainbow Test to check for linearity.
- **Residual Plots**: Assessed homoscedasticity and model assumptions.

**Key Findings from EDA:**

- Positive correlation between `total_bill` and `tip`.
- Party `size` also shows a positive relationship with `tip`.
- Categorical variables like `sex`, `smoker`, `day`, and `time` require encoding.

## Modeling Techniques

We applied various regression models to predict tip amounts:

### 1. Linear Regression

A basic model assuming a linear relationship between independent variables and the tip amount.

- **Strategy**: Used as a baseline model.
- **Performance**: RMSE = 0.8387, R² = 0.4373.

### 2. Ridge Regression

An extension of Linear Regression with L2 regularization to prevent overfitting.

- **Strategy**: Tuned the `alpha` parameter using cross-validation.
- **Performance**: RMSE = 0.7896, R² = 0.5012.

### 3. Lasso Regression

Linear Regression with L1 regularization, which can perform feature selection.

- **Strategy**: Tuned the `alpha` parameter to minimize RMSE.
- **Performance**: **Best Model** with RMSE = 0.7687, R² = 0.5273.

### 4. Decision Tree Regression

A non-linear model that splits data into subsets based on feature values.

- **Strategy**: Pruned the tree to prevent overfitting.
- **Performance**: RMSE = 1.0230, R² = 0.1627.

### 5. Random Forest Regression

An ensemble model using multiple decision trees to improve performance.

- **Strategy**: Tuned `n_estimators` and `max_depth`.
- **Performance**: RMSE = 0.9195, R² = 0.3235.

### 6. Support Vector Regression (SVR)

Uses support vectors to perform regression with kernels for non-linear relationships.

- **Strategy**: Scaled features and tuned `C`, `epsilon`, and kernel parameters.
- **Performance**: RMSE = 0.8992, R² = 0.3532.

### 7. K-Nearest Neighbors (KNN)

Predicts the tip amount based on the average of the nearest neighbors.

- **Strategy**: Scaled features and optimized `n_neighbors`.
- **Performance**: RMSE = 0.9227, R² = 0.3189.

## Model Evaluation and Results

|       **Model**       |   **RMSE**  | **R² Score** | **Cross-Validated R²** |
|:---------------------:|:-----------:|:------------:|:----------------------:|
|  Linear Regression    |   0.8387    |    0.4373    |         0.4215         |
|   Ridge Regression    |   0.7896    |    0.5012    |         0.4540         |
|   **Lasso Regression**| **0.7687**  |  **0.5273**  |       **0.4566**       |
|    Decision Tree      |   1.0230    |    0.1627    |         0.3633         |
|    Random Forest      |   0.9195    |    0.3235    |         0.3777         |
|         SVR           |   0.8992    |    0.3532    |         0.3729         |
|         KNN           |   0.9227    |    0.3189    |         0.3240         |

**Conclusion:** The **Lasso Regression** model outperformed others, providing the lowest RMSE and highest R² Score.

## Significant Factors Impacting Tip Amounts

Using the coefficients from the Lasso Regression model, we identified the following significant factors:

- **Total Bill**: Strong positive impact on tip amount.
- **Party Size**: Positive correlation, larger parties tend to tip more.
- **Time of Day**: Dinner times are associated with higher tips.
- **Smoker Status**: Non-smokers tend to tip more than smokers.

## Insights for Management

Based on our findings, here are actionable insights for restaurant management:

### 1. Enhance High-Bill Experiences

- **Upselling Techniques**: Train staff to suggest premium items.
- **Exclusive Offers**: Create special menus for higher spending.

### 2. Attract Larger Groups

- **Group Discounts**: Offer incentives for large parties.
- **Event Hosting**: Promote the venue for celebrations and corporate events.

### 3. Focus on Dinner Service

- **Ambiance Improvement**: Enhance lighting and music for dinner.
- **Dinner Specials**: Introduce exclusive evening dishes.

### 4. Cater to Non-Smokers

- **Smoke-Free Promotions**: Highlight smoke-free environments.
- **Health-Conscious Options**: Offer menus appealing to non-smokers.

## Conclusion

By leveraging regression analysis, we've identified key factors influencing tip amounts and developed a predictive model to aid restaurants in strategic decision-making. Implementing these insights can lead to improved customer satisfaction and increased revenue.

## Getting Started

To replicate the analysis:

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/tip-prediction-analysis.git
   ```

2. **Run the Notebook**

   Open `Tip_Prediction_Analysis.ipynb` in Jupyter Notebook or JupyterLab.
