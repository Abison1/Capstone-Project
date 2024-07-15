# Capstone-Project
Udacity capstone project - Customer Segmentation Report for a Financial Services
# Project Definition
In this project, I will undertake a comprehensive analysis of customer demographics for a mail-order sales company based in Germany. This analysis will involve comparing customer demographics with those of the general population to gain insights into how the company's customer base aligns with broader societal trends. To achieve this, I will employ unsupervised learning techniques to segment the customer base into distinct groups, thereby identifying the specific demographic segments that most accurately represent the company's core customers.

Following this segmentation, I will apply the insights gained to a third dataset, which contains demographic information for individuals targeted in a recent marketing campaign. Using predictive modeling techniques, I will evaluate which of these targeted individuals are most likely to convert into paying customers. This step will help the company tailor its marketing strategies more effectively, improving the likelihood of successful customer acquisition and enhan

# Libraries used
  - pandas as pd
  - numpy as np
  - matplotlib.pyplot as plt
  - seaborn as sns
  - from sklearn.preprocessing import StandardScaler
  - from sklearn.decomposition import PCA
  - from sklearn.cluster import KMeans
  - from sklearn.neighbors import KNeighborsClassifier
  - from sklearn.ensemble import RandomForestClassifier
  - from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
  - from sklearn.model_selection import RandomizedSearchCV
  - from sklearn.linear_model import LogisticRegression
  
# Project Analysis
The project involves analyzing four distinct datasets:
i.	Dataset 1: Demographics data for the general population of Germany, comprising 891,211 individuals and 366 features.
ii.	Dataset 2: Demographics data for customers of a mail-order company, with 191,652 individuals and 369 features.
iii.	Dataset 3: Demographics data for individuals targeted in a marketing campaign, including 42,982 individuals and 367 features.
iv.	Dataset 4: Demographics data for individuals targeted in a marketing campaign, containing 42,833 individuals and 366 features.

The data contains information about personal individual, household and building.
i.	The information from the first two files will be used to figure out how customers are similar to or differ from the general population at large( what is measure-metric)
ii.	,then use the analysis to make predictions on the other two files, predicting which recipients are most likely to become a customer for the mail-order company (what is measure-metric)

•  Analysis of Customer Demographics:
•	The first two datasets will be analyzed to understand how the company’s customers compare to the general population. This comparison will involve calculating similarity and divergence metrics between the two groups. Common metrics for this analysis include clustering similarity indices (e.g., silhouette score) and statistical measures (e.g., mean and variance differences).
•  Predictive Modeling for Marketing Campaigns:
•	Insights gained from the demographic comparison will be applied to the remaining two datasets to predict which individuals are most likely to convert into customers. For this, predictive metrics such as precision, recall, and the area under the ROC curve (AUC) will be used to evaluate the effectiveness of the models in identifying potential customers.


# Data exploration ana visualization
Visualization played a crucial role during the exploratory phase of the project, a period dedicated to becoming acquainted with the dataset and understanding its underlying characteristics. During this initial phase, visual tools and techniques were employed to delve into the data's structure and patterns, akin to peering "under the hood" to uncover how the data is organized and what it reveals.
By leveraging various types of visualizations—such as bar plots scatter plots, and box plots—I was able to gain insights into the distribution of different features, identify potential correlations, and spot any anomalies or outliers. This approach not only facilitated a deeper understanding of the dataset but also helped in formulating hypotheses and guiding subsequent data preprocessing and analysis steps.

# Modelling

# Principal Component Analysis (PCA) for Dimensionality Reduction

Principal Component Analysis (PCA) was employed to reduce the dimensionality of the dataset. As detailed in the preceding visualization plot, PCA transforms the original high-dimensional data into a lower-dimensional space while preserving as much variance as possible.

# KMeans Clustering for Grouping Data
KMeans clustering was utilized to segment the data into distinct groups based on the features after dimensionality reduction with PCA.
# K-Nearest Neighbors (KNN) Model Performance
The K-Nearest Neighbors (KNN) algorithm was applied as a classification model, and it demonstrated high accuracy in predictions. However, despite its overall good performance in terms of accuracy, the ROC (Receiver Operating Characteristic) curve output was poor.

# Conclusion
The market segmentation project offered valuable insights into customer behavior and preferences, facilitating more strategic and data-driven decision-making. By employing PCA for dimensionality reduction, KMeans for clustering, and KNN for classification, we were able to uncover meaningful patterns and segments within the data. The project highlighted the importance of data quality, effective dimensionality reduction, and comprehensive model evaluation. Moving forward, these insights will inform the development of targeted marketing strategies and contribute to more effective customer engagement.
