
Proposal
===
Abstract
---
The US Stock market is a hot-debated topic in people's normal lives. Many companies are facing fierce competition and costly in order to be outstanding in the market. This project is going to analyze US Stocks between 2014 and 2018 from CSV datasets. It will reveal the financial indicators for the stock market trends. This project makes use of machine learning data analytic techniques to reveal the trend of datasets. It contains computational methods implementing by python language. Through this project, four group members aim to generate predictive US stock developments, and achieve a more visible, accessible stock financial services to the general public.

I. Introduction
---
The main objective of this course project is to leverage the algorithms and techniques we learned from class to analyze real-world data, compare the performance of different algorithms, techniques and interpret the results. To achieve this goal, we select the “Financial Indicators of the US stocks” dataset as our data. Detailed information about this dataset is given in the following section. Selected machine learning classification algorithms will be implemented by Spark and Scikit-learn libraries in our project. 

The rapid progress in machine learning has revealed its strength in handling voluminous data streams, extracting informative features and tackling complexities. Various kinds of fields, such as computer vision, natural language processing, robots have seen the huge success of the advanced machine learning models.

A rich line of machine learning approaches has also been introduced to predict the stock price in this area. Jigar et al (2015)[1] compare four prediction models, Artificial Neural Network (ANN), Support Vector Machine (SVM), random forest and naive-Bayes regarding the problem of prediction stock price index movement. Sotirios et al (2018)[2] analyzed the Extreme Gradient Boosting and Deep Neural Networks on the stock price prediction.

Consider the unexpected events may have severe impacts on short-term stock price, 
rather than look at the historical stock price and treat it as a regression problem, we hope to test if we can provide long term gain prediction via machine learning methods. The main task is to tell if the stock will be worth buying based on the sufficient financial information of one company.
 
Possible challenges within this dataset may be: 1) The financial indicators may not be sufficient enough to provide the hidden features to do the prediction. Which means that the models may encounter underfitting issues. If so, we will dig deep into the features and try to interpret the results;  2) The data of different years may not follow the i.i.d assumption, which means that the models trained on the past years may not be able to achieve good performance in the following years. Aware of this issue, we will probably embed time series analysis or domain adaptation techniques into the model and help to improve the performance.


II. Materials and Methods 
---
**Dataset: 200+ Financial Indicators of US stocks (2014-2018)**[3]

This Data repo contains five datasets(2014-2018), each with 200+ financial indicators, that are commonly found in the 10-K filings releases yearly by publicly-traded companies. And there are approximately 4000 data samples in each dataset.

The third-to-last column of these datasets is ‘sector’, which lists the sector of each stock. In the US stock market, each company is part of a sector that classifies it in a macro-area. Since all the sectors have been collected (Basic Materials, Communication Services, Consumer Cyclical, Consumer Defensive, Energy, Financial Services, Healthcare, Industrial, Real Estate, Technology, and Utilities), we could choose to perform per-sector analyses and comparisons. Because stokes of different sectors may be affected by different financial indicators. 

The last column of datasets is ‘class’, lists a binary classification for each stock. From a trading perspective, the 1 identifies those stocks that a hypothetical trader should buy at the start of the year and sell at the end of the year for a profit, while the 0 identifies those stocks that a hypothetical trader should not buy since their value will decrease, meaning a loss of capital. We will use this column as labels to compare with the results from our machine learning model.

We will use these datasets to train several machine learning models to learn the differences between stocks that perform well and those that don't, and then leverage this knowledge to predict if the stock will be worth buying.

Multiple steps will be carried out in our project:
Data Preprocessing: The raw data will be preprocessed before feeding into models. Including handling missing values, feature normalization and feature reduction.
Model Tuning: We will conduct grid search to tune the hyperparameters of each model with k-fold validation.
Results comparison: We will report and analyze the performance of different algorithms via several metrics including accuracy, recall, precision, and f1 score.

**Technology:**

For the libraries, we choose the MLlib and Scikit-learn.
MLlib is Apache Spark's scalable machine learning library which is usable in different languages and platforms
Scikit-learn is another popular free software machine learning library for the Python programming language, with plenty of built-in algorithms and can be easily implemented with user-friendly APIs.

We select the following algorithms which can be found in the aforementioned two libraries.
Logistic regression:
Decision tree classifier:
Random forest classifier:
Gradient-boosted tree classifier:
Multilayer perceptron classifier:
Linear Support Vector Machine:
Naive Bayes:

Ref:

[1] Patel, Jigar, et al. "Predicting stock and stock price index movement using trend deterministic data preparation and machine learning techniques." Expert systems with applications 42.1 (2015): 259-268.

[2] Chatzis, Sotirios P., et al. "Forecasting stock market crisis events using deep and statistical machine learning techniques." Expert Systems with Applications 112 (2018): 353-371.

[3] https://www.kaggle.com/cnic92/200-financial-indicators-of-us-stocks-20142018/data

