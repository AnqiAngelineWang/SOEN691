
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

We select the following algorithms which can be found in the aforementioned two libraries:
Logistic regression,
Decision tree classifier,
Random forest classifier,
Gradient-boosted tree classifier,
Multilayer perceptron classifier,
Linear Support Vector Machine,
Naive Bayes.

III.Results
---
**Result-Case 1** 
* Comparison of sklearn and spark(same year)
| Scikit-learn  |  2014  |  2015  |  2016  |  2017  |  2018  |  Averaged exclude 2014 and 2017  |
| :-----------: |:------:|:-----: |:------:|:------:|:------:| :-------------------------------:|
| LR            | 49.53% | 80.44% | 79.63% | 11.41% | 81.82% |             80.63%               |
| NB            | 59.88% | 80.19% | 14.50% | 44.09% | 21.85% |             38.85%               |
| SVM           | 45.83% | 80.70% | 80.75% | 7.36%  | 82.64% |             81.37%               |
| CART          | 31.19% | 80.74% | 80.75% | 7.36%  | 82.64% |             80.73%               |
| RF            | 50.87% | 82.89% | 80.28% | 26.16% | 82.34% |             81.84%               |
| GBDT          | 45.61% | 82.38% | 81.01% | 11.11% | 81.28% |             81.56%               |
| MLP           | 47.12% | 80.06% | 72.03% | 30.36% | 79.06% |             77.05%               |


| Spark         |  2014  |  2015  |  2016  |  2017  |  2018  |  Averaged exclude 2014 and 2017  |
| :-----------: |:------:|:-----: |:------:|:------:|:------:| :-------------------------------:|
| LR            | 0.00%  | 83.52% | 80.12% | 0.00%  | 82.52% |             82.05%               |
| NB            | 1.14%  | 83.53% | 80.07% | 0.00%  | 82.11% |             81.91%               |
| SVM           | 3.88%  | 83.33% | 80.10% | 0.68%  | 82.33% |             81.92%               |
| CART          | 47.45% | 82.87% | 80.12% | 28.69% | 78.79% |             80.59%               |
| RF            | 47.01% | 84.54% | 79.20% | 22.79% | 82.21% |             81.99%               |
| GBDT          | 48.01% | 83.48% | 76.65% | 23.08% | 80.12% |             80.09%               |
| MLP           | 33.40% | 83.26% | 80.10% | 0.00%  | 81.30% |             81.55%               |

For this case , we first split the data within one year to training and testing set to validate the data quality. The split rate is 4:1. We report the F1 score considering the data imbalance. 
Because data of 2014 and 2017 is of poor quality, the average precision we exclude these two years.

* Comparison of sklearn and spark(different year)
| Scikit-learn next year |  2014-2015  |  2015-2016  |  2016-2017  |  2017-2018  |
| :---------------------:|:-----------:|:----------: |:-----------:|:-----------:|
| LR                     |    41.49%   |    78.84%   |    43.48%   |    9.01%    |
| NB                     |    80.70%   |    78.79%   |    10.79%   |    79.78%   |
| SVM                    |    34.47%   |    78.18%   |    43.40%   |    0.66%    |
| CART                   |    24.06%   |    78.33%   |    43.29%   |    0.00%    | 
| RF                     |    24.06%   |    77.96%   |    42.58%   |    0.00%    |
| GBDT                   |    33.18%   |    79.07%   |    43.29%   |    3.41%    |
| MLP                    |    43.27%   |    80.30%   |    38.53%   |    40.61%   |


| Spark next year        |  2014-2015  |  2015-2016  |  2016-2017  |  2017-2018  |
| :---------------------:|:-----------:|:----------: |:-----------:|:-----------:|
| LR                     |    0.00%    |    80.30%   |    43.29%   |    0.07%    |
| NB                     |    3.58%    |    80.30%   |    43.31%   |    39.59%   |
| SVM                    |    3.05%    |    80.33%   |    43.23%   |    0.07%    |
| CART                   |    54.38%   |    74.63%   |    41.00%   |    39.59%   | 
| RF                     |    41.56%   |    78.67%   |    43.73%   |    14.61%   |
| GBDT                   |    32.44%   |    77.13%   |    42.58%   |    21.95%   |
| MLP                    |    19.61%   |    80.25%   |    43.15%   |    0.00%    |

For this case , we use the data of one year to train and the data of next year to test.
Because data of 2014 and 2017 is of poor qualit, we only see the results of 2015-2016.

**Result-Case 2**
* positive and negative class of each year

| Year |  Positive |  Negative |  Positive Percentange |
| :---:|:---------:|:---------:|:---------------------:|
| 2014 |    1634   |    3808   |         30.03%        | 
| 2015 |    2891   |    4120   |         41.24%        |
| 2016 |    3218   |    4797   |         40.15%        |
| 2017 |    1370   |    4960   |         21.64%        |
| 2018 |    3046   |    4392   |         40.95%        | 
* Comparison of sklearn and spark(different year)
 * Oversampling Results(same year):
 
| Scikit-learn  |  2014  |  2015  |  2016  |  2017  |  2018  |
| :-----------: |:------:|:-----: |:------:|:------:|:------:|
| LR            | 55.60% | 70.86% | 68.71% | 39.73% | 71.82% |
| NB            | 59.75% | 79.84% | 14.73% | 44.37% | 22.81% |
| SVM           | 5.95%  | 80.09% | 80.80% | 11.73% | 82.62% |
| CART          | 53.78% | 75.66% | 72.84% | 35.80% | 77.09% |
| RF            | 56.75% | 80.85% | 76.31% | 35.87% | 80.10% |
| GBDT          | 57.57% | 81.04% | 78.55% | 36.73% | 81.77% |
| MLP           | 53.41% | 76.34% | 71.96% | 38.72% | 74.58% |

* comparison of results of imbalanced data and oversampled data:

| Year |  Imbalanced |  Oversample |  Improvement |
| :---:|:---------: |:-----------:|:------------:|
| 2014 |   47.37%   |    56.14%   |    8.77%     | 
| 2015 |   81.12%   |    77.43%   |    -3.68%    |
| 2016 |   68.12%   |    63.85%   |    -4.27%    |
| 2017 |   20.52%   |    38.54%   |    18.02%    |
| 2018 |   71.09%   |    68.03%   |    -3.06%    | 

IV.Discussion
---
Predict a stock is worth buying or not is hard, the correctness of model  very dependent the training data. 
High quality or satisfied data sets are efficient to forecast the future stoke trends.  For example, the results in year 2015 and 2016.
This also follows the assumption that social influences have not been considered. 
For example, it could not forecast the stock trend due to COVID-19

Ref:
---

[1] Patel, Jigar, et al. "Predicting stock and stock price index movement using trend deterministic data preparation and machine learning techniques." Expert systems with applications 42.1 (2015): 259-268.

[2] Chatzis, Sotirios P., et al. "Forecasting stock market crisis events using deep and statistical machine learning techniques." Expert Systems with Applications 112 (2018): 353-371.

[3] https://www.kaggle.com/cnic92/200-financial-indicators-of-us-stocks-20142018/data

