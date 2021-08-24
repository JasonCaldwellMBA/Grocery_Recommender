# Grocery Recommendation System

## Problem Statement
Educational project to gather data on grocery purchases to recommend additional products to consumers using the data science method of: 
- Problem Identification
- Data Wrangling
- Exploratory Data Analysis
- Pre-processing and Training
- Modeling
- Documentation

These recommendations are intended to help increase sales and improve customer satisfaction with the application.

### Data Wrangling
Two different Amazon datasets from 2018 were used for the project: Grocery and Gourmet Food 5-core and Grocery and Gourmet Food Meta. Merged them using an inner join creating 29 columns with 1.1 million+ rows. 

Removed 6 columns that had less than 5% of the data and didn't look critical for this task. Next, dropped another 8 columns that would have required additional analysis outside the scope of this project to be useful. Finally, removed duplicate asin rows that were very similar for both the product and review info. The data for the rest of the project included 15 columns with about 149k to 1.1 million rows.

The features of interest related to implicit feedback are title, also_buy, also_view and rank. While the features related to explicit feedback are overall and summary. The target feature is ‘overall’; which is the overall rating of the product.

### Exploratory Data Analysis
There were only 3 numeric (overall, price, and vote) and 1 boolean (verified) features in the data set. All of these features were skewed towards a particular value; however, none were correlated with each other. Ended up dropping the summary feature because it contained a lot of duplicate data with the target feature of overall rating.

The majority of the features were categories/objects/text. There are so many categories it wasn't possible to do one hot encoding because of memory issues. However, this doesn’t matter because the project will use processes designed specifically for recommendation systems to overcome these limitations.

### Preprocessing and Training
Using a subset of the data going forward for processing speed reasons. Would still limit the amount of data first in most scenarios to get feedback from several models faster. Performed an 80/20 train/test split on this smaller dataset.

Using the Root Mean Square Error (RMSE) to determine how well the model performed; i.e. how close the prediction was to the actual rating. The baseline model was found by hard coding a rating of 4 for all recommendations. This resulted in a RMSE of 1.1813. A lower score is better; with 0 being perfect.

### Modeling
Created three different versions of the recommendation model for predictions:
- v1 used the mean
- v2 used several custom similarity functions, and 
- v3 was a hybrid using both content-based and collaborative systems.

Version 1 using the mean wasn’t a good predictor. The collaborative mean (simple average of the users) was better (had a lower error) than the content mean (simple average of the product). But the best RMSE of 1.5540; which was much worse than the baseline model.

Version 2 used several custom similarity functions; such as pearson, euclidean, cosine, and jaccard. The best performing was euclidean with a RMSE of 1.2234. In all cases, they were: 
- very close to one another with no function being clearly better than another.
- better than the v1 predictions.
- worse than the best baseline prediction.

Version 3 employed a hybrid approach. In all cases, the estimates on the training data were significantly better than all previous notebook versions. The best performing was euclidean with a RMSE of 0.2027. However, the estimates on the testing data were not as good at 1.1413. Despite this, these predictions were still better than all previous versions; including the baseline.

Several popular recommendation systems return only 10 to 20 results. Therefore, creating a function to return the top x recommendations worked even better. Also, added other performance evaluation systems; such as precision, recall, f1-score, and precision-recall curve. The results were significantly better when choosing a random sample of 10 and only including recommendations that were predicted to be 5-stars. During analysis and testing these predictions were 60 to 90% accurate.

### Next Steps
The best performing model’s results could be improved by training on more of the data. Only about 1% of the available data was used because of time constraints. Then, this model could be deployed into production with a serverless architecture; such as using Amazon API Gateway, Lambda, and SageMaker.

Next, on the company’s website, do A/B testing to determine the most helpful recommendations by analyzing Click Through Rate (CTR), Conversion Rate (CR), and Return on Investment (ROI). Finally, the models, and the related recommendations, could continue to be enhanced by adjusting to new data, the user’s behavior, and purchases over time.

### References
1) Jianmo Ni 2018 [Amazon Review Data (2018)](https://nijianmo.github.io/amazon/).
2) Unata 2015 [Hands-on with PyData: How to Build a Minimal Recommendation Engine](https://www.youtube.com/watch?v=F6gWjOc1FUs).  
3) Scikit Learn 0.24.2 2021 [Precision-Recall](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html).  
