#### Kaggle Competition

# M5 Forecasting - Accuracy

## Estimate the unit sales of Walmart retail goods

<img src="images/md_img.jpg" alt="md_img" width="60%;" />

## 1. 개요

월마트의 판매량을 예측하는 대회

*Can you estimate, as precisely as possible, the point forecasts of the unit sales of various products sold in the USA by Walmart?*

https://www.kaggle.com/c/m5-forecasting-accuracy/overview/



## 2. Data

In the challenge, you are predicting item sales at stores in various locations for two 28-day time periods. Information about the data is found in the [M5 Participants Guide](https://mofc.unic.ac.cy/m5-competition/).

### Files

- `calendar.csv` - Contains information about the dates on which the products are sold.
- `sales_train_validation.csv` - Contains the historical daily unit sales data per product and store `[d_1 - d_1913]`
- `sample_submission.csv` - The correct format for submissions. Reference the [Evaluation](https://www.kaggle.com/c/m5-forecasting-accuracy/overview/evaluation) tab for more info.
- `sell_prices.csv` - Contains information about the price of the products sold per store and date.
- *`sales_train_evaluation.csv` - Available once month before competition deadline. Will include sales `[d_1 - d_1941]`*



## 3. Evaluation(평가)

The accuracy of the point forecasts will be evaluated using the **Root** **Mean Squared Scaled Error** (**RMSSE**), which is a variant of the well-known Mean Absolute Scaled Error (MASE) proposed by Hyndman and Koehler (2006)[[1\]](#_ftn1). The measure is calculated for each series as follows:

<img src="images/evaluation.jpg" alt="md_img" width="60%;" />



------

[[1\]](#_ftnref1) R. J. Hyndman & A. B. Koehler (2006). Another look at measures of forecast accuracy. International Journal of Forecasting 22(4), 679-688.



## 4. Preprocessing (데이터 전처리)





## 5. Result (결과)



