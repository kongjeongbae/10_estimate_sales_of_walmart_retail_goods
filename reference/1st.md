https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/138881

Few thoughts about M5 competition

posted in [M5 Forecasting - Accuracy](https://www.kaggle.com/c/m5-forecasting-accuracy) 



I'll try to prepare few kernels for this competition.

Here is the 1st one with simple fe and "hints" how to work with kaggle memory limit.
https://www.kaggle.com/kyakovlev/m5-simple-fe

------

Here is some initial thoughts about this competition:

1. Memory limits
   1.1 You have to find a way to have more features (splits by "type"/date limits)
   1.2 Grid creation (there are several ways to make grid)
   1.3 Careful work with dtypes
   1.4 lags nan cutoff or not
2. Models
   2.1. I think that the winner will be NN because of loss functions flexibility
   2.2 Catboost works here but need much more memory and time than lgbm (to have same accuracy)
   2.3 ARIMA?
   3.4 Ensemble and stacking works
3. Cost/Loss function (most important)
   3.1 You have to create own one for green zone (not now)
   3.2 Weights with rmse can "simulate" cost/loss function

------

Current score (0.51756)

- Own cost

- Single lgbm

- 0.54 local WRMSSE
  +/- 50 features

- n_rounds(n_estimators) - mean of last 3 months

  - ```
    train -> >1913 - 28*4 -> validate <1913 - 28*3 -> get boosting rounds
    train -> >1913 - 28*3 -> validate <1913 - 28*2 -> get boosting rounds
    train -> >1913 - 28*2 -> validate <1913 - 28*1 -> get boosting rounds
    ```

- 0.05 LR(learning rate)

- params differ a lot from current public kernels