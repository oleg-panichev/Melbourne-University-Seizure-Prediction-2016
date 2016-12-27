# Melbourne University AES/MathWorks/NIH Seizure Prediction 2016: 5-th place
https://www.kaggle.com/c/melbourne-university-seizure-prediction

This code is the solution of team nullset, which took 5-th place in private leaderboard in Melbourne University AES/MathWorks/NIH Seizure Prediction.

### Dependencies
  * Python 2.7
  * scikit-learn
  * numpy
  * pandas
  * scipy
  * xgboost 

### AUC score on LeaderBoard
  
  public : 0.81423
  private: 0.79363

### Description
Post on the Kaggle forum:
https://www.kaggle.com/c/melbourne-university-seizure-prediction/forums/t/26117/solution-5th-place

### Preprocessing

The signal from each file was divided on epochs 30 seconds length without any filtration. From each epoch features were extracted. We have tried also 15 and 60 seconds epoch length but the results were worse.

### Feature extraction

We tried many features in different combinations during this competition, but not all of them were used in final models. Feature sets we’ve tried:

1. Deep’s kernel for features extraction.
2. Tony Reina’s kernel for features extraction.
3. Correlation between all channels (120 features).
4. Correlation between spectras of all channels (120 features).
5. Spectral features version 1: total energy (sum of all elements in range 0-30 Hz), energy in delta (0-3 Hz), theta (3-8 Hz), alpha (8-14 Hz) and beta (14-30 Hz) bands, energy in delta, theta, alpha and beta bands divided by total energy, ratios between energies of all bands.
6. Spectral features version 2: the same as Spectral features set 1 plus low and high gamma band were used in calculation of total energy, energy in bands and ratios between energies in bands. In addition, mean energy in bands was extracted.
7. Spectral features version 3: power spectral density was calculated for the whole epoch. Then it was divided on 1 Hz ranges and in each range energy was calculated (30 features).

### Fitting and cross-validation

Dividing signals on epochs allowed to increase training dataset size, so total number of observations No was equal to

No = Nf * Ne,

where Nf - number of 10-minute signals, Ne - number of epochs per one 10-minute signal.

For cross-validation stratified K-folds with 6 folds was used. It was extremely important to use K-fold without shuffling the data, otherwise the leakage is very high and cross-validation performance estimations are much higher. The leakage during shuffling was present because two neighboring epochs with very similar parameters were often present both in train and test sets.

Each model predicted probability of epoch belongs to preictal class. The final probability for 10-minute signal was calculated as mean of all probabilities for epochs in this signal.

We tried both patient-specific and non-patient-specific approaches on the same model but performance was higher when patient-specific approach was used.

### Models 

The final solution was an ensemble of best performing models (the first one is the best performing and the last one - is the worst):

1. AdaBoost with Decision Tree base estimator with combined feature sets 1, 4 and 5 .
2. Gradient Boosting Classifier with feature set 1.
3. Random Forest Classifier with feature set 2.
4. Random Forest Classifier with combined feature sets 1, 4 and 5.
5. GridSearch for “number of estimators” parameter for AdaBoost with Decision Tree base estimator with combined feature sets 1, 4 and 5.
6. Voting classifier with feature set 1. Voting was performed for 3 classifiers: GridSearch for “number of estimators” parameter for AdaBoost with Decision Tree base estimator; Gradient Boosting Classifier and Bagging Classifier.
7. XGBoost Classifier with feature set 1.

AdaBoost with Decision Tree base estimator with combined feature sets 1, 4 and 5 showed the highest performance among the models.

Final result P was calculated as follows:

P = 1/13 * (3*Model 1 + Model 2 + Model 3 + 3*Model 4 + 3*Model 5 + Model 6 + Model 7)

