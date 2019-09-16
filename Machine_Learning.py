from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
#################################################################
#                      Cross Validation                         #
#################################################################
# Train/Test Split
x_train, x_test, y_train, y_test = train_test_split(Train.drop('SalePrice', axis=1), Train.SalePrice, random_state=4)

# Cross Validation
k_scores = []
for k in range(1,31):
	knn = KNeighborsClassifier(n_neighbors = k)
	scores = cross_val_score(knn, x_train, x_test, cv=10, scoring='accuracy')
	k_scores.append(scores.mean())
	


#################################################################
#                       Basic Methods                           #
#################################################################
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Linear Regression
lin_reg = LinearRegression()
rMSE = cross_val_score(lin_reg, x_train, x_test, cv=10, scoring='neg_mean_squared_error').mean()

# Ridge Regression
rid_reg = Ridge()
param = {'alpha': [0.001, 0.01, 0.1, 1, 2, 5, 10, 20]}
search = GridSearchCV(estimator=rid_reg, param, scoring='neg_mean_squared_error', cv=10)
search.fit(x_train, x_test)
# To see the parameters
print(search.best_params_)
print(search.best_score_)

# Lasso Regression
lasso_reg = Lasso()
param = {'alpha': [0.001, 0.01, 0.1, 1, 2, 5, 10, 20]}
search = GridSearchCV(estimator=lasso_reg, param, scoring='neg_mean_squared_error', cv=10)
search.fit(x_train, x_test)
# To see the parameters
print(search.best_params_)
print(search.best_score_)

# Random Forest CLASSIFIER (for continuous data, use RandomForestRegressor)
forest = RandomForestClassifier(n_estimators=100,   # The number of trees in the forest
                               bootstrap=True,      # Whether bootstrap samples are used when building trees
                               max_features='sqrt'  # The number of features to consider when looking for the best split
)
forest.fit(x_train, x_test)
# Actual class predictions
rf_predictions = forest.predict(test)
# Probabilities for each class
rf_probs = forest.predict_proba(test)[:, 1]
roc_value = roc_auc_score(test_labels, rf_probs)
	

#################################################################
#                    Elastic Net Regression                     #
#################################################################
from sklearn.linear_model import ElasticNet
elastic = ElasticNet(normalize=False)
# Use Cross Validation to Determine the best parameters
search = GridSearchCV(estimator=elastic, 
		      param_grid={'alpha':np.logspace(-5,2,8),'l1_ratio':[.2,.4,.6,.8]},
		      scoring='neg_mean_squared_error',          # ‘neg_mean_absolute_error’ or ‘accuracy’ or ‘roc_auc’or ‘r2’
		      n_jobs=1,                                  # Number of jobs to run in parallel
		      refit=True,                                # Refit an estimator using the best found parameters on the whole dataset
		      cv=10                                      # Determines the cross-validation splitting strategy
)
search.fit(Train.drop('SalePrice', axis=1), Train.SalePrice)
# Get the best estimator
best = search.best_estimator_
# Get the parameters of the best estimator
param = search.best_params_
best.fit(Train.drop('SalePrice', axis=1), Train.SalePrice)
# Get the coefficients of each predictor
coef_dict_baseline = {}
for coef, feat in zip(best.coef_, Train.drop('SalePrice', axis=1).columns):
    coef_dict_baseline[feat] = coef
print(coef_dict_baseline)



#################################################################
#                           XGBoost                             #
#################################################################
import XGBoost as xgb
from sklearn.metrics import mean_squared_error
## For CLASSIFICATION use XGBClassifier()
## For accuracy: from sklearn.metrics import accuracy_score
## print("Accuracy for model: %.2f" % (accuracy_score(y_test, pred) * 100))
xg_reg = xgb.XGBRegressor(objective ='reg:linear', 
			  colsample_bytree = 0.3,  # percentage of features used per tree. High value can lead to overfitting
			  learning_rate = 0.1,     # step size shrinkage used to prevent overfitting. Range is [0,1]
                          max_depth = 5, 	   # determines how deeply each tree is allowed to grow during any boosting round
			  alpha = 10,		   # L1 regularization on leaf weights
			  lambda = 5, 		   # L2 regularization on leaf weights and is smoother than L1 regularization
			  n_estimators = 10        # number of trees you want to build.
)
xg_reg.fit(X_train, y_train)
preds = xg_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))

# XGBoost with K-fold Cross Validation
## XGBoost specific data structure
data_dmatrix = xgb.DMatrix(data=X, label=y)
params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 5, 'alpha': 10}
cv_results = xgb.cv(dtrain=data_dmatrix,       
		    params=params, 
		    nfold=3,                   # number of cross validation sets you want to build
                    num_boost_round=50,        # denotes the number of trees you build
		    early_stopping_rounds=10,  # finishes training of the model early if the hold-out metric does not improve for a given number of rounds.
		    metrics="rmse", 	       # tells the evaluation metrics to be watched during CV
		    as_pandas=True,            # to return the results in a pandas DataFrame
		    seed=123	               # for reproducibility of results
)

# Feature Importance
xg_reg = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=10)
xgb.plot_importance(xg_reg)
plt.rcParams['figure.figsize'] = [20, 20]
plt.show()
