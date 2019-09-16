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
#                    Elastic Net Regression                     #
#################################################################
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
elastic = ElasticNet(normalize=False)
# Use Cross Validation to Determine the best parameters
search = GridSearchCV(estimator=elastic, 
		      param_grid={'alpha':np.logspace(-5,2,8),'l1_ratio':[.2,.4,.6,.8]},
		      scoring='neg_mean_squared_error',
		      n_jobs=1,
		      refit=True,
		      cv=10
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
import XGBoost



