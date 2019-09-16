from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
#################################################################
#                      Cross Validation                         #
#################################################################
# Train/Test Split
x_train, x_test, y_train, y_test = train_test_split(Train.drop('SalePrice'), Train.SalePrice, random_state=4)

# Cross Validation
k_scores = []
for k in range(1,31):
	knn = KNeighborsClassifier(n_neighbors = k)
	scores = cross_val_score(knn, x_train, x_test, cv=10, scoring='accuracy')
	k_scores.append(scores.mean())
	
