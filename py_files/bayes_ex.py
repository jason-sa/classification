from skopt import BayesSearchCV

# parameter ranges are specified by one of below
from skopt.space import Real, Categorical, Integer

from sklearn.datasets import load_iris 
from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression


X, y = load_iris(True) 
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=0)

# log-uniform: understand as search over p = exp(x) by varying x
opt = BayesSearchCV( KNN(), { 'C': Real(1e-6, 1e+6, prior='log-uniform'), 'gamma': Real(1e-6, 1e+1, prior='log-uniform'), 'degree': Integer(1,8), 'kernel': Categorical(['linear', 'poly', 'rbf']), }, n_iter=32 )

# executes bayesian optimization
opt.fit(X_train, y_train)

# model can be saved, used for predictions or scoring
print(opt.score(X_test, y_test))