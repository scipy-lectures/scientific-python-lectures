"""
================================================================
Use the RidgeCV and LassoCV to set the regularization parameter
================================================================


"""

############################################################
# Load the diabetes dataset
from sklearn.datasets import load_diabetes

data = load_diabetes()
X, y = data.data, data.target
print(X.shape)

############################################################
# Compute the cross-validation score with the default hyper-parameters
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, Lasso

for Model in [Ridge, Lasso]:
    model = Model()
    print(f"{Model.__name__}: {cross_val_score(model, X, y).mean()}")

############################################################
# We compute the cross-validation score as a function of alpha, the
# strength of the regularization for Lasso and Ridge
import numpy as np
import matplotlib.pyplot as plt

alphas = np.logspace(-3, -1, 30)

plt.figure(figsize=(5, 3))

for Model in [Lasso, Ridge]:
    scores = [cross_val_score(Model(alpha), X, y, cv=3).mean() for alpha in alphas]
    plt.plot(alphas, scores, label=Model.__name__)

plt.legend(loc="lower left")
plt.xlabel("alpha")
plt.ylabel("cross validation score")
plt.tight_layout()
plt.show()
