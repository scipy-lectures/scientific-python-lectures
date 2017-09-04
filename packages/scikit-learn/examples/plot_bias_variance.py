"""
====================================
Bias and variance of polynomial fit
====================================

Fit polynomes of different degrees to a dataset: for too small a degree,
the model *underfits*, while for too large a degree, it overfits.

"""

import numpy as np
import matplotlib.pyplot as plt


def test_func(x, err=0.5):
    return np.random.normal(10 - 1. / (x + 0.1), err)


def compute_error(x, y, p):
    yfit = np.polyval(p, x)
    return np.sqrt(np.mean((y - yfit) ** 2))


############################################################
# A polynomial regression
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

############################################################
# A simple figure to illustrate the problem

n_samples = 8

np.random.seed(0)
x = 10 ** np.linspace(-2, 0, n_samples)
y = test_func(x)

xfit = np.linspace(-0.2, 1.2, 1000)

titles = ['d = 1 (under-fit; high bias)',
            'd = 2',
            'd = 6 (over-fit; high variance)']
degrees = [1, 2, 6]

fig = plt.figure(figsize = (9, 3.5))
fig.subplots_adjust(left = 0.06, right=0.98,
                    bottom=0.15, top=0.85,
                    wspace=0.05)

for i, d in enumerate(degrees):
    ax = fig.add_subplot(131 + i, xticks=[], yticks=[])
    ax.scatter(x, y, marker='x', c='k', s=50)

    model = make_pipeline(PolynomialFeatures(d), LinearRegression())
    model.fit(x[:, np.newaxis], y)
    ax.plot(xfit, model.predict(xfit[:, np.newaxis]), '-b')

    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(0, 12)
    ax.set_xlabel('house size')
    if i == 0:
        ax.set_ylabel('price')

    ax.set_title(titles[i])


############################################################
# Generate a larger dataset
from sklearn.model_selection import train_test_split

n_samples = 200
test_size = 0.4
error = 1.0

# randomly sample the data
np.random.seed(1)
x = np.random.random(n_samples)
y = test_func(x, error)

# split into training, validation, and testing sets.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

# show the training and validation sets
plt.figure(figsize=(6, 4))
plt.scatter(x_train, y_train, color='red', label='Training set')
plt.scatter(x_test, y_test, color='blue', label='Test set')
plt.title('The data')
plt.legend(loc='best')

############################################################
# Now plot a validation curve
from sklearn.model_selection import train_test_split
from sklearn import metrics

degrees = np.arange(1, 21)
train_err = np.zeros(len(degrees))
validation_err = np.zeros(len(degrees))

for i, d in enumerate(degrees):
    model = make_pipeline(PolynomialFeatures(d),
                LinearRegression(fit_intercept=False, normalize=True))
    model.fit(x_train[:, np.newaxis], y_train)

    train_err[i] = np.sqrt(metrics.mean_squared_error(y_train,
        model.predict(x_train[:, np.newaxis])))
    validation_err[i] = np.sqrt(metrics.mean_squared_error(y_test,
        model.predict(x_test[:, np.newaxis])))

plt.figure(figsize=(6, 4))
plt.plot(degrees, validation_err, lw=2, label='cross-validation error')
plt.plot(degrees, train_err, lw=2, label='training error')
plt.plot([0, 20], [error, error], '--k', label='intrinsic error')

plt.legend(loc=0)
plt.xlabel('degree of fit')
plt.ylabel('rms error')
plt.title('Validation curve')
plt.tight_layout()

plt.show()

