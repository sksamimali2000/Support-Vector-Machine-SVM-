# ⚡ Support Vector Machines (SVM) Implementation and Visualization

This project demonstrates multiple **SVM tasks** using scikit-learn:
1. Linear SVM classification with simple 2D data  
2. SVM classification on the Iris dataset (visualized decision boundary)  
3. SVM regression on the Boston Housing dataset using Grid Search for hyperparameter tuning

---

## ✅ Linear SVM Classification (Toy Data)

```python
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
# Sample data
X = np.array([[1,1],[2,1],[1,2],[1.5,1.5],[3,4],[2,5],[4,3],[7,2],[3,5],[2,6],[6,2],[3,4],[4,4]])
y = [0,1,0,0,1,1,1,1,1,1,1,1,1]
# Visualize data
plt.scatter(X[:,0], X[:,1], c=y)
plt.show()

# Train linear SVM
svcLinear = SVC(kernel='linear', C=1e5).fit(X, y)

# Decision boundary
x1 = np.array([0, 5])
x2 = -1 * (svcLinear.intercept_ + svcLinear.coef_[0][0] * x1) / svcLinear.coef_[0][1]
plt.plot(x1, x2)
plt.scatter(X[:,0], X[:,1], c=y)
plt.axis([0, 8, 0, 8])
plt.show()
```

✅ SVM Classification on Iris Dataset
```Python
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

iris = datasets.load_iris()
x = iris.data[:, :2]  # Using first two features
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y)

clf = svm.SVC(kernel='linear')
clf.fit(x_train, y_train)
print("Test Accuracy:", clf.score(x_test, y_test))

# Create meshgrid for visualization
def makegrid(x1, x2, h=0.02):
    x1_min, x1_max = x1.min() - 1, x1.max() + 1
    x2_min, x2_max = x2.min() - 1, x2.max() + 1
    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, h),
                         np.arange(x2_min, x2_max, h))
    return xx, yy

xx, yy = makegrid(x[:, 0], x[:, 1])
predictions = clf.predict(np.c_[xx.ravel(), yy.ravel()])

plt.scatter(xx.ravel(), yy.ravel(), c=predictions, alpha=0.3)
plt.scatter(x[:, 0], x[:, 1], c=y)
plt.show()
```

✅ SVM Regression on Boston Housing Dataset
```Pyhton
from sklearn import datasets
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVR

boston = datasets.load_boston()
x = boston.data
y = boston.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

# SVR with RBF kernel
clf_rbf = SVR(kernel="rbf")
clf_rbf.fit(x_train, y_train)
print("RBF Kernel SVR Score:", clf_rbf.score(x_test, y_test))

# SVR with Linear kernel
clf_linear = SVR(kernel="linear")
clf_linear.fit(x_train, y_train)
print("Linear Kernel SVR Score:", clf_linear.score(x_test, y_test))

# Grid Search for Hyperparameter Tuning
grid = {
    'C': [1e2, 1e3, 5e3, 1e4, 5e4, 1e5],
    'gamma': [1e-3, 5e-4, 1e-4, 5e-3]
}

clf = SVR(kernel='rbf')
grid_search = GridSearchCV(clf, grid)
grid_search.fit(x_train, y_train)

print("Best SVR Estimator:", grid_search.best_estimator_)
print("Best SVR Test Score:", grid_search.score(x_test, y_test))
```

⚙️ Requirements

Python >= 3.7

numpy

matplotlib

scikit-learn

Install dependencies using:

pip install numpy matplotlib scikit-learn


Made with ❤️ by Sk Samim Ali
