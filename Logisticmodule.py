import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

k_values = np.linspace(0.1, 10000, 50)  
accuracies = []

for k in k_values:
    model = LogisticRegression(C=1/k, max_iter=100)  
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))

import matplotlib.pyplot as plt
plt.plot(k_values, accuracies, marker='o')
plt.xlabel('k value (inverse of regularization)')
plt.ylabel('Accuracy')
plt.title('Effect of k on Classification Accuracy')
plt.grid()
plt.show()

#광학적인 비선형 효과를 이용한 암호화 기술

