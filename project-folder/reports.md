## Logistic Regression

````
              precision    recall  f1-score   support

   malignant       0.98      0.98      0.98        42
      benign       0.99      0.99      0.99        72

    accuracy                           0.98       114
   macro avg       0.98      0.98      0.98       114
weighted avg       0.98      0.98      0.98       114

````

Confusion Matrix (rows=true [malignant=0, benign=1], cols=pred):

[[41  1]
 [ 1 71]]



## Decision Tree

````
              precision    recall  f1-score   support

   malignant       0.85      0.93      0.89        42
      benign       0.96      0.90      0.93        72

    accuracy                           0.91       114
   macro avg       0.90      0.92      0.91       114
weighted avg       0.92      0.91      0.91       114

````

Confusion Matrix (rows=true [malignant=0, benign=1], cols=pred):

[[39  3]
 [ 7 65]]



## kNN

````
              precision    recall  f1-score   support

   malignant       0.95      0.93      0.94        42
      benign       0.96      0.97      0.97        72

    accuracy                           0.96       114
   macro avg       0.96      0.95      0.95       114
weighted avg       0.96      0.96      0.96       114

````

Confusion Matrix (rows=true [malignant=0, benign=1], cols=pred):

[[39  3]
 [ 2 70]]



## Naive Bayes (Gaussian)

````
              precision    recall  f1-score   support

   malignant       0.93      0.90      0.92        42
      benign       0.95      0.96      0.95        72

    accuracy                           0.94       114
   macro avg       0.94      0.93      0.93       114
weighted avg       0.94      0.94      0.94       114

````

Confusion Matrix (rows=true [malignant=0, benign=1], cols=pred):

[[38  4]
 [ 3 69]]



## Random Forest (Ensemble)

````
              precision    recall  f1-score   support

   malignant       0.93      0.93      0.93        42
      benign       0.96      0.96      0.96        72

    accuracy                           0.95       114
   macro avg       0.94      0.94      0.94       114
weighted avg       0.95      0.95      0.95       114

````

Confusion Matrix (rows=true [malignant=0, benign=1], cols=pred):

[[39  3]
 [ 3 69]]



## XGBoost (Ensemble)

````
              precision    recall  f1-score   support

   malignant       0.97      0.90      0.94        42
      benign       0.95      0.99      0.97        72

    accuracy                           0.96       114
   macro avg       0.96      0.95      0.95       114
weighted avg       0.96      0.96      0.96       114

````

Confusion Matrix (rows=true [malignant=0, benign=1], cols=pred):

[[38  4]
 [ 1 71]]

