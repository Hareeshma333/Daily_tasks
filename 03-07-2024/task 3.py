from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score

# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)

# Decision Tree Classifier
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)
y_pred_dt = dt_clf.predict(X_test)

# Random Forest Classifier
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)

# Confusion matrix
conf_matrix_log_reg = confusion_matrix(y_test, y_pred_log_reg)
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

# Cross-validation
cv_log_reg = cross_val_score(log_reg, X, y, cv=10)
cv_dt = cross_val_score(dt_clf, X, y, cv=10)
cv_rf = cross_val_score(rf_clf, X, y, cv=10)

# Evaluation metrics
def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return accuracy, precision, recall, f1

eval_log_reg = evaluate_model(y_test, y_pred_log_reg)
eval_dt = evaluate_model(y_test, y_pred_dt)
eval_rf = evaluate_model(y_test, y_pred_rf)

# Print results
print("Logistic Regression:", eval_log_reg)
print("Decision Tree:", eval_dt)
print("Random Forest:", eval_rf)
print("Logistic Regression CV:", cv_log_reg.mean(), cv_log_reg.std())
print("Decision Tree CV:", cv_dt.mean(), cv_dt.std())
print("Random Forest CV:", cv_rf.mean(), cv_rf.std())
