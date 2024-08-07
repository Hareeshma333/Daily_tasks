from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin_reg = lin_reg.predict(X_test)

# Decision Tree Regressor
dt_reg = DecisionTreeRegressor()
dt_reg.fit(X_train, y_train)
y_pred_dt_reg = dt_reg.predict(X_test)

# Evaluation metrics
r2_lin_reg = r2_score(y_test, y_pred_lin_reg)
mse_lin_reg = mean_squared_error(y_test, y_pred_lin_reg)

r2_dt_reg = r2_score(y_test, y_pred_dt_reg)
mse_dt_reg = mean_squared_error(y_test, y_pred_dt_reg)

# Cross-validation
cv_lin_reg = cross_val_score(lin_reg, X, y, cv=10, scoring='r2')
cv_dt_reg = cross_val_score(dt_reg, X, y, cv=10, scoring='r2')

# Print results
print("Linear Regression R2:", r2_lin_reg)
print("Linear Regression MSE:", mse_lin_reg)
print("Decision Tree Regressor R2:", r2_dt_reg)
print("Decision Tree Regressor MSE:", mse_dt_reg)
print("Linear Regression CV:", cv_lin_reg.mean(), cv_lin_reg.std())
print("Decision Tree Regressor CV:", cv_dt_reg.mean(), cv_dt_reg.std())
