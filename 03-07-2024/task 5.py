from sklearn.metrics import plot_confusion_matrix

# Plot confusion matrix for Random Forest Classifier
plot_confusion_matrix(rf_clf, X_test, y_test)
plt.title('Confusion Matrix for Random Forest Classifier')
plt.show()
