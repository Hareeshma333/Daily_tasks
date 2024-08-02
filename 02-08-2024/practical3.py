import tensorflow as tf
from tensorflow.keras.losses import Loss

# Define a custom loss function
class CustomLoss(Loss):
    def __init__(self, threshold=1.0, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold

    def call(self, y_true, y_pred):
        # Compute the absolute error
        absolute_error = tf.abs(y_true - y_pred)
        # Apply custom penalty: increase loss if error exceeds threshold
        custom_loss = tf.where(absolute_error > self.threshold,
                               2 * absolute_error,
                               absolute_error)
        return tf.reduce_mean(custom_loss)

# Example usage of the custom loss function
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss=CustomLoss())

# Example data
import numpy as np
X_train = np.random.rand(100, 10)
y_train = np.random.rand(100, 1)

# Train the model
history = model.fit(X_train, y_train, epochs=5)

# Print the model summary
model.summary()

# Print the training history
print("Training history:")
print(history.history)
