import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np

celsius_q = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit_a = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

l0 = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([l0])

model.compile(
              loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.1)
              )

history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print("Finished training the model...")

user_input = int(input("Input a temperature in Celsius: "))

guess = model.predict([user_input])[0][0]
print(f"\nThe model's prediction is: {guess:.2f}")

actual_value = (1.8 * user_input) + 32
print(f"The actual value is: {actual_value:.2f}")

percent_error = (abs(guess - actual_value)/actual_value) * 100
print(f"This is a percent error of: {percent_error:.2f}%")
