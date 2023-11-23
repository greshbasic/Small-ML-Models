import tensorflow as tf
import numpy as np
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

done = False
input_array = []
transformed_array = []

print("\nThis model will predict your transformations. Firstly you will input some numbers.")
print("Then you will input the result of your transformation on said numbers.")
print("For example, if your transformation was the doubling of numbers, you would input 1, 5, 8, 13 as the input array")
print("Then you would input: 2, 10, 16, 26 as the transformed array.\n")

print("Please enter your input array, type 'q' when done: ")
while not done:
    current_input = input()
    if current_input == "q":
        done = True
    else:
        input_array.append(int(current_input))
print(f"Your input array is: {input_array}\n")
        
done = False
i = 0
print("Please enter your transformed array: ")
while not done and i < len(input_array):
    current_input = input()
    if current_input == "q":
        done = True
    else:
        transformed_array.append(int(current_input))
        i += 1
print(f"Your transformed array is: {transformed_array}\n")
    
input_q = np.array(input_array, dtype=float)
transformed_a = np.array(transformed_array, dtype=float)

l0 = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([l0])

model.compile(
              loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.1)
              )

history = model.fit(input_array, transformed_array, epochs=500, verbose=False)
print("\nFinished training the model...")

user_input = int(input("Input a number for the model to take a guess on: "))

guess = model.predict([user_input])[0][0]
print(f"\nThe model's prediction is: {guess}\n")
