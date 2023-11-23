import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np


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
rounded_guess = round(guess, 2)
print(f"\nThe model's prediction is: {rounded_guess}")

w = l0.get_weights()[0][0][0]
b = l0.get_weights()[1][0]

rounded_w = round(w,2)
rounded_b = round(b,2)

print(f"The model guesses that your transformation is: {rounded_w}x + {rounded_b}\n")