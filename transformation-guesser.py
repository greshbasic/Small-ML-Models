import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

def file_mode():
    print("File Mode...")
    input_array = []
    transformed_array = []
    transformed_index = 0
    
    with open('transformation_data.txt', "r") as file:
        stripped_lines = []
        lines = file.readlines()
        for idx, line in enumerate(lines[1:]):
            stripped_line = line.strip()
            if stripped_line == "TRANSFORMED":
                transformed_index = idx
            else:
                stripped_lines.append(int(line))
        input_array = stripped_lines[:transformed_index]
        transformed_array = stripped_lines[transformed_index:]
        print(f"Input Array: {input_array}")
        print(f"Input Array: {transformed_array}")
        return input_array, transformed_array
    
def input_mode():
    print("Input Mode...")
    done = False
    input_array = []
    transformed_array = []

    print("--------------------------------------------------------------------------------------------------------------------")
    print("| This model will predict your transformations. Firstly you will input some numbers.                               |")
    print("| Then you will input the result of your transformation on said numbers.                                           |")
    print("| For example, if your transformation was the doubling of numbers, you would input 1, 5, 8, 13 as the input array  |")
    print("| Then you would input: 2, 10, 16, 26 as the transformed array.                                                    |")
    print("--------------------------------------------------------------------------------------------------------------------\n")
    print("Please enter your input array, type 'q' when done: ")
    while not done:
        current_input = input()
        if current_input == "q":
            done = True
        else:
            input_array.append(float(current_input))
    
    print(f"Input Array: {input_array}")
            
    i = 0
    print("Please enter your transformed array: ")
    while i < len(input_array):
        current_input = input()
        transformed_array.append(float(current_input))
        i += 1
    print(f"Transformed Array: {transformed_array}")
    return input_array, transformed_array


mode = (input("Type 'i' for input mode or 'f' for file mode: ")).lower()
if mode == "i":
    input_array, transformed_array = input_mode()
elif mode == "f":
    input_array, transformed_array = file_mode()
else:
    print("Invalid mode.")
    quit
    
    
print("Training the model...")

input_q = np.array(input_array, dtype=float)
transformed_a = np.array(transformed_array, dtype=float)

l0 = tf.keras.layers.Dense(units=4, input_shape=[1])
model = tf.keras.Sequential([l0])

model.compile(
            loss='mean_squared_error',
            optimizer=tf.keras.optimizers.Adam(0.1)
            )

history = model.fit(input_array, transformed_array, epochs=500, verbose=False)
print("Finished training the model...\n")

user_input = float(input("Input a number for the model to take a guess on: "))

guess = model.predict([user_input])[0][0]
rounded_guess = round(guess, 2)
print(f"\nThe model's prediction is: {rounded_guess}")

w = l0.get_weights()[0][0][0]
b = l0.get_weights()[1][0]

operation = ""
rounded_w = round(w,2)
rounded_b = round(b,2)
if rounded_b <= 0:
    operation = "-"
    rounded_b *= -1
else:
    operation = "+"

string = f"The model guesses that your transformation is: y = {rounded_w}x"
if rounded_b != 0:
    string += f" {operation} {rounded_b}\n"
else:
    string += "\n"
print(string)


# Matlablib
plt.title("Transformation Trend")
plt.xlabel("Inputted Values")
plt.ylabel("Transformed Values")
plt.scatter(input_array, transformed_array)
input_array.append(user_input)
transformed_array.append(guess)
plt.scatter(user_input,rounded_guess, c="red")
plt.axis([min(input_array)-1,max(input_array)+1,min(transformed_array)-1,max(transformed_array)+1])
plt.show()