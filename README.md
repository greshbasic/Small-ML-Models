These are just a few small machine learning models I have made while teaching myself TensorFlow

temp_conv_ai.py : 
    already trained with a few data points, user inputs a temp in C and is given the model's guess as to the equivalent temperature in F
    
    example:
    ```
    Input a temperature in Celsius: 10
    1/1 [==============================] - 0s 50ms/step
    The model's prediction is: 47.13
    The actual value is: 50.00
    This is a percent error of: 5.75%
    ```
    
transformation-guesser.py:
    takes in an array of numbers, then an array of those numbers after a transformation, then takes a guess of what the transformation's   
    affect is on a user inputted number.

    example:
    ```
    Please enter your input array, type 'q' when done: 
    1
    2
    3
    q
    Your input array is: [1, 2, 3]
    
    Please enter your transformed array: 
    2
    3
    4
    Your transformed array is: [2, 3, 4]

    Input a number for the model to take a guess on: 6
    1/1 [==============================] - 0s 50ms/step
    
    The model's prediction is: 6.999999523162842
