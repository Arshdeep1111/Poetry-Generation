# Import necessary libraries
import tensorflow as tf  # For loading and working with the trained model
import numpy as np  # For numerical computations
import random  # For generating random indices
from train_poetry import *  # Import variables and functions from the training script

# Load the pre-trained text generation model
model = tf.keras.models.load_model('text_generator.h5')

# Function to sample the next character index based on predictions and temperature
def sample(preds, temperature=1.0):
    # Convert predictions to a NumPy array and adjust based on temperature
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature  # Apply temperature scaling
    exp_preds = np.exp(preds)  # Compute exponentials of adjusted predictions
    preds = exp_preds / np.sum(exp_preds)  # Normalize to get probabilities
    probas = np.random.multinomial(1, preds, 1)  # Sample based on probabilities
    return np.argmax(probas)  # Return the index of the chosen character

# Function to generate text of a given length with a specified temperature
def generate_text(length, temperature):
    # Randomly choose a starting index in the text
    start_index = random.randint(0, len(text) - seq_length - 1)
    generated = ''  # Initialize the generated text
    sentence = text[start_index:start_index + seq_length]  # Extract initial seed sequence
    generated += sentence  # Add the seed sequence to the generated text

    # Generate `length` characters
    for i in range(length):
        # Initialize input array for the model
        x = np.zeros((1, seq_length, len(characters)))
        
        # Populate the input array with the current sentence
        for t, character in enumerate(sentence):
            x[0, t, char_to_index[character]] = 1  # Encode each character in the sentence
        
        # Predict the probabilities for the next character
        predictions = model.predict(x, verbose=0)[0]
        
        # Sample the next character index using the `sample` function
        next_index = sample(predictions, temperature)
        
        # Map the index to the corresponding character
        next_character = index_to_char[next_index]
        
        # Append the predicted character to the generated text
        generated += next_character
        
        # Update the sentence by removing the first character and adding the new one
        sentence = sentence[1:] + next_character
    
    return generated  # Return the generated text

# Generate and print text at various temperatures
print('--------------0.2------------')
print(generate_text(300, 0.2))  # Generate 300 characters with a low temperature
print('--------------0.4------------')
print(generate_text(300, 0.4))  # Generate 300 characters with a slightly higher temperature
print('--------------0.6------------')
print(generate_text(300, 0.6))  # Generate 300 characters with a moderate temperature
print('--------------0.8------------')
print(generate_text(300, 0.8))  # Generate 300 characters with a higher temperature
print('--------------1.0------------')
print(generate_text(300, 1.0))  # Generate 300 characters with the highest temperature
