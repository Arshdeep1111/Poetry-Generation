# Import necessary libraries
import numpy as np  # For numerical computations
import tensorflow as tf  # For building and training the neural network
from tensorflow.keras.models import Sequential  # For creating a sequential model
from tensorflow.keras.layers import LSTM, Dense, Activation  # Required layers for the model
from tensorflow.keras.optimizers import RMSprop  # Optimizer for model training

# Load the dataset from a URL (Shakespeare's text)
filepath = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# Read the text file and decode it to a string, converting to lowercase
text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()

# Extract a specific portion of the text (from character 300,000 to 800,000)
text = text[300000:800000]

# Create a sorted list of unique characters in the text
characters = sorted(set(text))

# Map each character to a unique index
char_to_index = dict((c, i) for i, c in enumerate(characters))

# Map each index back to its corresponding character
index_to_char = dict((i, c) for i, c in enumerate(characters))

# Define the length of input sequences and step size for creating sequences
seq_length = 40  # Length of each input sequence
step_size = 3  # Step size for moving the window

# Prepare lists to store input sequences (sentences) and their corresponding next characters
sentences = []
next_characters = []

# Iterate through the text to create overlapping sequences
for i in range(0, len(text) - seq_length, step_size):
    sentences.append(text[i: i + seq_length])  # Extract a sequence of length `seq_length`
    next_characters.append(text[i + seq_length])  # Store the next character

# Initialize input (x) and output (y) arrays as zero matrices
x = np.zeros((len(sentences), seq_length, len(characters)), dtype=np.bool)  # Input: 3D array
y = np.zeros((len(sentences), len(characters)), dtype=np.bool)  # Output: 2D array

# Populate the input and output arrays
for i, sentence in enumerate(sentences):
    for t, character in enumerate(sentence):
        x[i, t, char_to_index[character]] = 1  # Encode characters in the input sequence
    y[i, char_to_index[next_characters[i]]] = 1  # Encode the target character

# Define a function to build, train, and save the LSTM model
def train_model():
    model = Sequential()  # Initialize a sequential model
    
    # Add an LSTM layer with 128 units and input shape matching the input data
    model.add(LSTM(128, input_shape=(seq_length, len(characters))))
    
    # Add a Dense layer with output size equal to the number of unique characters
    model.add(Dense(len(characters)))
    
    # Add a softmax activation layer for multiclass classification
    model.add(Activation('softmax'))
    
    # Compile the model with categorical crossentropy loss and RMSprop optimizer
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.01))
    
    # Train the model on the input data
    model.fit(x, y, batch_size=256, epochs=4)
    
    # Save the trained model to a file
    model.save('text_generator.h5')

# Train the model if the script is run directly
if __name__ == '__main__':
    train_model()
