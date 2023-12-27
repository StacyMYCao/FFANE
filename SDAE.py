import os
import tensorflow as tf
physical_devices = tf.config.list_physical_devices()
print(physical_devices)
print(tf.test.is_gpu_available())
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
#from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import EarlyStopping

def stacked_denoising_autoencoder_node_embedding(info_matrix, output_dim, encoder_params, learning_rate=0.001, batch_size=32, epochs=100, noise_factor=0.1):
    # Normalize the input matrix
    normalized_matrix = (info_matrix - np.mean(info_matrix, axis=0)) / np.std(info_matrix, axis=0)
    
    # Add noise to the input data
    noisy_matrix = normalized_matrix + noise_factor * np.random.normal(size=normalized_matrix.shape)
    noisy_matrix = np.clip(noisy_matrix, 0., 1.)  # Clip values to [0, 1]
    
    # Split data into training and validation sets
    X_train, X_val = train_test_split(noisy_matrix, test_size=0.2, random_state=42)
    y_train, y_val = train_test_split(normalized_matrix, test_size=0.2, random_state=42)
    
    # Define the encoder architecture
    encoder_input = Input(shape=(info_matrix.shape[1],))
    encoded = encoder_input
    
    for units in encoder_params:
        encoded = Dense(units, activation='relu')(encoded)
    
    # Define the decoder architecture
    decoded = encoded
    
    for units in reversed(encoder_params[:-1]):
        decoded = Dense(units, activation='relu')(decoded)
    
    decoded = Dense(info_matrix.shape[1], activation='linear')(decoded)
    
    # Create the denoising autoencoder model
    with tf.device('/GPU:0'):  # Specify the GPU device
        pass
        autoencoder = Model(encoder_input, decoded)
    
        # Compile the model
        optimizer = Adam(learning_rate=learning_rate)
        autoencoder.compile(optimizer=optimizer, loss='mean_squared_error')
    
        # Implement early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    
        # Train the denoising autoencoder
        autoencoder.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size, epochs=epochs, verbose=1,
            callbacks=[early_stopping]
        )
    
    # Obtain the node embeddings using the encoder part of the model
    encoder = Model(encoder_input, encoded)
    node_embeddings = encoder.predict(normalized_matrix)
    
    # Reduce the dimensionality of the node embeddings
    reduced_node_embeddings = Dense(output_dim, activation='relu')(node_embeddings)
    
    # Compute the loss value
    loss = autoencoder.evaluate(noisy_matrix, normalized_matrix, verbose=0)
    if (isinstance(type(reduced_node_embeddings),np.float64) or isinstance(type(reduced_node_embeddings),np.float32)) is False:
        reduced_node_embeddings = reduced_node_embeddings.numpy().astype(np.float64)
        # tf.cast(reduced_node_embeddings, dtype=tf.float64)
    return reduced_node_embeddings, loss
