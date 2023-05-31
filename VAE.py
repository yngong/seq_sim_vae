from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
#from keras.preprocessing.sequence import pad_sequences
from keras import layers
from keras.models import Model
from keras import backend as K
import keras
import numpy as np
from Bio import SeqIO

# Read sequences from a FASTA file
sequences = []
for seq_record in SeqIO.parse("sequences.fasta", "fasta"):
    sequences.append(str(seq_record.seq).upper())

# Parameters
max_len = 29903
embedding_dim = 10
intermediate_dim = 256
latent_dim = 2

# Prepare tokenizer
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(sequences)

# Convert sequences to integer sequences
sequences_int = tokenizer.texts_to_sequences(sequences)

# Pad sequences
sequences_padded = pad_sequences(sequences_int, maxlen=max_len)

# Create an embedding layer
embedding_layer = layers.Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=embedding_dim, input_length=max_len)

# Encoder
inputs = keras.Input(shape=(max_len,))
x = embedding_layer(inputs)
h = layers.GRU(intermediate_dim)(x)
z_mean = layers.Dense(latent_dim)(h)
z_log_sigma = layers.Dense(latent_dim)(h)

# Sampling function
def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=0.1)
    return z_mean + K.exp(z_log_sigma) * epsilon

z = layers.Lambda(sampling)([z_mean, z_log_sigma])

# Encoder model
encoder = keras.Model(inputs, [z_mean, z_log_sigma, z], name='encoder')

# Decoder
latent_inputs = keras.Input(shape=(latent_dim,), name='z_sampling')
x = layers.Dense(max_len*embedding_dim, activation='relu')(latent_inputs)
x = layers.Reshape((max_len, embedding_dim))(x)
outputs = layers.GRU(len(tokenizer.word_index)+1, return_sequences=True, activation='softmax')(x)

# Decoder model
decoder = keras.Model(latent_inputs, outputs, name='decoder')

# VAE model
outputs = decoder(encoder(inputs)[2])
vae = keras.Model(inputs, outputs, name='vae_mlp')

# Compile
vae.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Prepare targets
sequences_targets = np.expand_dims(sequences_padded, -1)

# Train
vae.fit(sequences_padded, sequences_targets, epochs=15, batch_size=32)

vae.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'

# Sample from the latent space and decode
# Generate sequences from the latent space and decode
z_samples = np.random.normal(size=(1, latent_dim))
#print(z_samples)
decoded_sequences = decoder.predict(z_samples)
#print(decoded_sequences)
# Choose the base with the highest probability
decoded_sequences = decoded_sequences.argmax(axis=2)
#print(decoded_sequences)

# Map from indices to bases
index_to_base = {index+1: base for base, index in tokenizer.word_index.items()}

# Convert the indices back to bases
decoded_sequences = [[index_to_base.get(index, '') for index in seq] for seq in decoded_sequences]

# Join the bases into strings
decoded_sequences = [''.join(seq) for seq in decoded_sequences]

print(decoded_sequences)

