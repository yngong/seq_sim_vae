from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras import layers
from keras.models import Model
from keras import backend as K
import keras
import numpy as np
from Bio import SeqIO
from keras.models import load_model

# Read sequences from a FASTA file
# sequences = []
# for seq_record in SeqIO.parse("sequences.fasta", "fasta"):
#     sequences.append(str(seq_record.seq).upper())

# Read sequences from a FASTA file
sequences = []
for seq_record in SeqIO.parse("sequences.fasta", "fasta"):
    seq = str(seq_record.seq).upper()
    seq = seq.replace('R', 'N')  # Replace ambiguous nucleotide R with N
    seq = seq.replace('Y', 'N')  # Replace ambiguous nucleotide Y with N
    seq = seq.replace('S', 'N')  # Replace ambiguous nucleotide S with N
    seq = seq.replace('W', 'N')  # Replace ambiguous nucleotide W with N
    seq = seq.replace('K', 'N')  # Replace ambiguous nucleotide K with N
    seq = seq.replace('M', 'N')  # Replace ambiguous nucleotide M with N
    seq = seq.replace('B', 'N')  # Replace ambiguous nucleotide B with N
    seq = seq.replace('D', 'N')  # Replace ambiguous nucleotide D with N
    seq = seq.replace('H', 'N')  # Replace ambiguous nucleotide H with N
    seq = seq.replace('V', 'N')  # Replace ambiguous nucleotide V with N
    sequences.append(seq)

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
h = layers.Bidirectional(layers.GRU(intermediate_dim, return_sequences=True))(x)
h = layers.GRU(intermediate_dim)(h)
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
outputs = layers.TimeDistributed(layers.Dense(len(tokenizer.word_index)+1, activation='softmax'))(x)

# Decoder model
decoder = keras.Model(latent_inputs, outputs, name='decoder')

##  VAE model
## the first training
vae = keras.Model(inputs, outputs, name='vae_mlp')

## load pre-train model
# vae = load_model('best_model_v0601.h5')

# Compile
vae.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Prepare targets
sequences_targets = np.expand_dims(sequences_padded, -1)

# Train
vae.fit(sequences_padded, sequences_targets, epochs=10, batch_size=32)

vae.save('best_model_v0601a.h5')  # creates a HDF5 file 'my_model.h5'

# Generate sequences from the latent space and decode
z_samples = np.random.normal(size=(1, latent_dim))

# Predict sequences from the latent space
pred_sequences = decoder.predict(z_samples)
# print(pred_sequences)

# Get the list of tokens, add a placeholder for the 0 index
tokens = [''] + list(tokenizer.word_index.keys())

# Sample from the softmax outputs instead of the argmax indices
decoded_sequences = []
for seq in pred_sequences[0]:
    decoded_sequences.append(np.random.choice(tokens, p=seq))

# Convert the sequences back to strings
decoded_sequences = ''.join(decoded_sequences)

print(decoded_sequences)
