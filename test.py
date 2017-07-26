import time
import random
import numpy as np
import tensorflow as tf
from seq2seq import TrainableSequence2Sequence

# Number of Epochs
episodes = 10000

# Batch Size
batch_size = 128

# RNN Size
rnn_size = 50

# Number of Layers
num_layers = 1

# Learning Rate
learning_rate = 0.001

# Source sequence length
source_sequence_length = 5

# Target sequence length
target_sequence_length = 5

# n_feature
n_features = 1



def sample(source_sequence_length, target_sequence_length):
    xi = random.random()
    vi = random.random()
    xf = xi+vi*1.2
    source = xi + vi*np.linspace(0, 1, source_sequence_length)
    target = xf + vi*np.linspace(0, 1, target_sequence_length)
    return source, target



def test_seq2seq():

    model = TrainableSequence2Sequence(
        rnn_size=rnn_size,
        num_layers=num_layers,
        num_features=n_features
    )

    init = tf.global_variables_initializer()

    # Train
    with tf.Session() as sess:
        sess.run(init)

        for episode in range(episodes):
            # Generate new samples
            samples = [sample(target_sequence_length, source_sequence_length) for i in range(batch_size)]
            sources_batch = np.stack(s[0] for s in samples)[:,:,None]
            targets_batch = np.stack(s[1] for s in samples)[:,:,None]

            # All our sources and targets are full length
            targets_lengths = batch_size*[target_sequence_length]
            sources_lengths = batch_size*[source_sequence_length]

            loss = model.train(
                sess,
                sources_batch,
                targets_batch,
                learning_rate,
                targets_lengths,
                sources_lengths
            )

            if episode%20==0:
                # Evaluate on this dataset
                tloss,eloss = model.evaluate(
                    sess,
                    sources_batch,
                    targets_batch,
                    learning_rate,
                    targets_lengths,
                    sources_lengths
                )
                print('Episode {0}, Train Loss {1:.4f}, Eval Loss {2:.4f}'.format(episode,tloss,eloss))

            if episode%500==0:
                outputs_batch = model.predict(
                    sess,
                    sources_batch,
                    targets_lengths,
                    sources_lengths
                )
                print()
                print("Sources: ", sources_batch[0].flatten())
                print("Targets: ", targets_batch[0].flatten())
                print("Outputs: ", outputs_batch[0].flatten())
                print()


if __name__=="__main__":
    test_seq2seq()
