from lda2vec import utils, b_model
import numpy as np
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Path to preprocessed data
data_path  = "data/clean_data"
# Whether or not to load saved embeddings file
load_embeds = True

# Load data from files
(idx_to_word, word_to_idx, freqs, pivot_ids, target_ids, doc_ids, embed_matrix, bias_idxes) = utils.load_preprocessed_data(
      data_path, load_embed_matrix=load_embeds, load_bias_idxes=True)

bias_words = ['privacy', 'anonymity','confidentiality','disclosure']
base_bias_idxes = [word_to_idx[word] for word in bias_words]
bias_idxes = [[base_bias_idxes[0], base_bias_idxes[1]],
              [base_bias_idxes[0], base_bias_idxes[2]],
              [base_bias_idxes[0], base_bias_idxes[3]],
              [base_bias_idxes[0]]
              [base_bias_idxes[2]]


# Number of unique documents
num_docs = len(np.unique(doc_ids))
# Number of unique words in vocabulary (int)
vocab_size = embed_matrix.shape[0] 
# Embed layer dimension size
# If not loading embeds, change 128 to whatever size you want.
embed_size = embed_matrix.shape[1] if load_embeds else 128
# Number of topics to cluster into
num_topics = 20
# Number of topics to bias
num_bias_topics = 5
# How strongly we bias the topics
bias_lambda = 1e-2
# Factor that determines how much bias topics have to be close to all bias terms
# 0 is uniform focus, 100+ is hard specialization
bias_unity = 20.0

target_bias_topic_cov=0.8
# Epoch that we want to "switch on" LDA loss
switch_loss_epoch = 5
# Pretrained embeddings
pretrained_embeddings = embed_matrix if load_embeds else None
# If True, save logdir, otherwise don't
save_graph = True
num_epochs = 200
batch_size = 512 #4096
lmbda = 1e-4
logdir = "bias_experiment"

# Initialize the model
m = b_model(num_docs,
          vocab_size,
          num_topics,
          bias_idxes,
          bias_topics=num_bias_topics,
          bias_lmbda=bias_lambda,
          bias_unity=bias_unity,
          target_bias_topic_cov=0.8,
          embedding_size=embed_size,
          pretrained_embeddings=pretrained_embeddings,
          freqs=freqs,
          batch_size = batch_size,
          save_graph_def=save_graph,
          logdir=logdir)

# Train the model
m.train(pivot_ids,
        target_ids,
        doc_ids,
        len(pivot_ids),
        num_epochs,
        idx_to_word=idx_to_word,
        switch_loss_epoch=switch_loss_epoch)

# Visualize topics with pyldavis
utils.generate_ldavis_data(data_path, m, idx_to_word, freqs, vocab_size)