from lda2vec import utils, s_model
import numpy as np
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Path to preprocessed data
data_path  = "data/clean_data"
# Whether or not to load saved embeddings file
load_embeds = True

# Load data from files
(idx_to_word, word_to_idx, freqs, pivot_ids, target_ids, doc_ids, embed_matrix) = utils.load_preprocessed_data(
      data_path, load_embed_matrix=load_embeds)

seed_words = ['privacy', 'anonymity','confidentiality','disclosure']
seed_idxes = [word_to_idx[word] for word in seed_words if word in word_to_idx]

base_seed_idxes = [word_to_idx[word] for word in seed_words]
seed_idxes = [[base_seed_idxes[0], base_seed_idxes[1]],
              [base_seed_idxes[0], base_seed_idxes[2]],
              [base_seed_idxes[0], base_seed_idxes[3]],
              [base_seed_idxes[0]]
              [base_seed_idxes[2]]

# Number of unique documents
num_docs = len(np.unique(doc_ids))
# Number of unique words in vocabulary (int)
vocab_size = embed_matrix.shape[0] 
# Embed layer dimension size
# If not loading embeds, change 128 to whatever size you want.
embed_size = embed_matrix.shape[1] if load_embeds else 128
# Number of topics to cluster into
num_topics = 20
# How strongly we seed the topics
seed_lambda = 1e-2
# Epoch that we want to "switch on" LDA loss
switch_loss_epoch = 5
# Pretrained embeddings
pretrained_embeddings = embed_matrix if load_embeds else None
# If True, save logdir, otherwise don't
save_graph = True
num_epochs = 200
batch_size = 512 #4096
lmbda = -1e-4
logdir = "seed_experiment"

# Initialize the model
m = s_model(num_docs,
          vocab_size,
          num_topics,
          seed_idxes,
          lmbda=lmbda,
          seed_lmbda=seed_lambda,
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