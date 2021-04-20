# Imports
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import re # for using regular expressions 
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter
from torch.utils.data import DataLoader


def create_vocab(stopwords_removed_sentences):
  """Create the vocab and correpsonding dictionaries for indices to word and word to indices conversion.

  Args:
    stopwords_removed_sentences (List[str]): The list of clean sentences.

  Returns:
    V (List[str]): The list containing all the unique words in our data
    word2idx (Dict): The dictionary containing mapping from word to indices.
    idx2word (Dict): The dictionary containing mapping from indices to words.
  """
  
  V = []
  for sent in stopwords_removed_sentences:
    tokens = sent.split() # Get each of the token in the vocab
    for token in tokens:
      if token not in V: # Add the token to the vocab list if it is not there already.
        V.append(token)

  # Creating dictionaries for converting word to unique indices and converting words back from the unique indices. 
  word2idx = {word:idx for idx, word in enumerate(V)}
  idx2word = {idx:word for idx, word in enumerate(V)}
  
  return V, word2idx, idx2word


def word_to_one_hot(word, word2idx):
  """Return the one hot encoding of the word given to this function. 
  
    Args:
      word (str): The word for which one hot representation is required.
      word2idx (Dict): The dictionary mapping from word to indices.

    Returns:
      x (torch.Tensor): The one hot representation for the word.
  """

  # Create a vector or zeros equal to the length of the vocab
  x = torch.zeros(len(word2idx)).float() 
  
  # Setting the value corresponding to the index of word 1
  x[word2idx[word]] = 1.0 
  
  return x


def sampling_prob(word, probs):

  """ This function gives the sampling probability of the word using the probabilities which we will pre calculate in the next function. """

  return probs[word]

def pre_calculate_probabilities(stopwords_removed_sentences):
  """Calculate the probability to keep it in the context for all the words.

    Args:
      stopwords_removed_sentences (List[str]): The cleaned sentences.

    Returns:
      probs (Dict): The dictionary containing words and probability to keep it in a context.
  """

  # all tokens contain all the tokens in the corpus, even if they are repeated. We will use it next for calculating the word frequencies.
  all_tokens = [] 

  # Add all of the tokens in the vocab.
  for sent in stopwords_removed_sentences:
    tokens = sent.split()
    for token in tokens:
      all_tokens.append(token)

  # Get the counts of all the tokens in the vocab.
  counts = Counter(all_tokens) 
  print(len(counts), counts)

  # Find the relative word frequencies.
  freqs = {}
  for word, count in counts.items():
    freqs[word] = count / len(all_tokens)
  print("Frequencies: ", freqs)

  # Find the probability to keep a word in the context using the above given formula. # As told in the course discussion forum, 0.000001 should be taken instead of 0.001.
  probs = {}
  for word, freq in freqs.items():
    probs[word] = (np.sqrt(freq/0.000001) + 1)*(0.000001/freq)
  print("Probabilities", probs)

  return probs


def get_target_context(sentence, window_size, word2idx, idx2word, probs):
  """Yield a pair of (current_word, context) for us and also samples on the basis of the function defined in the previous code cell.
  
    Args:
      sentence (str): The sentence for which we want to generate the (current_word, context) pair.
      window_size (int): The size of the context windows before and after a word.
      word2idx (Dict): The dictionary mapping from words to indices.
      idx2word (Dict): The dictionary mapping from indices to words.
      probs (Dict): Dictionary containing a word and the probability to keep it in a context.

    Returns:
      (current_word, context) (Tuple): The current word and the context for each pair in the sentence.
  """
  
  # Making a list of words in the sentence.
  words = sentence.split()

  # Take every word in the current sentence as the centre word.
  for center_word in range(0, len(words)): 

    # Go through each word in the window_size around the current word.
    for current_window in range(-window_size, window_size+1): 
      context_word = center_word + current_window

      # If word is out of bounds, or equal to center word, do not yield it.
      if center_word==context_word or context_word<0 or context_word>=len(words):
        continue

      # Keeping this range because maximum probability to keep in context for our dataset is 0.35.
      p = np.random.uniform(0, 0.35)

      # Do not remove words which occur one time.
      if p<0.3:
        # Sampling the word based on its sampling probability.
        if sampling_prob(words[context_word], probs) < p:
          continue

      # Yield the (current_word, context) pair.
      yield (words[center_word], words[context_word])



class Word2Vec(nn.Module):
  def __init__(self, vocabulary_size, embedding_size):

    super(Word2Vec, self).__init__()

    # Weight matrix for input to hidden layer
    self.w1 = nn.Parameter(torch.randn(vocabulary_size, embedding_size, requires_grad=True)) 

    # Weight matrix for hidden to output layer.
    self.w2 = nn.Parameter(torch.randn(embedding_size, vocabulary_size, requires_grad=True)) 

  def forward(self, one_hot):
    # Doing the required matrix multiplications
    z1 = torch.matmul(one_hot, self.w1)
    z2 = torch.matmul(z1, self.w2)

    # Taking log softmax on the output layer.
    log_softmax = F.log_softmax(z2, dim=1)
    
    return log_softmax


def get_input_layer_onehot(data, word2idx):
  """Get matrix of one hot encoded input data by calling word_to_one_hot function defined earlier

      Args:
        data (List[str]): The list of words to one hot encode.

      Returns:
        mat (torch.Tensor): A matrix containing the one hot representations for all the words in data list.
  """

  data_points = len(data)

  # Initiaise the matrix as zeros.
  mat = torch.zeros(data_points, len(word2idx))

  # For each of the data point, add it to the matrix after getting its one hot representation by calling word_to_one_hot function defined earlier.
  for i, index in enumerate(data):
    mat[i] = word_to_one_hot(index, word2idx)

  return mat


def train(epochs, word_pairs, batch_size, model, device, criterion, optimizer, word2idx):
 
  print("Training started")

  for epoch in range(epochs):
    loss_val = 0

    # for each batch in the dataloaders
    for data, target in zip(DataLoader(word_pairs[:, 0], batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True), DataLoader(word_pairs[:, 1], batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True)):

      # Clear out the gradients from the previous batch
      optimizer.zero_grad()

      # Get the one hot representation for the input words in the batch.
      x = get_input_layer_onehot(data, word2idx).to(device)

      # Get the indices for the target context words in the batch
      y_true = torch.from_numpy(np.array([word2idx[t] for t in target])).long().to(device)

  		# Do the forward pass of the model
      y_pred = model(x)

      # calculate the loss value using our loss function on this batch
      loss = criterion(y_pred, y_true)
      loss_val = loss_val + loss.item()

  		# Do backpropagation of the gradients
      loss.backward()
  
      # Update the weights
      optimizer.step()

    loss_epoch = loss_val/(len(word_pairs)/batch_size)

    # Criterion for the training procesure to terminate if a certain loss value is reached.
    if loss_epoch<4.75:
      break

    # Print the loss after every epoch.
    print(f'Loss at epo {epoch}: {loss_epoch}')


def create_bengali_embeddings(df, device):

	stopwords_removed_sentences = list(df["text"].values)
	V, word2idx, idx2word = create_vocab(stopwords_removed_sentences)
	# print(len(V), len(word2idx), len(idx2word))
	probs = pre_calculate_probabilities(stopwords_removed_sentences)
	window_size = 5
	# List to store all the (current_word, context) pairs.
	word_pairs = [] 

	for sentence in stopwords_removed_sentences:  
	  # For each of the sentence, get the traget context from the above generator
	  for pair in get_target_context(sentence, window_size, word2idx, idx2word, probs):
	    word_pairs.append(pair)

	word_pairs = np.array(word_pairs)
	for j in range(0, 5):
	  print(word_pairs[j][0], word_pairs[j][1])

	# Word2vec paper recommends window size of 10, but we keep here to 5 as keeping size 10 leads to a lot of noise due to the small size of the dataset.
	window_size = 5
	embedding_size = 300

	# More hyperparameters
	learning_rate = 0.03
	epochs = 300

	vocabulary_size = len(V)

	# Initializing the model
	model = Word2Vec(vocabulary_size, embedding_size)

	# SGD as used by the original paper
	optimizer = optim.SGD(model.parameters(), lr=learning_rate)

	# Using Negative log likelihood loss on the output from the forward pass.
	criterion = nn.NLLLoss()


	model = model.to(device)

	batch_size = 16

	train(epochs, word_pairs, batch_size, model, device, criterion, optimizer, word2idx)


