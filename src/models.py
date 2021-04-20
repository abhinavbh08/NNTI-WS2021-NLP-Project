import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence


def get_embeddings_dictionary(embeddings_path):
	"""Creates an embeddings dictionary which contains the embeddings for each word trained from Task-1.

	Args:
		embeddings_path (str): Path where the embeddings file is stored.

	Returns:
		embeddings_dict (Dict): Dictionary containing the word as key and its corresponding embedding as value.
	"""

	embeddings_dict = {}

	# Open the file containing the embeddings.
	file = open(embeddings_path, encoding='utf-8')

	# For each of the line in the file, the first element is the word and the rest are the embeddings.
	for line in file:
		values = line.split()
		word = values[0]
		embs = np.array(values[1:]).astype(np.float)

		# Store the embedding corresponding to the word in the embeddings_dict dictionary.
		embeddings_dict[word] = embs

	file.close()
	return embeddings_dict


def create_embeddings_matrix(embeddings_path, word2idx, embedding_size=300):
	""" Converts the embeddings dictionary in to matrix of embeddings where each row corresponds to embeddings of a particular word in our vocabulary.
		
		Args:
			embeddings_path (str): Path where the embeddings file is stored.
			word2idx (Dict): Dictionary containing mapping from words to indices for the elements in our vocabulary.
			embedding_size (int): The size of the embedding vector for each of the word.

		Returns:
			embeddings_matrix (torch.Tensor): Matrix containing the embeddings for each of the word in the vocabulary.
	"""

	embeddings_dict = get_embeddings_dictionary(embeddings_path)

	# Initialise the embeddings matrix size being equal to number of words in vocab and the size of embedding vector.
	embeddings_matrix = torch.zeros([len(word2idx), embedding_size], dtype=torch.float32)
	
	# For each of the word in the vocabulary, get the word and its index
	for i, (word, values) in enumerate(word2idx.items()):

		# If the word is in embeddings_dict, the add the corresponding embedding vextor to the matrix, else add a random vector to the matrix.
		if word in embeddings_dict:
			embeddings_matrix[values] = torch.FloatTensor(embeddings_dict[word])
		else:
			embeddings_matrix[values] = torch.randn(embedding_size, )

	return embeddings_matrix


def create_embedding_layer(embeddings_matrix, trainable=False):
	""" Creates the embedding layer for the neural network.

	Args:
		embeddings_matrix (torch.Tensor): Matrix containing the embeddings for each of the word in the vocabulary.
		trainable (Boolean): Whether we want to train our embeddings layer or not. Default: False

	Returns:
		embedding_layer (torch.nn.Embedding): Pytorch Embedding layer with weights initialised to the embeddings matrix.
	"""

	num_embeddings, embeddings_dim = embeddings_matrix.size()

	# Create the embedding layer.
	emb_layer = nn.Embedding(num_embeddings, embeddings_dim)

	# Initialise the weights of the embedding layer with the embeddings_matrix
	emb_layer.load_state_dict({'weight': embeddings_matrix})

	# Set whether to train or not to train the embedding layer.
	if trainable==False:
		emb_layer.weight.requires_grad = False


	return emb_layer


class CNN(nn.Module):
	"""Defining the CNN architecture to be used in Task 2 of the project"""
	def __init__(self, vocab_size, embedding_dim, out_dim, embeddings_matrix, trainable, filter_sizes=[2,3,4,5], num_filters=18):
		"""Constructor for the CNN Module

			Args: 
				vocab_size (int): The size of the vocabulary of our train set.
				embedding_dim (int): The size of the embeddings vector for a single word.
				out_dim (int): The output dimension of the network
				embeddings_matrix (torch.Tensor): Matrix containing the embeddings for each of the word in the vocabulary.
				trainable (Boolean): Whether the train the embedding layer along with the model or not.
				filter_sizes (List[int]): The sizes of the kernels for the Convolutional layers
				num_filters (int): The number of filters for each of the filter size in the convolution layer 
		"""


		super().__init__()
		self.vocab_size = vocab_size
		self.embedding_dim = embedding_dim
		self.out_dim = out_dim

		# Create the embeddings layer
		self.emb = create_embedding_layer(embeddings_matrix, trainable=trainable)

		# List of the 1-D convolution layers according to the filter sizes which have taken.
		self.convs_list = nn.ModuleList([nn.Conv1d(self.embedding_dim, num_filters, filter_sizes[i]) for i in range(len(filter_sizes))])

		# Dropout layer
		self.dropout = nn.Dropout(0.3)

		# Output feed forward layer.
		self.out = nn.Linear(len(filter_sizes)*num_filters, self.out_dim)

	def forward(self, inputs):
		"""Computes the forward pass of the CNN model.

		Args:
			inputs (torch.Tensor): The input text data in the form of sequences.

		Returns:
			logit (torch.Tensor): The logits values after applying the linear layer at the end.
		"""

		# Pass the inputs through the embedding layer.
		embeds = self.emb(inputs)

		# Permute the inputs to make them compatible with the conv1d layer.
		embeds = embeds.permute(0, 2, 1)

		# Loop over all the convolution layers and pass the dataset through those.
		x_convolved = [F.relu(conv(embeds)) for conv in self.convs_list]

		# Maxpool over all the outputs of the convolution layers
		x_pooled = [F.max_pool1d(op, op.size(2)) for op in x_convolved]

		# Concatenate all the max pooled outputs
		x_concatenated = torch.cat([layer.squeeze(2) for layer in x_pooled], 1)

		# Apply dropout to the concated data
		x = self.dropout(x_concatenated)

		# Apply linear layer to the droupouted data
		logit = self.out(x)

		return logit


class CNN_extend(nn.Module):
	"""Defining the CNN architecture which adds a linear layer on top of previous CNN module."""
	def __init__(self, model, filter_sizes = [2,3,4,5], num_filters=18):
		"""Contructor for the CNN_extend module.

			Args:
				model (torch.nn.Module): The pytorch model, in our case the 1D CNN model to which we will add a linear layer on top.
				filter_sizes (List[int]): The filter sizes, we just need their lengths in our function.
				num_filter (int): The number of filters corresponding to each filter size. 
		"""

		super().__init__()
		self.model = model

		# Replacing the output layer of the CNN with a linear layer
		self.model.out = nn.Linear(len(filter_sizes)*num_filters, 256)

		# The output layer
		self.out = nn.Linear(256, 1)

	def forward(self, inputs):
		"""Computes the forward pass of the CNN model.

		Args:
			inputs (torch.Tensor): The input text data in the form of sequences.

		Returns:
			logit (torch.Tensor): The logits values after applying the linear layer at the end.
		"""

		# Pass the data through the model to get the feature representation.
		x = self.model(inputs)

		# Apply the output layer
		logit = self.out(x)

		return logit


class BiLSTM(nn.Module):
	"""Defining the Bidirectional LSTM architecture to be used in Task 3 of the project"""
	def __init__(self, vocab_size, embedding_dim, hidden_dim, out_dim, num_layers=1, dropout_prob=0.5):
		"""Constructor for the BiLSTM Module

			Args: 
				vocab_size (int): The size of the vocabulary of our train set.
				embedding_dim (int): The size of the embeddings vector for a single word.
				hidden_dim (int): The dimension of the hidden state of the LSTM.
				out_dim (int): The output dimension of the network
				num_layers (int): Number of layers of the Bidirectional LSTM
				dropout_prob (Float): Probability value for dropout.

		"""

		super().__init__()
		self.vocab_size = vocab_size
		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim
		self.out_dim = out_dim

	    # The embedding layer.
		self.emb = nn.Embedding(vocab_size, embedding_dim)

	    # The Bilstm layer. We set bidirectional=True for a Bi-LSTM
		self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, batch_first=True, num_layers=num_layers, bidirectional=True)

		# The dropout layer
		self.dropout = nn.Dropout(dropout_prob)

	    #  The output layer.
		self.out = nn.Linear(self.hidden_dim*2, self.out_dim)
	    
	def forward(self, inputs, lengths):
		"""Computes the forward pass of the BiLSTM model.

		Args:
			inputs (torch.Tensor): The input text data in the form of sequences.
			lengths (torch.Tensor): The actual lengths of each of the sentence in the inputs.

		Returns:
			logit: The logits values after applying the linear layer at the end.
		"""
						
		# Pass the inputs through the embedding layer.
		embeds = self.emb(inputs)

		# Pack the embedded inputs for input to LSTMs.
		embeds = pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=False)

		# Pass the packed inputs to the lstms.
		outputs, (hidden, cell) = self.lstm(embeds)

		outputs, lengths = pad_packed_sequence(outputs, batch_first=True)

		# Apply dropout on the concatenation of the last forward and backward hidden states of the BiLSTM.
		hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

		# Pass the result to the output layer.
		out = self.out(hidden)
		
		return out


class Attention(nn.Module):
	"""Defining the attention layer to be used with Bi-LSTM"""
	def __init__(self, hidden_dim):
		"""Constructor for the Attention class.

			Args:
				hidden_dim (int): The double of the hidden vector size of the LSTM unit. This is because we are using Bi-LSTM so double hidden layer size.
		"""

		super(Attention, self).__init__()

		# Weight matrix to multiply with the hidden units.
		self.weights = nn.Linear(hidden_dim, hidden_dim)

		# Context vector to multiply with the hidden units after passing them through linear layer.
		self.context_u = nn.Linear(hidden_dim, 1, bias=False)

	def forward(self, lstm_outputs):
		"""Computes the forward pass for the attention layer.
			
			Args:
				lstm_outputs (torch.Tensor): The concatenated forward and backward hidden state output for each of the word in the sentence

			Returns:
				weighted_sum (torch.Tensor): The attention weighted sum for the hidden states.
		"""

		# Passing the lstm outputs through a linear layer.
		tanh_h = torch.tanh(self.weights(lstm_outputs))
		
		# Multiplying the context with the above vector to get the attention scores.
		context_multiplied = self.context_u(tanh_h)
		
		# Taking softmax to normalise the scores.
		scores = F.softmax(context_multiplied, dim=1)
		
		# Weighting the inputs and then taking the sum.
		weighted_sum = (scores * lstm_outputs).sum(1)

		return weighted_sum


class BiLSTM_attention(nn.Module):
	"""Defining the Bidirectional LSTM with attention architecture to be used in Task 3 of the project"""
	def __init__(self, vocab_size, embedding_dim, hidden_dim, out_dim, num_layers=1, dropout_prob=0.5):
		"""Constructor for the BiLSTM Module

			Args: 
				vocab_size (int): The size of the vocabulary of our train set.
				embedding_dim (int): The size of the embeddings vector for a single word.
				hidden_dim (int): The dimension of the hidden state of the LSTM.
				out_dim (int): The output dimension of the network
				num_layers (int): Number of layers of the Bidirectional LSTM
				dropout_prob (Float): Probability value for dropout.

		"""

		super().__init__()
		self.vocab_size = vocab_size
		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim
		self.out_dim = out_dim

	    # The embedding layer.
		self.emb = nn.Embedding(vocab_size, embedding_dim)

	    # The Bilstm layer
		self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, batch_first=True, num_layers=num_layers, bidirectional=True)

		# The attention layer.
		self.attention = Attention(self.hidden_dim*2)

		# The dropout layer
		self.dropout = nn.Dropout(dropout_prob)

	    #  The output layer.
		self.out = nn.Linear(self.hidden_dim*2, self.out_dim)
	    
	def forward(self, inputs, lengths):
		"""Computes the forward pass of the BiLSTM model.

		Args:
			inputs (torch.Tensor): The input text data in the form of sequences.
			lengths (torch.Tensor): The actual lengths of each of the sentence in the inputs.

		Returns:
			logit: The logits values after applying the linear layer at the end.
		"""
						
		# Pass the inputs through the embedding layer.
		embeds = self.emb(inputs)

		# Pack the embedded inputs for input to LSTMs.
		embeds = pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=False)

		# Pass the packed inputs to the lstms.
		outputs, (hidden, cell) = self.lstm(embeds)
		outputs, lengths = pad_packed_sequence(outputs, batch_first=True)

		# Pass the otuput from the bi-lstm to the attention layer to get the attention weighted summed output.
		weighted_outputs = self.attention(outputs)

		# Pass the result to the output layer.
		out = self.out(weighted_outputs)
		
		return out