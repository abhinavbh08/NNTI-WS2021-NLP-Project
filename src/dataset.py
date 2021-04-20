import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import re
import torch
from torch.utils.data import DataLoader, Dataset


def read_dataset_hindi(file_path):
	"""Read the dataset from the file path, converts it into a dataframe and converts the labels of the required fields to numeric.
		
	Args:
		file_path (str): Path to the text file which we will read.

	Returns:
		df (pd.DataFrame): The dataframe representation of the file in the path given. 
	"""

	# storing the file as a list of strings after splitting it.
	with open(file_path, "r", encoding="utf-8") as file:
		data = file.read().split("\n")
  
	data = [line.split("\t") for line in data]

	# Converting the file to a pandas dataframe.
	header = data.pop(0)
	df = pd.DataFrame(data, columns=header)

	# Converting hate and offensive content label to 0 and the other label to 1.
	sentiment_map = {'HOF': 1, 'NOT': 0}
	df['task_1'] = df['task_1'].apply(lambda x : sentiment_map[x])
	return df


def read_dataset_bengali(file_path, HOF_hindi=2400, NOT_hindi=2000):
	"""Read the dataset from the Bengali file path, and makes the label distribution in it similar to the Hindi dataset.
		
	Args:
		file_path (str): Path to the text file which we will read.
		HOF_hindi (int): The approximate count of the hate and offensive content in the Hindi dataset.
		NOT_hindi (int): The approximate count of not hate content in the Hindi dataset.

	Returns:
		df (pd.DataFrame): The dataframe representation of the file in the path given. 
	"""	

	# Read the file in a pandas dataframe.
	df = pd.read_csv(file_path, encoding="utf-8")

	# Make the data size same as that of the Hindi Dataset for the hate type.
	df_hate = df[df["hate"] == 1].sample(n=HOF_hindi, random_state=200).reset_index(drop=True)

	# Make the data size same as that of the Hindi Dataset for the non-hate type
	df_non_hate = df[df["hate"] == 0].sample(n=NOT_hindi, random_state=200).reset_index(drop=True)

	# Concatenate the dataframes together for the portions corresponding to the hate and non hate splits.
	df = df_hate.append(df_non_hate, ignore_index=True)
	df.reset_index(drop=True, inplace=True)

	# Make the columns names same as that of the hindi dataset to make the other methods consistent.
	df.columns = ["text", "task_1", "category"]
	return df


def clean_text(txt, regex_lst, RE_EMOJI, stopwords_hindi):
	"""This function takes the regular expression for various things such as punctuations and all which we want to remove and returns the cleaned sentence.
			
	Args:
		txt (str): Sentence which we have to clean.
		regex_lst (List[re]): List of all the regular expressions according to whom we have to clean the data.
		RE_EMOJI (re): The regular expression for the emojis removal.
		stopwords_hindi (List[str]): List of stopwords in Hindi Language
			
	Returns:
		str_cleaned (str): The cleaned sentence.
	"""
	  
	str_cleaned = txt

	# Iterate over the list of regular expressions. 
	for regex in regex_lst:
		str_cleaned = re.sub(regex, '', str_cleaned)
	str_cleaned = RE_EMOJI.sub(r'', str_cleaned)

	sent_splitted = str_cleaned.split()

	# Do not add the word to the list if it is in the stopwords.
	str_cleaned = " ".join([x.lower() for x in sent_splitted if x not in stopwords_hindi])
	return str_cleaned


def clean_data_and_remove_stopwords(df, stopwords_file_path):
	"""Cleans the data and removes the stopwords from the data file.

	Args:
		df (pd.DataFrame): The dataframe containing our data.
		stopwords_file_path (str): The path of the file containing the stopwords.

	Returns:
		df (pd.DataFrame): The cleaned dataframe with the stopwords removed.
	"""

	# For removing the emojis
	RE_EMOJI = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)
	regex_lst = ["@[\w]+", # removing the words with @ symbols
             "#[\w]+" , # removing the hashtags with the full words which have hashtags
             r"http\S+", # removing the urls
             r"[\\.,\/#!$%\^&\*;:{}\।=\-_`~()\?]", # removing the punctuations.
             r"[0-9]",  # Also remove the numbers
             r"[a-zA-Z]"] # Also, remove the characters


	# Reading the file containing the stopwords.
	with open(stopwords_file_path, "r") as f:
		content = f.readlines()

	# Store the stopwords in a list
	stopwords_hindi = [x.strip() for x in content]

    # Getting the sentences from the dataframe which we have to clean.
	sentences = df['text'].values
	cleaned_sentences = []

	# For each of the sentence, do the regular expressions based data cleaning and add it to the cleaned sentences list.
	for sent in sentences:
		cleaned_sentences.append(clean_text(sent, regex_lst, RE_EMOJI, stopwords_hindi))

	df["text"] = cleaned_sentences

	# Removing the sentences where we have one or less than one word.
	indices_to_drop = []
	for i, item in enumerate(df["text"].values):
		if len(item.split()) <= 1:
			indices_to_drop.append(i)

	# Dropping the indices for the sentences which have one or less than one word.
	df.drop(df.index[indices_to_drop], inplace=True)
	df.reset_index(inplace=True, drop=True)

	return df


def get_splits(df, percentage=0.15):
	"""Get the train, validation and test split from the dataframe.  The ratio is 70:15:15

	Args:
		df (pd.DataFrame): The dataframe which we want to split into train, test and validation sets.
		percentage (float): The percentage of data we want in the testing and validation sets. Default: 0.15

	Returns:
		df_train (pd.DataFrame): The training portion of the dataframe.
		df_val (pd.DataFrame): The portion of the dataframe used for validation.
		df_test (pd.DataFrame): The portion of the dataframe used for testing our model.
	"""

	# Create the train and test dataframe from the original dataframe.
	df_train, df_test = train_test_split(df, test_size=percentage, random_state=400, stratify=df[["task_1"]])
	df_train.reset_index(drop=True, inplace=True)
	df_test.reset_index(drop=True, inplace=True)

	# Create the validation and subsetted train dataframe from the train dataframe.
	df_train, df_val = train_test_split(df_train, test_size=(percentage/(1-percentage)), random_state=500, stratify=df_train[["task_1"]])
	df_train.reset_index(drop=True, inplace=True)
	df_val.reset_index(drop=True, inplace=True)

	return df_train, df_val, df_test


def get_vocab(df_train):
	"""Get the vocabulary from the training data.

	Args:
		df_train (pd.DataFrame): The Dataframe containing the training dataset.

	Returns:
		word2idx (Dict): Mapping of the words to indices for the words in the dataset.
		idx2word (Dict): Mapping from indices to words for the words in the dataset.
	"""

	# Get all the words in the corpus even if they are repeated.
	all_words = []
	for sent in df_train["text"]:
		for token in sent.split():
			all_words.append(token)

	# Create a counter object for all the words
	word_counter = Counter(all_words)

	# Assign index 0 to passing and 1 to unknown words
	word2idx = {'_PAD': 0, '_UNK': 1}

	# COnvert from words to indices.
	word2idx.update({word: i+2 for i, (word, count) in enumerate(word_counter.items())})

	# COnvert from indices to words.
	idx2word = {idx: word for word, idx in word2idx.items()}
	  
	return word2idx, idx2word


def convert_to_index(word2idx, sent):
	"""Converts the words in a sentence to the corresponding indices.
	
	Args:
		word2idx (Dict): Mapping of the words to indices for the words in the dataset.
		sent (str): The sentence whose indiced representation we want.

	Returns:
		representation_indices (List[int]): List containing the indices for the words in the sentence according to our vocabulary.
	"""

	# Convert sentence word by word in the corrresponding indices. Keep word as UNK if it is not in the vocab.
	representation_indices = [word2idx.get(str(token), 1) for token in sent.split()]
	return representation_indices


class TextDataset(Dataset):
	"""Class for representing the text dataset"""
	def __init__(self, df, word2idx):
		""" Constructor for the TextDataset class.

		Args:
			df (pd.DataFrame): The dataframe for which we want the indices and the labels.
			word2idx (Dict): The dictionary for mapping the words to the corresponding indices.
		"""

		self.df = df
		
		# COnvert each of the sentence to its corresponding indices and add to the list.
		indices_sentences = []
		for sent in self.df["text"].values:
			indices_sentences.append(convert_to_index(word2idx, sent))
		self.df["text_indices"] = indices_sentences

	def __len__(self):
		"""Magic method for the len() function"""
		
		return self.df.shape[0]

	def __getitem__(self, idx):
		""" Magic method for getting items at specific index from the dataset.

		Args:
			idx (int): The index for the data which we want to get.

		Returns:
			x (List[int]): The list of word indices for a particular sentence.
			y (int): The output index corresponding to a particular sentence.
		"""
		
		x = self.df.text_indices[idx]
		y = self.df.task_1[idx]
		
		return x, y


def pad_collate(batch):
	"""This function does zero padding of the batch according to the maximum length of the sequence in the batch. 

	Args:
		batch (List[tuple]): Contains the sentences indices along with the label.

	Returns:
		padded_sents (torch.Tensor): The zero padded sentences.
		ys (torch.Tensor): The y labels for the dataset.
		x_lens (List[int]): The original lengths of the sentences before padding.
	"""

	(xs, ys) = zip(*batch)

	# Find the lengths of each of the sentences.
	x_lens = [len(x) for x in xs]

	# Create a matrix of all zeros
	padded_sents = torch.zeros(len(batch), max(x_lens)).long()

	# Add the sentences indices in the matrix. This will leave the leftover elements in the sentences as zero leading to zero padding.
	for i, (x, y) in enumerate(batch):
		padded_sents[i, :x_lens[i]] = torch.LongTensor(x)

	return padded_sents, torch.tensor(ys).float(), x_lens


def get_dataloaders(df_train, df_val, df_test, word2idx, batch_size=16):
	"""Returns the data loaders for the train, validation and the test data

	Args:
		df_train (pd.DataFrame): The dataframe containing the training data.
		df_val (pd.DataFrame): The dataframe containing the validation data.
		df_test (pd.DataFrame): The dataframe containing the test data.
		word2idx (Dict): The words to indices mapping dictionary.
		batch_size (int): The batch size with which we want to feed elements to our model. Default: 16

	Returns:
		train_loader (torch.util.data.DataLoader): Dataloader for the training data.
		val_loader (torch.util.data.DataLoader): Dataloader for the validation data.
		test_loader (torch.util.data.DataLoader): Dataloader for the test data.
	"""

	# Creating an object of TextDataset class for the training dataset.
	train_dataset = TextDataset(df_train, word2idx)

	# Creating an object of TextDataset class for the training dataset.
	val_dataset = TextDataset(df_val, word2idx)

	# Creating an object of TextDataset class for the training dataset.
	test_dataset = TextDataset(df_test, word2idx)

	# Creating Dataloaders for the train, test and validation data.
	train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate)
	val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, collate_fn=pad_collate)
	test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, collate_fn=pad_collate)

	return train_loader, val_loader, test_loader
