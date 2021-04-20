import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from src.utils import calculate_accuracy

def train(model, iterator, optimizer, criterion, device, scheduler):
	"""Runs the training loop for the model.

		Args:
			model (nn.Module): Pytorch model object of our defined class.
			iterator (torch.utils.Data.DataLoader): The iterator for the data.
			optimizer (torch.optim): The optimization algorithm used to train the model.
			criterion (torch.nn): The loss function.
			device (str): The device which is available, it can be either cuda or cpu.
			scheduler (torch.optim.lr_scheduler): Learning rate scheduler.

		Returns:
			epoch_loss (float): The average loss for one epoch on the given dataloader.
			epoch_acc (float): The average accuracy one the epoch on the given dataloader.
	"""
	
	epoch_loss = 0
	epoch_acc = 0

	# Put the model in the training mode.
	model.train()

	# for each batch in the dataloader
	for batch in iterator: 

		# Clear out the gradients from the previous batch
		optimizer.zero_grad() 

		# move the inputs and the labels to the device.
		inputs = batch[0].to(device) 
		labels = batch[1].to(device)

		# If we are training LSTM based model, then we also need to pass original lengths of the sequences to the forward function
		if model.__class__.__name__ == "BiLSTM" or model.__class__.__name__ == "BiLSTM_attention":
			lengths = torch.as_tensor(batch[2], dtype=torch.int64)
			predictions = model(inputs, lengths).squeeze(1)
		else:
			# No need for lengths in the CNN based model.
			predictions = model(inputs).squeeze(1)

		# calculate the loss value using our loss function on this batch
		loss = criterion(predictions, labels)

		# calculate the accuracy for this batch
		accuracy = calculate_accuracy(predictions, labels)

		# Do backpropagation of the gradients
		loss.backward()

		# update the weights
		optimizer.step()

		# add the loss and the accuracy for the epoch.
		epoch_loss += loss.item()
		epoch_acc += accuracy.item()

	return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, device):
	"""Runs the evaluation loop for the model.

		Args:
			model (nn.Module): Pytorch model object of our defined class.
			iterator (torch.utils.Data.DataLoader): The iterator for the data.
			criterion (torch.nn): The loss function.
			device (str): The device which is available, it can be either cuda or cpu.

		Returns:
			epoch_loss (float): The average loss for the epoch on the given dataloader.
			epoch_acc (float): The average accuracy for the epoch on the given dataloader.
	"""
	epoch_loss = 0
	epoch_acc = 0

	# Put the model in the evaluation mode.
	model.eval()

	# Do not calculate the gradients in the evaluaion mode. 
	with torch.no_grad():

		# for each batch in the dataloader
		for batch in iterator:

			# move the inputs and the labels to the device.
			inputs = batch[0].to(device)
			labels = batch[1].to(device)

			# If we are training LSTM based model, then we also need to pass original lengths of the sequences to the forward function
			if model.__class__.__name__ == "BiLSTM" or model.__class__.__name__ == "BiLSTM_attention":
				lengths = torch.as_tensor(batch[2], dtype=torch.int64)
				predictions = model(inputs, lengths).squeeze(1)
			else:
				# No need for lengths in the CNN based model.
				predictions = model(inputs).squeeze(1)

	    	# calculate the loss value using out loss function on this batch.
			loss = criterion(predictions, labels)

			# calculate the accuracy for this batch
			accuracy = calculate_accuracy(predictions, labels)

			# add the loss and the accuracy for the epoch.
			epoch_loss += loss.item()
			epoch_acc += accuracy.item()

	return epoch_loss / len(iterator), epoch_acc / len(iterator)


def train_and_evaluate(num_epochs, model, train_loader, val_loader, test_loader, optimizer, criterion, device, scheduler):
	"""Call the train and evaluate function for each of the epoch, print the loss and accuracies.

		Args:
			num_epochs (int): The number of epochs for which to train the model.
			model (nn.Module): Pytorch model object of our defined class.
			train_loader (torch.utils.Data.DataLoader): The iterator for the training data.
			val_loader (torch.utils.Data.DataLoader): The iterator for the validation data.
			test_loader (torch.utils.Data.DataLoader):  The iterator for the test data.
			optimizer (torch.optim): The optimization algorithm used to train the model.
			criterion (torch.nn): The loss function.
			device (str): The device which is available, it can be either cuda or cpu.
			scheduler (torch.optim.lr_scheduler): The learning rate scheduler

		Returns:
			train_set_loss (List[float]): The loss for the training set
			train_set_acc (List[float]): The accuracy for the training set
			val_set_loss (List[float]): The loss for the validation set
			val_set_acc (List[float]): The accuracy for the validation set
			test_set_loss (List[float]): The loss for the testing set
			test_set_loss (List[float]): The accuracy for the testing set
	"""

	train_set_loss = []
	train_set_acc = []
	val_set_loss = []
	val_set_acc = []
	test_set_loss = []
	test_set_acc = []

	for epoch in range(num_epochs):
    
    	# Call the training function with the training data loader and save loss and accuracy
		train_loss, train_acc = train(model, train_loader, optimizer, criterion, device, scheduler)
		train_set_loss.append(train_loss)
		train_set_acc.append(train_acc)

		scheduler.step()

	    # Call the evaluation function with the vaidation data laoder and save loss and accuracy
		valid_loss, valid_acc = evaluate(model, val_loader, criterion, device)
		val_set_loss.append(valid_loss)
		val_set_acc.append(valid_acc)


    	# Call the evaluation function with the test data loader and save loss and accuracy.
		test_loss, test_acc = evaluate(model, test_loader, criterion, device)
		test_set_loss.append(test_loss)
		test_set_acc.append(test_acc)

		print(f"======================EPOCH {epoch}=========================") 
		print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
		print(f'Val. Loss: {valid_loss:.3f}  |  Val. Acc: {valid_acc*100:.2f}%')
		print(f'Test. Loss: {test_loss:.3f} |  Val. Acc: {test_acc*100:.2f}%')

	return train_set_loss, train_set_acc, val_set_loss, val_set_acc, test_set_loss, test_set_acc