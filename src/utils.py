import torch
import matplotlib.pyplot as plt


def calculate_accuracy(predictions, labels):
	"""Calculates the average accuracy 

		Args:
			predictions (torch.Tensor): The predicted label values.
			labels (torch.Tensor): The true label values.

		Returns:
			accuracy (float): The average accuracy for the given predicted and true labels.
	"""

	# Round the predictions
	predicted_labels = torch.round(torch.sigmoid(predictions))

	# Compute the average accuracy
	accuracy = (predicted_labels == labels).sum() / len(labels)

	return accuracy
	

def create_plots(train_set_loss, train_set_acc, val_set_loss, val_set_acc, test_set_loss, test_set_acc):
	"""Creates the plots for losses and accuracy for the train, validation, and test set.
		
		Args:
			train_set_loss (List[float]): The loss for the training set
			train_set_acc (List[float]): The accuracy for the training set
			val_set_loss (List[float]): The loss for the validation set
			val_set_acc (List[float]): The accuracy for the validation set
			test_set_loss (List[float]): The loss for the testing set
			test_set_loss (List[float]): The accuracy for the testing set
	"""

	epochs = range(len(train_set_loss))
	plt.plot(epochs, train_set_loss, 'g', label="Training loss")
	plt.plot(epochs, val_set_loss, 'b', label="Validation loss")
	plt.plot(epochs, test_set_loss, 'r', label="Test loss")
	plt.title("Train, Test and Validation Loss")
	plt.xlabel("Epochs")
	plt.ylabel("Loss")
	plt.legend()
	plt.show()

	epochs = range(len(train_set_loss))
	plt.plot(epochs, train_set_acc, 'g', label="Training Accuracy")
	plt.plot(epochs, val_set_acc, 'b', label="Validation Accuracy")
	plt.plot(epochs, test_set_acc, 'r', label="Test Accuracy")
	plt.title("Train, Test and Validation Accuracy")
	plt.xlabel("Epochs")
	plt.ylabel("Accuracy")
	plt.legend()
	plt.show()
