import torch
import spacy
import pandas as pd
from torchtext.legacy import data
from torchtext.legacy import datasets


# define path
path = "data/"
train_file = "train.csv"
val_file = "valid.csv"
test_file = "test.csv"
ext = "csv"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# prepare custom dataset for training LSTM
def generate_subsets(path,ext,train_file,val_file,test_file,fields,BATCH_SIZE,device):
	train_data, valid_data, test_data = data.TabularDataset.splits(
                                        path = path,
                                        train = train_file,
                                        validation = val_file,
                                        test = test_file,
                                        format = ext,
                                        fields = fields,
                                        skip_header = True)

	train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    													(train_data, valid_data, test_data),
    													sort = False, #don't sort test/validation data
    													batch_size=BATCH_SIZE,
    													device=device)

	return train_iterator,valid_iterator,test_iterator




