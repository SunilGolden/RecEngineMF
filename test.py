import torch
from torch.utils.data import DataLoader
import argparse
from dataset import RatingsDataset
from utils import get_device, reset_random, test


def main():
	parser = argparse.ArgumentParser(
		description='Test Matrix Factorization model')
	
	parser.add_argument('--data_path', type=str, default='./ratings.csv')
	parser.add_argument('--batch_size', type=int, default=64000)
	parser.add_argument('--random_seed', type=int, default=42)
	parser.add_argument('--model_path', type=str, default='./mf_model.pth')
	
	args = parser.parse_args()

	reset_random(args.random_seed)

	train_dataset = RatingsDataset(args.data_path, split='train')

	test_dataset = RatingsDataset(args.data_path, 
	                              split='test', 
	                              user_set=train_dataset.user_set, 
	                              item_set=train_dataset.item_set)

	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
	test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

	test(args.model_path, test_loader)


if __name__ == '__main__':
	main()