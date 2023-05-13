import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import wandb
import json

from dataset import RatingsDataset
from models import MF
from utils import reset_random, train_epochs


def main():
	parser = argparse.ArgumentParser(
		description='Train Matrix Factorization powered Recommendation Engine')
	
	parser.add_argument('--data_path', type=str, default='./ratings.csv')
	parser.add_argument('--emb_size', type=int, default=100)
	parser.add_argument('--random_seed', type=int, default=42)
	parser.add_argument('--batch_size', type=int, default=64000)
	parser.add_argument('--epochs', type=int, default=100)
	parser.add_argument('--learning_rate', type=float, default=0.001)
	parser.add_argument('--weight_decay', type=float, default=1e-5)
	parser.add_argument('--step_size', type=int, default=10)
	parser.add_argument('--gamma', type=float, default=0.1)
	parser.add_argument('--patience', type=int, default=3)
	parser.add_argument('--model_name', type=str, default='mf_model.pth')
	parser.add_argument('--metrics_csv_name', type=str, default='metrics.csv')
	parser.add_argument('--silent', action='store_true')
	parser.add_argument('--log_wandb', action='store_true')
		
	args = parser.parse_args()

	reset_random(args.random_seed)

	if args.log_wandb:
		with open('secrets.json', 'r') as f:
			secrets = json.load(f)

		WANDB_API_KEY = secrets['WANDB_API_KEY']

		wandb.login(key=WANDB_API_KEY)

		wandb.init(
			project="RecEngineMF",

			# Track hyperparameters and run metadata
			config = {
				"random_seed": args.random_seed,
				"batch_size": args.batch_size,
				"epochs": args.epochs,
				"learning_rate": args.learning_rate,
				"weight_decay": args.weight_decay,
				"step_size": args.step_size,
				"gamma": args.gamma,
				"patience": args.patience,
				"model_name": args.model_name
			}
		)

	train_dataset = RatingsDataset(args.data_path, split='train')

	val_dataset = RatingsDataset(args.data_path, 
	                             split='val', 
	                             user_set=train_dataset.user_set, 
	                             item_set=train_dataset.item_set)

	test_dataset = RatingsDataset(args.data_path, 
	                              split='test', 
	                              user_set=train_dataset.user_set, 
	                              item_set=train_dataset.item_set)

	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
	test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

	model = MF(train_dataset.num_users, train_dataset.num_items, emb_size=args.emb_size)

	if torch.cuda.device_count() > 1:
		print("Available GPUs", torch.cuda.device_count())
		model = nn.DataParallel(model)

	train_epochs(model, 
	             train_loader, 
	             val_loader, 
	             epochs=args.epochs, 
	             lr=args.learning_rate, 
	             weight_decay=args.weight_decay,
	             step_size=args.step_size,
	             gamma=args.gamma,
	             patience=args.patience,
	             model_name=args.model_name,
	             metrics_csv_name=args.metrics_csv_name,
	             silent=args.silent,
				 log_wandb=args.log_wandb)


if __name__ == '__main__':
	main()