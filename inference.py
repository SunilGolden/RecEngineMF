import torch
import argparse
from torch.utils.data import DataLoader
from dataset import RatingsDataset
from utils import reset_random, get_top_k_recommendations


def main():
	parser = argparse.ArgumentParser(
		description='Make Inference')
	
	parser.add_argument('--random_seed', type=int, default=42)
	parser.add_argument('--model_path', type=str, default='./mf_model.pth')
	parser.add_argument('--data_path', type=str, default='./ratings.csv')
	parser.add_argument('--batch_size', type=int, default=64000)
	parser.add_argument('--user_id', type=int, default=1)
	parser.add_argument('--n_items', type=int, default=10)
		
	args = parser.parse_args()

	reset_random(args.random_seed)

	train_dataset = RatingsDataset(args.data_path, split='train')
	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
	
	user_to_idx = train_dataset.user_to_idx.copy()
	idx_to_item = train_dataset.idx_to_item.copy()
	rated = train_dataset.get_rated_items_by_user(args.user_id).copy()

	model = torch.load(args.model_path)

	top_k_recommendations = get_top_k_recommendations(model,
		user_to_idx, 
		idx_to_item, 
		rated, 
		user_id=args.user_id,
		k=args.n_items
	)
	
	print(top_k_recommendations)


if __name__ == '__main__':
	main()