import argparse
import pandas as pd
from utils import plot_loss_curve_from_csv


def main():
	parser = argparse.ArgumentParser(
		description='Plot Train and Validation Loss Curve')
	
	parser.add_argument('--save', type=bool, default=True)
	parser.add_argument('--file_name', type=str, default='loss_curve.png')
	parser.add_argument('--metrics_csv_path', type=str, default='./metrics.csv')
	
	args = parser.parse_args()

	plot_loss_curve_from_csv(args.metrics_csv_path, args.save, args.file_name)


if __name__ == '__main__':
	main()