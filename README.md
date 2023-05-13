# RecEngineMF

<br />

# Data

## [MovieLens 20M Dataset](https://grouplens.org/datasets/movielens/20m/)

#### Command to Download and Extract Data

1. **Download** MovieLens 20M Data

	```bash
	wget --output-document=./ml-20m.zip  https://files.grouplens.org/datasets/movielens/ml-20m.zip
	```

2. Once the download is complete, **extract** the dataset

	```bash
	unzip ml-20m.zip
	```

<br />

# Usage

## Install Dependencies

1. **Install Python:** Make sure Python is installed on your system. If not, you can download and install Python from the official Python website: https://www.python.org/downloads/

2. **Create a virtual environment:** 

	```bash
	python -m venv myenv
	```

3. **Activate the virtual environment**

	> For Windows
	```bash
	myenv\Scripts\activate
	```

	> For macOS/Linux
	```bash
	source myenv/bin/activate
	```

4. **Install the dependencies**
	
	```bash
	pip install -r requirements.txt
	```
5. **Add wandb API key**
	
	Sign in to https://wandb.ai and get your API key.
	<br />
	Create a file **secrets.json** in the root directory and put your wandb API key.

	```json
	{
		"WANDB_API_KEY": "YOUR_API_KEY"
	}
	```

<br />

## Train

```bash
python train.py --data_path DATA_PATH [--emb_size EMB_SIZE] [--random_seed RANDOM_SEED] 
                [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--learning_rate LEARNING_RATE] 
                [--weight_decay WEIGHT_DECAY] [--step_size STEP_SIZE] [--gamma GAMMA] 
                [--patience PATIENCE] [--model_name MODEL_NAME] [--metrics_csv_name METRICS_CSV_NAME]
                [--silent] [--log_wandb]
```

#### Required Flag
- **--data_path:** Path to the CSV file containing the ratings data.

#### Optional Flags
- **--emb_size:** Size of the embedding for users and items. Default is **100**.
- **--random_seed:** Random seed for reproducibility. Default is **42**.
- **--batch_size:** Batch size for training. Default is **64000**.
- **--epochs:** Number of epochs for training. Default is **100**.
- **--learning_rate:** Learning rate for optimizer. Default is **0.001**.
- **--weight_decay:** Weight decay for optimizer. Default is **1e-5**.
- **--step_size:** Step size for learning rate scheduler. Default is **10**.
- **--gamma:** Gamma value for learning rate scheduler. Default is **0.1**.
- **--patience:** Patience for early stopping based on validation loss. Default is **3**.
- **--model_name:** Name of the trained model file to be saved. Default is **'mf_model.pth'**.
- **--metrics_csv_name:** Name of the CSV file to save the training metrics. Default is **'metrics.csv'**.
- **--silent:** Whether to hide verbose output during training.
- **--log_wandb:** Whether to log metrics into weights and bias (wandb.ai).

<br />

## Test

```bash
python test.py  --data_path DATA_PATH --model_path MODEL_PATH [--batch_size BATCH_SIZE] [--random_seed RANDOM_SEED]
```

#### Required Flags
- **--data_path:** Path to the CSV file containing the ratings data.
- **--model_path:** Path to the trained model file to be loaded for testing.

#### Optional Flags
- **--batch_size:** Batch size for testing. Default is **64000**.
- **--random_seed:** Random seed for reproducibility. Default is **42**.

<br />

## Run Inference

```bash
python inference.py --data_path DATA_PATH --model_path MODEL_PATH --user_id USER_ID [--n_items N_ITEMS]
```

#### Required Flags
- **--data_path:** Path to the CSV file containing the ratings data.
- **--model_path:** Path to the trained model file to be loaded for testing. 
- **--user_id:** The id of the user for whom item is to be recommended.

#### Optional Flags
- **--n_items:** The top n number of items to be recommended to the user. Default is **10**.


<br />

## Plot Curve

```bash
python plot.py --metrics_csv_path METRICS_CSV_PATH [--patience PATIENCE] [--file_name FILE_NAME]
```

#### Required Flags
- **--metrics_csv_path:** Path to the CSV file containing the mertics data. [ CSV file with column names: **'Epoch', 'Train Loss', 'Val Loss'** ]

#### Optional Flags
- **--patience:** Patience for early stopping. Default is **None**.
- **--file_name:** The name for saving the plot. Default is **loss_curve.png**.

<br />

# References

- **[Matrix Factorization Techniques for Recommender Systems](https://ieeexplore.ieee.org/document/5197422)** 
(Y. Koren, R. Bell and C. Volinsky, "Matrix Factorization Techniques for Recommender Systems," in Computer, vol. 42, no. 8, pp. 30-37, Aug. 2009, doi: 10.1109/MC.2009.263.)