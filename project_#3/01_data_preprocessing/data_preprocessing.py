import os
import pandas as pd

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

# Reading Data
def load_dataset():
    full_path = os.path.join(
        DIR_PATH, './../../datasets/project_#3/02_section_data.csv')
    dataset = pd.read_csv(full_path, sep=',')
    return dataset

if __name__ == "__main__":
	dataset = load_dataset()
	
	X = dataset.iloc[:, :-1].values
	y = dataset.iloc[:, 3].values
