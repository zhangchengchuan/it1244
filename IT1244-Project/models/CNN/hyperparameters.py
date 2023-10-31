# Hyperparameters
target_length = 200000  # Check data_exploration.ipynb. average length is approximately this number
# target_sr = 16000  # All audio are at 16K Hz
n_fft = 2048 # Power 2
hop = 1024 # Typically 0.5/0.75 of n_fft
# train_dataframe = df.sample(frac=0.8, random_state=42) # From above
# test_dataframe = df.loc[~df.idx.isin(train_dataframe.idx)]
lr = 0.001
epochs = 50
batch_size = 32
n_splits = 2