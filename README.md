# Bybit-Trader-NN
#### Video Demo: https://youtu.be/6GfaZidsf3M

## Introduction

Bybit neural network trader is an automatic Cryptocurrency trading application. Automatically trading derivative "Inverse Perpetual" contracts on [Bybit](https://www.bybit.com/en-US/) with an easy to use GUI to keep track of your balance, and performance metrics.
Contained in the GUI is also options to train new models with ability to change layer configurations.
https://youtu.be/6GfaZidsf3M


#### gui.py

Start by running: *gui.py*
- contains the functions to initialize and construct the GUI interface.

#### bin/bybitRun.py

- contains the threads, loops and functions that allows the user to interact with the interface.

#### bin/dataHandler.py

- Handles fetching cryptocurrency data from bybit servers via API calls and parsing it into pandas dataframes.

#### bin/networkTraining.py

- When initiated via the user interface; *networkTraining.py* will parse, and train a neural network outputting feedback if verbose is true on the user interface.
- When initiated via *networkTraining.py*; training parameters can be changed in the *run* function and model layers can be changed in the *_make_model* function. These can be changed in much greater detail than training with the gui

##### Training Settings:
- **Target**: The curreny type the model will be trained on
- **Future**: Number of peiods in the future the model will try to predict ie. **Future** is 2 & **Data Period** is 3min, the model will be trained to predict the price 6min in the future.
- **Data Size**: Retrive the number of data chunks from Bybit, Each  1 "Data Size" value is equal to 200 rows of data. More data helps the model train more accurately but requires more system memory.
- **Data Period**: Time between each period. has no effect on end **Data Size**.
- **Seq Length**: The model trains timeseries data in Chunks or Sequences of data. If **Seq Length** is 128 the model will train on chunks of 128 rows of price data.
- **Batch Size**: Number of Chunks/Sequences of data to group together to train on. Higher **Batch Size** the faster the model can train, but lowers the frequency the model updates its weights, and requires more system memory.
- **Moving Average**: Computes a moving average for the model to train on. Mutiple can be added by separating values by "," ie. 10,20,30
- **EM Avg (Exponential Moving Average)**: A weighted moving that gives more weighting to recent price data

#### bin/res/*

- Contains resources, images, and icons.
- *b_df* contains user account balance history from bybit.
- *Config.yaml* contains configuration information for user settings.

#### data/*

- Contains cryptocurrency data taken from bybit to allow quicker training on the same set of data.


#### Future Plans

- Incorporate an auto train function to automaticly retrain the model after *X* number of iterations or minutes
- Add functions to withdraw funds to a wallet within the GUI
- Have GUI be resizable
- Add button to only display need-to-know information in a much smaller window 

