import time
from collections import deque
import keras as K
import keras.losses
import numpy as np
import random
import pandas as pd
import keras.backend
from keras.callbacks import Callback
from tqdm import tqdm

pd.options.mode.chained_assignment = None
import dataConstructor
from sklearn import preprocessing
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import MinMaxScaler
from os import environ

environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, BatchNormalization
from keras.callbacks import TensorBoard, ModelCheckpoint


def set_targets(data: pd.DataFrame, target: str, future_p: int):
    """
    Prepares target data from passed df in groups of var FUTURE_PERIOD

    Required Arguments:
        data(pd.DataFrame): data to be processed.
        target(str): target symbol to have labels set to ie. "BTC"
        future_p(int): number of future periods to have in label.

    :returns
        (pd.Dataframe): data modified with labels appended.
    """
    for i in range(future_p):
        data[f'target_{i}'] = list(data[f'{target}_close'].shift(-i))
    return data


def add_avgs(data: pd.DataFrame, moving_avg: list, ema: list):
    """
    Adds moving avg and exponential moving avg to data.
    Uses tqdm for a progress bar

    Required Arguments:
        data(pd.DataFrame): data to be processed.
        moving_avg(list): moving average windows to iterate through.
        ema(list):  exponential moving average windows to iterate through.

    :returns
        (pd.Dataframe): data modified with moving averages appended.
    """
    print('\tConstructing Moving avg and Exponential moving avg...')
    total = 0
    for col in data.columns:
        if '_close' in col:
            total += 1 * len(moving_avg) * len(ema)
    with tqdm(total=total, unit=' Columns Created') as mavg_prog_bar:
        if len(moving_avg) > 0:
            for col in data.columns:
                if '_close' in col:
                    for sum_mavg in moving_avg:
                        data[f'{col}_MA{sum_mavg}'] = data[col].rolling(window=sum_mavg).mean()
                        mavg_prog_bar.update(1)
                        if len(ema) > 0:
                            for e in ema:
                                data[f'{col}_{m}EMA{e}'] = data[f'{col}_MA{sum_mavg}'].ewm(span=e).mean()
                                mavg_prog_bar.update(1)
    data.dropna(inplace=True)
    return data


def sequence_data(data: pd.DataFrame, sequence_len: int, future_p: int):
    """
    Splits data into chunks of sequence_len

    Required Arguments:
        data(pd.DataFrame): data to be processed.
        sequence_len(int): length of sequences.
        future_p(int):  number of future labels to separate.

    :returns
        (list,list):  tuple with x, and y data.
    """
    sequential_data = []
    seq = deque(maxlen=sequence_len)
    for i in tqdm(data.values, unit=' Sequences Appended'):
        seq.append([n for n in i[:-future_p]])
        if len(seq) == SEQ_LEN:
            sequential_data.append([np.array(seq), i[-future_p:]])
    x = []
    y = []
    for seq, target in sequential_data:
        seq.shape = (sequence_len, len(data.columns))
        x.append(seq)
        y.append(target)
    return x, y


def build_data(data: pd.DataFrame, target: str, seq: int, future_p: int, moving_avg: list, ema: list):
    """
    Drops columns to drop.
    Splits data into training, validation and testing
    Normalizes data
    Takes given data and target symbol: adds moving averages
    Splits data per section into chunks of size: SEQ_LEN

    Required Arguments:
        data(pd.DataFrame): main dataframe to build sub data structures from.
        target(str): symbol to be targeted for labels ie. "BTC"
        seq(int): length of sequential data.
        future_p(int): number ot labels in the future.
        moving_avg(list): list of moving averages to add to data.
        ema(list): list of exponential moving averages to add to data.

    :returns
        (np.array() * 5, int, int): train_X, train_y, val_X, val_y, test_X, text_y, seq, features
    """
    # drop columns selected to be dropped.
    if len(DROP_SYMBOLS) > 0:
        for to_drop in DROP_SYMBOLS:
            data.drop(columns=[col for col in data.columns if to_drop in col], inplace=True)
    # add moving averages
    data = add_avgs(data=data, moving_avg=moving_avg, ema=ema)

    # get first 70% | 20% | last 10%
    train_df = data[:int(len(data.index) * 0.7)]
    features = len(train_df.columns)
    train_mean = train_df.mean()
    train_std = train_df.std()
    validation_df = data[int(len(data.index) * 0.7):-int(len(data.index) * 0.1)]
    test_df = data[-int(len(data.index) * 0.1):]
    del data  # remove data to help memory.

    # normalize data
    train_df = (train_df - train_mean) / train_std
    validation_df = (validation_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    # append future price of number of periods in the future from future period
    train_df = set_targets(train_df, target=target, future_p=future_p)
    validation_df = set_targets(validation_df, target=target, future_p=future_p)
    test_df = set_targets(test_df, target=target, future_p=future_p)

    # split data into chunks of seq size.
    train_X, train_y = sequence_data(train_df, sequence_len=seq, future_p=future_p)
    val_X, val_y = sequence_data(validation_df, sequence_len=seq, future_p=future_p)
    test_X, test_y = sequence_data(test_df, sequence_len=seq, future_p=future_p)

    print(f'Training Sequences:\t{len(train_X)}\nValidation Sequences:\t{len(val_X)}\nTest Sequences:\t{len(test_X)}')
    return train_X, train_y, val_X, val_y, test_X, test_y, seq, features


def make_model(seq: int, features: int):
    # Shape [batch, time, features] => [batch, time, lstm_units]
    drop = [0.3, 0.3, 0.3, 0.3]
    model = Sequential([
        LSTM(16, input_shape=(None, seq, features), return_sequences=True, dropout=drop[0]),
        LSTM(16, return_sequences=False, dropout=drop[2]),
        BatchNormalization(),
    ])
    # m.add(LSTM(16, return_sequences=True, dropout=drop[1]))
    m.add()
    m.add()
    # m.add(Dense(32, activation='relu'))
    m.add(Dropout(drop[3]))
    m.add(Dense(16, activation='relu'))
    m.add(Dense(2, activation='sigmoid'))
    lr = 0.001
    # lr = tf.keras.optimizers.schedules.CosineDecayRestarts(0.001, np.ceil(len(train_x) / BATCH_SIZE) * 20, 0.8, 0.8)
    # op = tf.keras.optimizers.SGD(learning_rate=lr, nesterov=True)
    op = tf.keras.optimizers.Adam(learning_rate=lr)
    # *** TRY LOSS 'mean_squared_error' ***
    # loss = keras.losses.BinaryCrossentropy()
    loss = keras.losses.SparseCategoricalCrossentropy()
    m.compile(loss=loss, optimizer=op, metrics=['accuracy'])


def _classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0


def _refine_data(cry, df):
    for to_drop in DROP_SYMBOLS:
        df.drop(columns=[col for col in df.columns if to_drop in col], inplace=True)
    df['future'] = df[f'{cry}_close'].shift(-FUTURE_PERIOD)
    df['target'] = list(map(_classify, df[f'{cry}_close'], df['future']))
    df.drop(columns=['future'], inplace=True)
    times = sorted(df.index.values)
    last_5pct = 0  # times[-int(VAL_PCT * len(times))]
    validation_main_df = df[(df.index >= last_5pct)]
    df = df[(df.index < last_5pct)]
    del times
    t_x, t_y = _preprocess_df(df)
    validation_x, validation_y = _preprocess_df(validation_main_df)
    print(f'Train data: {len(t_x)} Validation: {len(validation_x)}')
    print(f"TRAINING don't buys {t_y.count(0)}, Buys: {t_y.count(1)}")
    print(f"VALIDATION don't buys: {validation_y.count(0)}, Buys {validation_y.count(1)}\n\n")
    time.sleep(2)
    return t_x, t_y, validation_x, validation_y


def _preprocess_df(df):
    for col in tqdm(df.columns, unit=' column(s)'):
        if col not in ['target', 'time'] and 'Buy' not in col and 'Sell' not in col:
            df[col] = df[col].pct_change()
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df[col].dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)
        df[col] = pd.to_numeric(df[col], downcast='integer')
        df[col] = pd.to_numeric(df[col], downcast='float')

    df.dropna(inplace=True)
    df = df[[c for c in df if c not in ['target']] + ['target']]
    df.sort_index(inplace=True)
    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)
    for i in tqdm(df.values, unit=' prev day(s) append'):
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]])
    del df
    random.shuffle(sequential_data)
    buys = []
    sells = []
    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target])
    del sequential_data
    random.shuffle(buys)
    random.shuffle(sells)
    lower = min(len(buys), len(sells))
    buys = buys[:lower]
    sells = sells[:lower]
    sequential_data = buys + sells
    random.shuffle(sequential_data)
    x = []
    y = []
    for seq, target in sequential_data:
        x.append(seq)
        y.append(target)
    return x, y


def _gen_model_name(cry, model):
    summary_list = []
    name = ''
    model.summary(line_length=150, print_fn=lambda x: summary_list.append(x))
    for line in summary_list[4:-5]:
        line = line.replace(' ', '').replace('_', '').replace('=', '')
        sub_line = line.split('(')[0]
        if not sub_line:
            continue
        if 'batchnormalization' in sub_line:
            name += 'BN_'
        elif 'lstm' in sub_line:
            name += f"L{line.split('(')[-1].split(')')[0].split(',')[-1]}_"
        elif 'dense' in sub_line:
            name += f"D{line.split('(')[-1].split(')')[0].split(',')[-1]}_"
        else:
            continue

    d = str(drop).replace(' ', '').replace(',', '_')
    ma = str(MOVING_AVG).replace(' ', '').replace(',', '_')
    ema_s = str(EMA_SPAN).replace(' ', '').replace(',', '_')
    drop_col = ' '.join(DROP_SYMBOLS).strip().replace(' ', '_')
    m_name = f'{cry}_{name}DROP{d}_B{BATCH_SIZE}_F{FUTURE_PERIOD}_{int(time.time())}'
    m_args = f'S{SEQ_LEN}-MA{ma}-EMAspan{ema_s}-DSYM[{drop_col}]'
    return m_name, m_args


class LearningRateLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self._supports_tf_logs = True

    def on_epoch_end(self, epoch, logs=None):
        if logs is None or "learning_rate" in logs:
            return
        logs["learning_rate"] = self.model.optimizer.lr


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
np.random.seed(7)
THREADS = 2  # 2 max
DATA_SIZE = 1000
EPOCHS = 5
DATA_PERIOD = 1  # in min [1, 3, 5, 15, 30, 60, 120, 240, 360, 720]
# parameters
BATCH_SIZE = [16]
FUTURE_PERIOD = [3]
SEQ_LEN = [120]
DROP_SYMBOLS = []
MOVING_AVG = [[5], [5, 10]]  # [5, 10, 30, 60]
EMA_SPAN = [[4], [4, 8]]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if len(tf.config.list_physical_devices('GPU')) > 0:
    print('\tUSING GPU\n')
else:
    print('\tCPU ONLY **\n')

SYMBOLS, master_dataFrame = dataConstructor.get_data(iteration_count=DATA_SIZE, data_period=DATA_PERIOD,
                                                     threads=THREADS)

master_dataFrame = master_dataFrame[:int(len(master_dataFrame.index) / 20)]

for sym in SYMBOLS:
    if sym in DROP_SYMBOLS:
        continue

    for seq_len in SEQ_LEN:
        for future_period in FUTURE_PERIOD:
            for mavg in MOVING_AVG:
                for ema_span in EMA_SPAN:
                    for batch_size in BATCH_SIZE:
                        # split data into train, validation, and test chunks (70%,20%,10%)
                        train_x, train_y, val_x, val_y, test = build_data(data=master_dataFrame,
                                                                          target=sym, seq=seq_len, future_p=future_period,
                                                                          moving_avg=mavg, ema=ema_span)

    # make model

    # train model

    # test model

# TRY DROPPING EXCESS DATA IE: OPEN, LOW, HIGH PRICES
for _ in range(1):
    for cry in SYMBOLS:
        if cry in DROP_SYMBOLS:
            continue
        # if cry in ['BTC', 'XRP']:
        #    continue
        train_x, train_y, val_x, val_y = _refine_data(cry, master_dataFrame)
        train_x = np.asarray(train_x)
        print(train_x.shape)
        print(train_x.shape[1:])
        train_y = np.asarray(train_y)
        val_x = np.asarray(val_x)
        val_y = np.asarray(val_y)

        m = Sequential()
        drop = [0.3, 0.3, 0.3, 0.3]
        m.add(LSTM(16, input_shape=(train_x.shape[1:]), return_sequences=True, dropout=drop[0]))
        # m.add(LSTM(16, return_sequences=True, dropout=drop[1]))
        m.add(LSTM(16, return_sequences=False, dropout=drop[2]))
        m.add(BatchNormalization())
        # m.add(Dense(32, activation='relu'))
        m.add(Dropout(drop[3]))
        m.add(Dense(16, activation='relu'))
        m.add(Dense(2, activation='sigmoid'))
        lr = 0.001
        # lr = tf.keras.optimizers.schedules.CosineDecayRestarts(0.001, np.ceil(len(train_x) / BATCH_SIZE) * 20, 0.8, 0.8)
        # op = tf.keras.optimizers.SGD(learning_rate=lr, nesterov=True)
        op = tf.keras.optimizers.Adam(learning_rate=lr)
        # *** TRY LOSS 'mean_squared_error' ***
        # loss = keras.losses.BinaryCrossentropy()
        loss = keras.losses.SparseCategoricalCrossentropy()
        m.compile(loss=loss, optimizer=op, metrics=['accuracy'])
        model_name, model_args = _gen_model_name(cry=cry, model=m)
        print(f'\n{model_name}\n{model_args}\n')
        # https://gist.github.com/jeremyjordan/5a222e04bb78c242f5763ad40626c452
        tensorboard = TensorBoard(log_dir=f'log/{model_name}')
        checkpoint_format = 'E{epoch:02d}-A{val_accuracy:.4f}-L{val_loss:.4f}'
        checkpoint = ModelCheckpoint("models/{}/{}__{}.h5".format(model_name, checkpoint_format, model_args),
                                     monitor='val_accuracy', verbose=1,
                                     save_best_only=True, mode='max')

        history = m.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(val_x, val_y),
                        callbacks=[LearningRateLogger(), tensorboard, checkpoint])
        x = train_x[0]
        prediction = m.predict(train_x)
        del m
        keras.backend.clear_session()

"""
try percent change before moving avg and EMA
"""
