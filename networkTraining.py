import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import time
from collections import deque
import keras.losses
import numpy as np
import pandas as pd
import glob
import bin.bybit_run

pd.options.mode.chained_assignment = None
import keras.backend
from tqdm import tqdm
import matplotlib.pyplot as plt
import dataConstructor
from sklearn.preprocessing import MinMaxScaler
from os import environ

environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from keras.models import Sequential
from keras import layers
from keras.callbacks import TensorBoard, ModelCheckpoint

VERBOSE = 0
GUI = None
OUTCOME_TYPE = 'r'
PARENT = None


class bcolors:
    """
    Terminal colors
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def verbose_print(verbose: int, text: str, max_val: int = 0, update: int = 0, reset: bool = False):
    """
    Will print (x!=0) or withhold printing (x==0) to terminal depending on global VERBOSE value.
    Will update GUI progress bad and text as long as MAX_VAL is >= 0
    """
    global GUI, PARENT
    if PARENT is not None:
        PARENT.check_kill()
    if verbose != 0:
        if text != '':
            print(bin.bybit_run.bcolors.HEADER + text)

    if GUI is not None and max_val >= 0:
        update_progress(text, max_val, update, reset)


def update_progress(text: str = '', max_value: int = 0, update: int = 0, reset: bool = False):
    global GUI
    if GUI is not None:
        text = text[:30]
        if reset:
            GUI.prog_bar_train['value'] = 0
            GUI.prog_bar_train['maximum'] = max_value
            GUI.label_train_status['text'] = ''
        if len(text) > 0:
            escapes = ''.join([chr(char) for char in range(1, 32)])
            GUI.label_train_status['text'] = text.translate(str.maketrans('', '', escapes))

        GUI.prog_bar_train['value'] += update


def _classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0


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
    verbose_print(VERBOSE, 'Constructing MA & EMA...', max_val=1, reset=True)

    if len(moving_avg) > 0:
        for col in data.columns:
            if '_close' in col:
                for sub_mavg in moving_avg:
                    data[f'{col}_MA{sub_mavg}'] = data[col].rolling(window=sub_mavg).mean()
                    if len(ema) > 0:
                        for e in ema:
                            data[f'{col}_EMA{sub_mavg}'] = data[f'{col}_MA{sub_mavg}'].ewm(span=e).mean()
    data.dropna(inplace=True)
    verbose_print(VERBOSE, '', update=1, reset=True)
    return data


def preprocess_data(data):
    global VERBOSE
    verbose_print(VERBOSE, 'Scaling Data...', max_val=len(data.columns), reset=True)
    for col in tqdm(data.columns, unit=' column(s)', disable=not bool(VERBOSE)):
        if col not in ['time']:
            # data[col] = data[col].pct_change()
            data.replace([np.inf, -np.inf], np.nan, inplace=True)
            data[col] = MinMaxScaler(feature_range=(-1, 1)).fit_transform(data[col].values.reshape(-1, 1))
            # data[col] = preprocessing.scale(data[col].values)
        # data[col] = pd.to_numeric(data[col], downcast='integer')
        data[col] = pd.to_numeric(data[col], downcast='float')
        verbose_print(VERBOSE, '', update=1)
    data.dropna(inplace=True)
    return data


def split_data(data):
    # get first 70% | 20% | 10% split and normalize the data
    data = preprocess_data(data)
    train = data.iloc[:int(len(data.index) * 0.7)]
    val = data.iloc[int(len(data.index) * 0.7):-int(len(data.index) * 0.1)]
    test = data.iloc[-int(len(data.index) * 0.1):]
    return train, val, test


def set_targets(in_data: pd.DataFrame, target: str, future_p: int):
    """
    Prepares target data from passed df in groups of var FUTURE_PERIOD baced on OUTCOME_TYPE
    if OUTCOME_TYPE == 'r' regression will be used else if OUTCOME_TYPE == 'c' classification will be used

    Required Arguments:
        data(pd.DataFrame): data to be processed.
        target(str): target symbol to have labels set to ie. "BTC"
        future_p(int): number of future periods to have in label.

    :returns
        (pd.Dataframe): data modified with labels appended.
    """
    global OUTCOME_TYPE
    t_df = pd.DataFrame()
    if OUTCOME_TYPE == 'r':
        t_df['target'] = in_data[f'{target}_close'].shift(-future_p)
    elif OUTCOME_TYPE == 'c':
        t_df['future'] = in_data[f'{target}_close'].shift(-future_p)
        t_df['target'] = list(map(_classify, in_data[f'{target}_close'], in_data['future']))
        t_df.drop(columns=['future'], inplace=True)
    else:
        verbose_print(VERBOSE, f"ERROR: OUTCOME_TYPE:{OUTCOME_TYPE} not in: ['r','c']")
        exit()
    # in_data = (in_data - mean) / std
    in_data = pd.concat([in_data, t_df], axis=1, join='inner')
    in_data.dropna(inplace=True)
    return in_data


def sequence_data(data: pd.DataFrame, sequence_len: int):
    """
    Splits data into chunks of sequence_len

    Required Arguments:
        data(pd.DataFrame): data to be processed.
        sequence_len(int): length of sequences.
        future_p(int):  number of future labels to separate.

    :returns
        (list,list):  tuple with x, and y data.
    """
    global VERBOSE
    verbose_print(VERBOSE, f'Appending ~{len(data.values)} Sequences', max_val=len(data.values), reset=True)
    time.sleep(0.5)
    sequential_data = []
    seq = deque(maxlen=sequence_len)
    for i in tqdm(data.values, unit=' Sequences Appended', disable=not bool(VERBOSE)):
        seq.append([n for n in i[:-1]])
        if len(seq) == sequence_len:
            sequential_data.append([np.array(seq), i[-1:]])
        verbose_print(VERBOSE, '', update=1)
    x = []
    y = []
    for seq, target in sequential_data:
        x.append(seq)
        y.append(target)
    return np.asarray(x), np.asarray(y)


def build_data(data: pd.DataFrame, target: str, seq: int, future_p: int, drop: list, moving_avg: list,
               ema: list, verbose: int = VERBOSE, test_only: bool = False):
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
    global VERBOSE
    VERBOSE = verbose
    if len(drop) > 0:
        for to_drop in drop:
            data.drop(columns=[col for col in data.columns if to_drop in col], inplace=True)
    # add moving averages
    data = add_avgs(data=data, moving_avg=moving_avg, ema=ema)
    features = len(data.columns)
    data = set_targets(data, target=target, future_p=future_p)

    train_df, validation_df, test_df = split_data(data)
    del data  # remove data to help memory.
    # append future price of number of periods in the future from future period and normalize data
    # train_df = set_targets(train_df, target=target, future_p=future_p)
    # validation_df = set_targets(validation_df, target=target, future_p=future_p)
    # test_df = set_targets(test_df, target=target, future_p=future_p)
    # split data into chunks of seq size.
    if not test_only:
        train_X, train_y = sequence_data(train_df, sequence_len=seq)
        val_X, val_y = sequence_data(validation_df, sequence_len=seq)
    test_X, test_y = sequence_data(test_df, sequence_len=seq)
    if not test_only:
        verbose_print(VERBOSE, f'Training Sequences:\t{len(train_X)}\nValidation Sequences:\t{len(val_X)}\nTest Sequences:\t{len(test_X)}',
                      max_val=-1)
        return train_X, train_y, val_X, val_y, test_X, test_y, features
    else:
        return test_X, test_y


def make_model(seq: int, features: int):
    # Shape [batch, time, features] => [batch, time, lstm_units]
    m = Sequential([
        # layers.Normalization(axis=1),
        layers.GRU(128, input_shape=(seq, features), return_sequences=True),
        layers.GRU(64, return_sequences=False),
        # layers.GRU(64, return_sequences=False),
        layers.Dropout(0.3),
        layers.Dense(32),
        layers.Dropout(0.3),
        layers.Dense(8),
        layers.Dense(1, activation='tanh')
    ])
    m.summary()
    """DECAY FIXED AND COSINE"""
    lr = 0.0005
    # lr = tf.keras.optimizers.schedules.CosineDecayRestarts(0.001, np.ceil(len(t_x_data) / batch_s) * 20, 0.8, 0.8)
    """OPTIMIZERS"""
    op = tf.keras.optimizers.Adam(learning_rate=lr)
    # op = tf.keras.optimizers.SGD(learning_rate=lr, nesterov=True)
    """
    LOSS FUNCTIONS                  | activation |  Loss
    Regression:     numerical value:    Linear      MSE
    Classification: binary outcome:     Sigmoid     BCE
    Classification: single label:       Softmax     Cross Entropy
    Classification: multi labels:       Sigmoid     BCE
    """
    loss = keras.losses.MeanSquaredError()
    m.compile(loss=loss, optimizer=op)
    return m


def gen_model(seq: int, features: int, model_args: list):
    m = Sequential()
    key = bin.bybit_run.LAYER_OPTIONS
    ret_seq_layers = ['LSTM', 'GRU', 'RNN', 'SimpleRNN']
    for count, layer in enumerate(model_args):
        if layer[0] in ret_seq_layers:
            ret_seq = False
            if count < len(model_args) - 1 and model_args[count + 1][0] in ret_seq_layers:
                ret_seq = True
            if count == 0:
                m.add(key[layer[0]](layer[1], input_shape=(seq, features), return_sequences=ret_seq))
            else:
                m.add(key[layer[0]](layer[1], return_sequences=ret_seq))
        else:
            if count == 0:
                m.add(key[layer[0]](layer[1], input_shape=(seq, features)))
            else:
                m.add(key[layer[0]](layer[1]))
        if count == len(model_args) - 1:
            m.add(layers.Dense(1, activation='tanh'))
    m.compile(loss=keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam())
    return m


def gen_model_name(model: keras.models, sym: str, b_s: int, fut_p: int, s_l: int, ma: list, ema: list, drop_col: list):
    name = ''
    c = 0
    while True:
        try:
            layer = model.get_layer(index=c)
            name += ''.join(l[0].upper() for l in layer._name.split('_') if not l.isdigit())
            try:
                name += f'{layer.units}'
            except AttributeError:
                try:
                    name += f'{layer.rate}'
                except AttributeError:
                    pass
            name += '_'
        except ValueError:
            break
        c += 1

    ma = str(ma).replace(' ', '').replace(',', '_')
    ema_s = str(ema).replace(' ', '').replace(',', '_')
    drop_col = ' '.join(drop_col).strip().replace(' ', '_')
    m_name = f'{sym}_{name}B{b_s}_F{fut_p}_{int(time.time())}'
    m_args = f'S{s_l}-MA{ma}-EMAS{ema_s}-DSYM[{drop_col}]'
    verbose_print(VERBOSE, f'\n{m_name}\n{m_args}\n', max_val=-1)
    return m_name, m_args


def plot_data(data, plot_name=None):
    fig, axs = plt.subplots(1, len(data))
    if len(data) == 1:
        plot_d = data[0]
        axs.set_title(plot_d[0], fontsize=10)
        axs.plot(plot_d[1], 'tab:blue', label='Control')
        axs.plot(plot_d[2], 'tab:green', label='Prediction')
    else:
        for c, plot_d in enumerate(data):
            axs[c].set_title(plot_d[0], fontsize=10)
            axs[c].plot(plot_d[1], 'tab:orange', label='Test')
            axs[c].plot(plot_d[2], 'tab:green', label='Pred')

    plt.legend()
    if plot_name is not None:
        plt.savefig(f'{plot_name}.png', bbox_inches='tight')
    else:
        plt.show()


class LearningRateLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self._supports_tf_logs = True

    def on_epoch_end(self, epoch, logs=None):
        if logs is None or "learning_rate" in logs:
            return
        logs["learning_rate"] = self.model.optimizer.lr


class UpdateProgressTraining(keras.callbacks.Callback):

    def __init__(self, total_batches, total_epochs):
        super().__init__()
        self.current_batch = 1
        self.total_batches = total_batches
        self.current_epoch = 1
        self.total_epochs = total_epochs
        self.current_loss = 0
        self.text = 'Epoch: {}/{} | Loss: {}'

    def on_test_begin(self, logs=None):
        pass

    def on_test_batch_begin(self, batch, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch + 1
        self._call_update(update_val=0, reset=True)

    def on_batch_end(self, batch, logs=None):
        global GUI
        self.current_batch = batch + 1
        self.current_loss = round(logs['loss'], 8)
        PARENT.check_kill()
        self._call_update(update_val=1, reset=False)

    def _call_update(self, update_val, reset):
        update_progress(text=self.text.format(self.current_epoch, self.total_epochs, self.current_loss),
                        max_value=self.total_batches, update=update_val, reset=reset)


def train(parent, gui, data_args: dict, model_args: list, verbose: int, force_new: bool, key: str, secret: str):
    from keras.callbacks import LambdaCallback
    from bin.bybit_run import Model as bb_Model
    global VERBOSE, GUI, OUTCOME_TYPE, PARENT
    GUI = gui
    VERBOSE = verbose
    OUTCOME_TYPE = 'r'
    PARENT = parent
    drop = []

    if len(tf.config.list_physical_devices('GPU')) > 0:
        verbose_print(VERBOSE, '\tUSING GPU\n')
    else:
        verbose_print(VERBOSE, '\tUSING CPU\n')
    s, master_dataFrame = dataConstructor.get_data(key=key, secret=secret, symbols=[data_args['target']], data_s=data_args['size'],
                                                   data_p=data_args['period'], threads=4, force_new=force_new, verbose=VERBOSE)
    raw_data = master_dataFrame.copy()
    tr_x, tr_y, v_x, v_y, te_x, te_y, features = build_data(data=master_dataFrame, target=data_args['target'], seq=data_args['seq'],
                                                            future_p=data_args['future'], drop=drop,
                                                            moving_avg=data_args['ma'], ema=data_args['ema'])
    active_model = gen_model(data_args['seq'], features, model_args)
    model_name, model_args = gen_model_name(active_model, data_args['target'], data_args['batch'], data_args['future'],
                                            data_args['seq'], data_args['ma'], data_args['ema'],
                                            drop_col=[])

    early_stop_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=data_args['epochs'] + 1, verbose=VERBOSE,
                                                        mode="min", restore_best_weights=True, )

    verbose_print(VERBOSE, f'\n'
                           f'{"*" * 20}\n'
                           f'Symbol:\t{data_args["target"]}\nSequence length:\t{data_args["seq"]}\nFuture prediction period:\t{data_args["future"]}\n'
                           f'Moving Average(s):\t{data_args["ma"]}\nExponential Span:\t{data_args["ema"]}\n'
                           f'{"*" * 20}\n', max_val=-1)
    active_model.fit(tr_x, tr_y, batch_size=data_args['batch'], epochs=data_args['epochs'],
                     validation_data=(v_x, v_y), use_multiprocessing=True, verbose=VERBOSE
                     , callbacks=[early_stop_callback,
                                  UpdateProgressTraining(total_batches=len(tr_x) // data_args['batch'], total_epochs=data_args['epochs'])])
    keras.backend.clear_session()
    # active_model.evaluate(te_x, te_y, batch_size=data_args['batch'])

    ret_model = bin.bybit_run.Model(location=f'bin\\res\\models\\{model_name}__{model_args}', model=active_model,
                                    seq=data_args["seq"], batch=data_args['batch'], future=data_args['future'], moving_avg=data_args["ma"],
                                    e_moving_avg=data_args["ema"], drop=drop)
    return raw_data, ret_model


def run():
    global VERBOSE, OUTCOME_TYPE
    VERBOSE = 1
    THREADS = 2  # 2 max
    DATA_SIZE = 100
    DIV_DATA = 1
    EPOCHS = 2
    DATA_PERIOD = 1  # in min [1, 3, 5, 15, 30, 60, 120, 240, 360, 720]
    # parameters
    OUTCOME_TYPE = 'r'  # r = regression | c = classification
    BATCH_SIZE = [64, 128]
    FUTURE_PERIOD = [5]
    SEQ_LEN = [60, 120, 180]
    DROP_SYMBOLS = []  # ['ETH', 'XRP', 'EOS']
    MOVING_AVG = [[5, 10], [3, 5], []]  # [5, 10, 30, 60]
    EMA_SPAN = [[4, 8], [3, 5]]

    if len(tf.config.list_physical_devices('GPU')) > 0:
        verbose_print(VERBOSE, '\tUSING GPU\n\n')
    else:
        verbose_print(VERBOSE, '\tCPU ONLY **\n')
    key = 'jfCGTjZcCIuPDUbdDt'
    secret = 'IHz2iqBanafGGuesd0w4QKjrQyUb2NDgQ0q1'
    time.sleep(3)

    SYMBOLS, master_dataFrame = dataConstructor.get_data(key=key, secret=secret, data_s=DATA_SIZE,
                                                         data_p=DATA_PERIOD, threads=THREADS, )
    back_up_data = master_dataFrame.iloc[:int(len(master_dataFrame.index) / DIV_DATA)]

    plotting_data = []
    SYMBOLS = ['BTC']
    for symbol in SYMBOLS:
        if symbol in DROP_SYMBOLS:
            continue

        for seq_len in SEQ_LEN:
            for future_period in FUTURE_PERIOD:
                for mavg in MOVING_AVG:
                    for ema_span in EMA_SPAN:
                        master_dataFrame = back_up_data.copy()
                        # split data into train, validation, and test chunks (70%,20%,10%)
                        train_x, train_y, val_x, val_y, test_x, \
                        test_y, data_features = build_data(data=master_dataFrame,
                                                           target=symbol, seq=seq_len,
                                                           future_p=future_period, drop=DROP_SYMBOLS,
                                                           moving_avg=mavg, ema=ema_span)

                        for batch_size in BATCH_SIZE:
                            active_model = make_model(seq_len, data_features)
                            model_name, model_args = gen_model_name(active_model, symbol, batch_size, future_period, seq_len, mavg,
                                                                    ema_span,
                                                                    DROP_SYMBOLS)
                            tensorboard = TensorBoard(log_dir=f'log/{symbol}/{model_name}')
                            if OUTCOME_TYPE == 'r':
                                checkpoint_format = 'E{epoch:02d}-L{val_loss:.6f}'
                                parent_dir = f'models/{symbol}'
                                checkpoint = ModelCheckpoint(
                                    "{}/{}/{}__{}.h5".format(parent_dir, model_name, checkpoint_format, model_args),
                                    monitor='val_loss', verbose=2, save_best_only=True, mode='min')
                            else:
                                checkpoint_format = 'E{epoch:02d}-A{val_accuracy:.4f}-L{val_loss:.4f}'
                                parent_dir = f'models/{symbol}'
                                checkpoint = ModelCheckpoint(
                                    "{}/{}/{}__{}.h5".format(parent_dir, model_name, checkpoint_format, model_args),
                                    monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
                            plt.clf()
                            verbose_print(VERBOSE, f'\n'
                                                   f'{"*" * 20}\n'
                                                   f'Symbol:\t{symbol}\nSequence length:\t{seq_len}\nFuture prediction period:\t{future_period}\n'
                                                   f'Moving Average(s):\t{mavg}\nExponential Span:\t{ema_span}\n'
                                                   f'{"*" * 20}\n')
                            history = active_model.fit(train_x, train_y, batch_size=batch_size, epochs=EPOCHS,
                                                       validation_data=(val_x, val_y), use_multiprocessing=True
                                                       , callbacks=[LearningRateLogger(), checkpoint])
                            # [LearningRateLogger(), tensorboard, checkpoint]
                            active_model.evaluate(test_x, test_y, batch_size=batch_size)
                            prediction = active_model.predict(test_x)
                            plot_name = f'{model_name}\n{model_args}'
                            plotting_data = [[plot_name + ' *200*', test_y[:, 0][:200], prediction[:200]],
                                             [plot_name, test_y[:, 0], prediction]]
                            # plotting_data.append([plot_name + ' *100*', test_y[:, 0][:100], prediction[:100]])
                            # plotting_data.append([plot_name, test_y[:, 0], prediction])
                            del active_model
                            keras.backend.clear_session()
                            # plot_data(plotting_data, plot_name=model_name)
                            plot_data(plotting_data)


if __name__ == '__main__':
    run()
