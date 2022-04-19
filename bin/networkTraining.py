import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import time
from collections import deque
import keras.losses
import numpy as np
import pandas as pd
import bin.bybit_run as bybit_run

pd.options.mode.chained_assignment = None
import keras.backend
from tqdm import tqdm
import matplotlib.pyplot as plt
from bin import data
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras import layers
from keras.callbacks import TensorBoard, ModelCheckpoint


def _classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0


class Training:
    Regression = 'r'
    Classification = 'c'
    OUTCOME_TYPES = [Regression, Classification]

    class LearningRateLogger(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self._supports_tf_logs = True

        def on_epoch_end(self, epoch, logs=None):
            if logs is None or "learning_rate" in logs:
                return
            logs["learning_rate"] = self.model.optimizer.lr

    class UpdateProgressTraining(keras.callbacks.Callback):

        def __init__(self, outer_instance, total_batches, total_epochs):
            super().__init__()
            self.outer_instance = outer_instance
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
            self.current_batch = batch + 1
            self.current_loss = round(logs['loss'], 8)
            self.outer_instance.parent.check_kill()
            self._call_update(update_val=1, reset=False)

        def _call_update(self, update_val, reset):
            self.outer_instance.verbose_print(text=self.text.format(self.current_epoch, self.total_epochs, self.current_loss) + '\r',
                                              max_val=self.total_batches, update=update_val, reset=reset)

    def __init__(self, key: str, secret: str, parent=None, gui=None, verbose=0, outcome_type=Regression):
        self.parent = parent
        self.gui = gui
        self.verbose = verbose
        self.key = key
        self.secret = secret
        self.outcome_type = outcome_type

    def run(self, data_size: int, data_period: int, force_new_dataset: bool, future: list, targets: list, target_outcome: str,
            drop_symbols: list,
            moving_avg: list, e_moving_avg: list, sequence_length: list, batch_size: list, epochs: int, early_stop_patience: int):
        self._training_on()
        master_dataframe = data.Constructor(self, key=self.key, secret=self.secret, data_s=data_size, data_p=data_period, threads=4,
                                            force_new=force_new_dataset).get_data()
        back_up_data = master_dataframe.iloc[:int(len(master_dataframe.index) / 1)]
        for target in targets:
            if target in drop_symbols:
                continue
            for seq in sequence_length:
                for future_period in future:
                    for mavg in moving_avg:
                        for ema_span in e_moving_avg:
                            df = back_up_data.copy()
                            # split data into train, validation, and test chunks (70%,20%,10%)
                            t_x, t_y, v_x, v_y, test_x, test_y, features = self.build_data(df,
                                                                                           target, seq, future_period, drop_symbols,
                                                                                           moving_avg, e_moving_avg)
                            for batch in batch_size:
                                active_model = self._make_model(seq, features)
                                model_name, model_args = self._gen_model_name(active_model, target, batch, future_period, seq, data_period,
                                                                              moving_avg, e_moving_avg, drop_symbols)

                                parent_dir = f'models/{target}'
                                if target_outcome == Training.Regression:
                                    checkpoint_format = 'E{epoch:02d}-L{val_loss:.6f}'
                                    monitor = 'val_loss'
                                    mode = 'min'
                                else:
                                    checkpoint_format = 'E{epoch:02d}-A{val_accuracy:.4f}-L{val_loss:.4f}'
                                    monitor = 'val_accuracy'
                                    mode = 'max'
                                checkpoint = ModelCheckpoint(
                                    "{}/{}/{}__{}.h5".format(parent_dir, model_name, checkpoint_format, model_args),
                                    monitor=monitor, verbose=self.verbose, save_best_only=True, mode=mode)
                                early_stop_callback = keras.callbacks.EarlyStopping(monitor=monitor, patience=early_stop_patience,
                                                                                    verbose=self.verbose, mode=mode,
                                                                                    restore_best_weights=True, )
                                plt.clf()
                                self.verbose_print(f'\n{"*" * 20}\nSymbol:\t{target}\nSequence length:\t{seq}\n'
                                                   f'Future prediction period:\t{future_period}\nMoving Average(s):\t{mavg}\n'
                                                   f'Exponential Span:\t{ema_span}\n{"*" * 20}\n')
                                history = active_model.fit(t_x, t_y, batch_size=batch_size, epochs=epochs,
                                                           validation_data=(v_x, v_y), use_multiprocessing=True,
                                                           callbacks=[self.LearningRateLogger(),
                                                                      checkpoint,
                                                                      early_stop_callback,
                                                                      TensorBoard(log_dir=f'log/{target}/{model_name}')])
                                active_model.evaluate(test_x, test_y, batch_size=batch_size)
                                prediction = active_model.predict(test_x)
                                plot_name = f'{model_name}\n{model_args}'
                                plotting_data = [[plot_name + ' *200*', test_y[:, 0][:200], prediction[:200]],
                                                 [plot_name, test_y[:, 0], prediction]]
                                del active_model
                                keras.backend.clear_session()
                                plot_data(plotting_data)

    def train(self, client, data_args: dict, model_args: dict, force_new: bool):
        """
        add TRIX tripple EMA
        add MACD (12-Period EMA − 26-Period EMA)
        """
        from keras.callbacks import LambdaCallback
        self._training_on()

        master_dataFrame = data.Constructor(self, client=client, data_s=data_args['size'], data_p=data_args['period'], threads=4,
                                            force_new=force_new).get_data()
        raw_data = master_dataFrame.copy()
        tr_x, tr_y, v_x, v_y, te_x, te_y, features = self.build_data(master_dataFrame, target=data_args['target'], seq=data_args['seq'],
                                                                     future_p=data_args['future'], drop=[], moving_avg=data_args['ma'],
                                                                     ema=data_args['ema'])
        active_model = self._gen_model(data_args['seq'], features, model_args)
        model_name, model_name_args = self._gen_model_name(active_model, data_args['target'], data_args['batch'], data_args['future'],
                                                           data_args['seq'], data_args['period'], data_args['ma'], data_args['ema'],
                                                           drop_col=[])
        early_stop_callback = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, verbose=self.verbose, mode="min",
                                                            restore_best_weights=True, )
        self.verbose_print(f'\n{"*" * 20}\nSymbol:\t{data_args["target"]}\nSequence length:\t{data_args["seq"]}'
                           f'\nFuture prediction period:\t{data_args["future"]}\nMoving Average(s):\t{data_args["ma"]}'
                           f'\nExponential Span:\t{data_args["ema"]}\n{"*" * 20}\n', max_val=-1)
        active_model.fit(tr_x, tr_y, batch_size=data_args['batch'], epochs=data_args['epochs'], validation_data=(v_x, v_y),
                         use_multiprocessing=True, verbose=self.verbose - 1,
                         callbacks=[early_stop_callback, self.UpdateProgressTraining(self, total_batches=len(tr_x) // data_args['batch'],
                                                                                     total_epochs=data_args['epochs'])])
        keras.backend.clear_session()
        ret_model = bybit_run.Model(location=f'bin\\res\\models\\{model_name}__{model_name_args}.h5', model=active_model,
                                    seq=data_args["seq"], batch=data_args['batch'], future=data_args['future'], moving_avg=data_args["ma"],
                                    e_moving_avg=data_args["ema"], drop=[], period=data_args['period'], layers=model_args)
        return raw_data, ret_model

    def _training_on(self):
        if len(tf.config.list_physical_devices('GPU')) > 0:
            self.verbose_print('\tUSING GPU\n')
        else:
            self.verbose_print('\tUSING CPU\n')

    def verbose_print(self, text: str, max_val: int = 0, update: int = 0, reset: bool = False):
        """
        Will update GUI progress bar and text as long as MAX_VAL is >= 0
        """
        if self.parent is not None:
            self.parent.check_kill()
        if self.verbose == 0:
            return
        if self.verbose >= 1 and self.gui is not None and max_val >= 0:
            text = text[:30]
            if reset:
                self.gui.prog_bar_train['value'] = 0
                self.gui.prog_bar_train['maximum'] = max_val
                self.gui.label_train_status['text'], self.gui.label_prog_percent['text'] = '', ''
            if len(text) > 0:
                escapes = ''.join([chr(char) for char in range(1, 32)])
                self.gui.label_train_status['text'] = text.translate(str.maketrans('', '', escapes))
            try:
                pct = float(self.gui.prog_bar_train['value'] / self.gui.prog_bar_train['maximum']) * 100
                if update > 0:
                    if pct < 100:
                        pct = str(round(pct, 1))
                        while len(pct) < 3:
                            pct += '0'
                    else:
                        pct = round(pct, 0)
                    pct = f'{pct}%'
            except ZeroDivisionError:
                pct = ''
            self.gui.label_prog_percent['text'] = pct
            self.gui.prog_bar_train['value'] += update

        if self.verbose >= 2 and len(text) > 0:
            print(bybit_run.Bcolors.HEADER + text)

    def build_data(self, input_data: pd.DataFrame, target: str, seq: int, future_p: int, drop: list, moving_avg: list,
                   ema: list, test_data_only: bool = False, prediction=False):
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
        if len(drop) > 0:
            for to_drop in drop:
                input_data.drop(columns=[col for col in input_data.columns if to_drop in col], inplace=True)
        # add moving averages
        input_data = self._add_avgs(input_data, moving_avg=moving_avg, ema=ema)
        input_data = input_data[sorted(input_data.columns)]
        features = len(input_data.columns)
        if not prediction:
            input_data = self._set_targets(input_data, target=target, future_p=future_p)
        input_data = self._preprocess_data(input_data)

        if prediction:
            input_data = input_data[-seq:]
            return input_data
        split_data_list = self._split_data(input_data)
        del input_data  # remove data to help memory.

        if test_data_only:
            del split_data_list[0:2]
            return self._sequence_data(split_data_list[-1], sequence_len=seq)
        else:
            seq_info = ''
            info = ['Training Sequences:\t', 'Validation Sequences:\t', 'Test Sequences:\t']
            out_data = []
            while len(split_data_list) > 0:
                x, y = self._sequence_data(split_data_list[0], sequence_len=seq)
                out_data.append(x)
                out_data.append(y)
                seq_info += info.pop(0)
                split_data_list.pop(0)
            return *out_data, features

    def _add_avgs(self, input_data: pd.DataFrame, moving_avg, ema):
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
        if len(moving_avg) > 0:
            self.verbose_print('Constructing MA & EMA...', max_val=1, reset=True)
            for col in input_data.columns:
                if '_close' in col:
                    for sub_mavg in moving_avg:
                        input_data[f'{col}_MA{sub_mavg}'] = input_data[col].rolling(window=sub_mavg).mean()
                        if len(ema) > 0:
                            for e in ema:
                                input_data[f'{col}_EMA{sub_mavg}'] = input_data[f'{col}_MA{sub_mavg}'].ewm(span=e).mean()
        input_data.dropna(inplace=True)
        self.verbose_print('', update=1, reset=True)
        return input_data

    def _set_targets(self, input_data: pd.DataFrame, target, future_p):
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
        t_df = pd.DataFrame()
        if self.outcome_type == self.Regression:
            t_df['target'] = input_data[f'{target}_close'].shift(-future_p)
        elif self.outcome_type == self.Classification:
            t_df['future'] = input_data[f'{target}_close'].shift(-future_p)
            t_df['target'] = list(map(_classify, input_data[f'{target}_close'], input_data['future']))
            t_df.drop(columns=['future'], inplace=True)
        else:
            self.verbose_print(f"ERROR: OUTCOME_TYPE:{self.outcome_type} not in: {str(self.OUTCOME_TYPES)}")
            exit()
        input_data = pd.concat([input_data, t_df], axis=1, join='inner')
        input_data.dropna(inplace=True)
        return input_data

    def _preprocess_data(self, input_data):
        self.verbose_print('Normalizing Data...', max_val=len(input_data.columns), reset=True)
        for col in tqdm(input_data.columns, unit=' column(s)', disable=not bool(self.verbose)):
            if col not in ['time']:
                # input_data[col] = input_data[col].pct_change()
                input_data.replace([np.inf, -np.inf], np.nan, inplace=True)
                input_data[col] = MinMaxScaler(feature_range=(-1, 1)).fit_transform(input_data[col].values.reshape(-1, 1))
                # input_data[col] = preprocessing.scale(input_data[col].values)
            input_data[col] = pd.to_numeric(input_data[col], downcast='float')
            self.verbose_print('', update=1)
        input_data.dropna(inplace=True)
        return input_data

    def _sequence_data(self, input_data: pd.DataFrame, sequence_len: int):
        """
        Splits data into chunks of sequence_len

        Required Arguments:
            data(pd.DataFrame): data to be processed.
            sequence_len(int): length of sequences.
            future_p(int):  number of future labels to separate.

        :returns
            (list,list):  tuple with x, and y data.
        """
        self.verbose_print(f'Constructing ~{len(input_data.values)} Sequences', max_val=len(input_data.values), reset=True)
        time.sleep(0.5)
        sequential_data = []
        seq = deque(maxlen=sequence_len)
        for i in tqdm(input_data.values, unit=' Sequences Constructed', disable=not bool(self.verbose)):
            seq.append([n for n in i[:-1]])
            if len(seq) == sequence_len:
                sequential_data.append([np.array(seq), i[-1:]])
            self.verbose_print('', update=1)
        x = []
        y = []
        self.verbose_print(f'Splitting {len(sequential_data)} Sequences', max_val=len(sequential_data), reset=True)
        for seq, target in tqdm(sequential_data, unit=' Target Seq Split', disable=not bool(self.verbose)):
            x.append(seq)
            y.append(target)
            self.verbose_print('', update=1)
        return np.asarray(x), np.asarray(y)

    @staticmethod
    def _split_data(input_data):
        # get first 70% | 20% | 10% split and normalize the data
        d_list = [input_data.iloc[:int(len(input_data.index) * 0.7)],
                  input_data.iloc[int(len(input_data.index) * 0.7):-int(len(input_data.index) * 0.1)],
                  input_data.iloc[-int(len(input_data.index) * 0.1):]
                  ]
        return d_list

    @staticmethod
    def _gen_model(seq: int, features: int, model_args: list):
        m = Sequential()
        layer_keys = bybit_run.LayerOptions.layer_dict
        ret_seq_layers = ['LSTM', 'GRU', 'RNN', 'SimpleRNN']
        for count, layer in enumerate(model_args):
            layer = list(layer.values())
            if layer[0] in ret_seq_layers:
                ret_seq = False
                if count < len(model_args) - 1 and model_args[count + 1]['layer'] in ret_seq_layers:
                    ret_seq = True
                if count == 0:
                    m.add(layer_keys[layer[0]](layer[1], input_shape=(seq, features), return_sequences=ret_seq))
                else:
                    m.add(layer_keys[layer[0]](layer[1], return_sequences=ret_seq))
            else:
                if count == 0:
                    m.add(layer_keys[layer[0]](layer[1], input_shape=(seq, features,)))
                else:
                    m.add(layer_keys[layer[0]](layer[1]))
            if count == len(model_args) - 1 and model_args[count - 1]['n'] != 1:
                m.add(layers.Dense(1, activation='tanh'))
        m.compile(loss=keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam())
        return m

    def _gen_model_name(self, model: keras.models, sym: str, b_s: int, fut_p: int, s_l: int, period: int, ma: list, ema: list,
                        drop_col: list):
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
        m_name = f'{sym}_{name}B{b_s}_F{fut_p}_P{period}_{int(time.time())}'
        m_args = f'S{s_l}-MA{ma}-EMAS{ema_s}-DSYM[{drop_col}]'
        self.verbose_print(f'\n{m_name}\n{m_args}\n', max_val=-1)
        return m_name, m_args

    @staticmethod
    def _make_model(seq: int, features: int):
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


if __name__ == '__main__':
    config = bybit_run.Config(bybit_run.CONFIG_FILE_PATH)
    api_key = config.key
    api_secret = config.secret
    if api_key is None or api_secret is None:
        exit(f'Please enter API Key and Secret in Config file at: {bybit_run.CONFIG_FILE_PATH}')
    Training(gui=None, parent=None, verbose=2, key=api_key, secret=api_secret).run(data_size=100,
                                                                                   data_period=1,
                                                                                   # in min [1, 3, 5, 15, 30, 60, 120, 240, 360, 720]
                                                                                   force_new_dataset=False,
                                                                                   future=[1],
                                                                                   targets=['BTC', 'ETH', 'XRP', 'EOS'],
                                                                                   target_outcome=Training.Regression,
                                                                                   drop_symbols=[],
                                                                                   moving_avg=[[5, 10]],  # nested list
                                                                                   e_moving_avg=[[4, 8]],  # nested list
                                                                                   sequence_length=[128, 256],
                                                                                   batch_size=[64, 128],
                                                                                   epochs=2,
                                                                                   early_stop_patience=5)
