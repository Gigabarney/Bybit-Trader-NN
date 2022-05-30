import math
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import time
from collections import deque
import numpy as np
import pandas as pd
from bin import bybitRun
from tqdm import tqdm
import matplotlib.pyplot as plt
from bin import data_handler
from sklearn.preprocessing import MinMaxScaler
import keras.losses
import keras.backend
from keras.models import Sequential
from keras import layers
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback

pd.options.mode.chained_assignment = None
try:
    import tensorflow as tf
except ModuleNotFoundError:
    pass


def _classify(current, future):
    """Classify target price. returns either 1 'higher' or 0 'lower'"""
    if float(future) > float(current):
        return 1
    else:
        return 0


def plot_data(data, plot_name=None):
    """Display and graph prediction data to control data"""
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
        plt.grid()
        plt.show()


class Training:
    """Retrice data, process data, and train a model"""
    Regression = 'r'
    Classification = 'c'
    OUTCOME_TYPES = [Regression, Classification]

    class LearningRateLogger(Callback):
        """Log learning rate for tensorboard callback"""

        def __init__(self):
            super().__init__()
            self._supports_tf_logs = True

        def on_epoch_end(self, epoch, logs=None):
            if logs is None or "learning_rate" in logs:
                return
            logs["learning_rate"] = self.model.optimizer.lr

    class UpdateProgressTraining(Callback):
        """Update GUI with model training progress"""

        def __init__(self, outer_instance, total_batches, total_epochs):
            """init UpdateProgressTraining"""
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
        """init training class"""
        self.parent = parent
        self.gui = gui
        self.verbose = verbose
        self.key = key
        self.secret = secret
        self.outcome_type = outcome_type

    def run(self, data_size: int, data_period: int, force_new_dataset: bool, future: list, targets: list, target_outcome: str, drop_symbols: list,
            moving_avg: list, e_moving_avg: list, madc: list, sequence_length: list, batch_size: list, epochs: int, early_stop_patience: int):
        """
        Called when 'networktraining.py' for manual training of models.
        Retrieves data from bybit, parses data, and trains model.
        Edit model under '_make_model' function to change layers types and number of nodes.
        """
        self._training_on()
        master_dataframe = data_handler.Constructor(self, key=self.key, secret=self.secret, data_s=data_size, data_p=data_period, threads=4,
                                                    force_new=force_new_dataset).get_data()
        back_up_data = master_dataframe.iloc[:int(len(master_dataframe.index) / 1)]
        for target in targets:
            if target in drop_symbols:
                continue
            for seq in sequence_length:
                for future_period in future:
                    for mavg in moving_avg:
                        for ema_span in e_moving_avg:
                            for _madc in madc:
                                df = back_up_data.copy()
                                t_x, t_y, v_x, v_y, test_x, test_y, features = self.build_data(df, target, seq, future_period, drop_symbols,
                                                                                               mavg, ema_span, _madc)
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
                                    checkpoint = ModelCheckpoint("{}/{}/{}__{}.h5".format(parent_dir, model_name, checkpoint_format, model_args),
                                                                 monitor=monitor, verbose=self.verbose, save_best_only=True, mode=mode)
                                    early_stop_callback = keras.callbacks.EarlyStopping(monitor=monitor, patience=early_stop_patience, mode=mode,
                                                                                        verbose=self.verbose, restore_best_weights=True, )
                                    plt.clf()
                                    self.verbose_print(f'\n{"*" * 20}\nSymbol:\t{target}\nSequence length:\t{seq}\nBatch Size: {batch}\n'
                                                       f'Future prediction period:\t{future_period}\nMoving Average(s):\t{mavg}\n'
                                                       f'Exponential Span:\t{ema_span}\n{"*" * 20}\n')
                                    active_model.fit(t_x, t_y, batch_size=batch, epochs=epochs, shuffle=True,
                                                     validation_data=(v_x, v_y), use_multiprocessing=True,
                                                     callbacks=[self.LearningRateLogger(),
                                                                checkpoint,
                                                                early_stop_callback,
                                                                TensorBoard(log_dir=f'log/{target}/{model_name}')])
                                    active_model.evaluate(test_x, test_y, batch_size=batch)
                                    prediction = active_model.predict(test_x)
                                    plot_name = f'{model_name}\n{model_args}'
                                    plotting_data = [[plot_name + ' *200*', test_y[:, 0][:200], prediction[:200]],
                                                     [plot_name, test_y[:, 0], prediction]]
                                    del active_model
                                    keras.backend.clear_session()
                                    plot_data(plotting_data)

    def train(self, client, data_args: dict, model_args: dict, force_new: bool, save_file=False):
        """
        Retrieves data from bybit, parses data, and trains model.
        Takes passed data arguments and model arguments to process data and construct model.
        Returns raw up parsed data and the constructed model
        """
        from keras.callbacks import LambdaCallback
        self._training_on()
        master_dataFrame = data_handler.Constructor(self, client=client, data_s=data_args['size'], data_p=data_args['period'], threads=4,
                                                    force_new=force_new, save_file=save_file).get_data()
        drop = [sym for sym in bybit_run.Symbols.all if sym != data_args['target']]
        raw_data = master_dataFrame.copy()
        tr_x, tr_y, v_x, v_y, te_x, te_y, features = self.build_data(master_dataFrame, target=data_args['target'], seq=data_args['seq'], drop=drop,
                                                                     future_p=data_args['future'], moving_avg=data_args['ma'], ema=data_args['ema'],
                                                                     macd=True)
        active_model = self._gen_model(data_args['seq'], features, model_args)
        model_name, model_name_args = self._gen_model_name(active_model, data_args['target'], data_args['batch'], data_args['future'],
                                                           data_args['seq'], data_args['period'], data_args['ma'], data_args['ema'], drop_col=drop)
        early_stop_callback = keras.callbacks.EarlyStopping(monitor="val_loss", patience=math.ceil(data_args['epochs'] / 3), verbose=self.verbose,
                                                            mode="min",
                                                            restore_best_weights=True, )
        self.verbose_print(f'\n{"*" * 20}\nSymbol:\t{data_args["target"]}\nSequence length:\t{data_args["seq"]}'
                           f'\nFuture prediction period:\t{data_args["future"]}\nMoving Average(s):\t{data_args["ma"]}'
                           f'\nExponential Span:\t{data_args["ema"]}\n{"*" * 20}\n', max_val=-1)
        active_model.fit(tr_x, tr_y, batch_size=data_args['batch'], epochs=data_args['epochs'], validation_data=(v_x, v_y),
                         use_multiprocessing=True, verbose=self.verbose - 1,
                         callbacks=[early_stop_callback, self.UpdateProgressTraining(self, total_batches=len(tr_x) // data_args['batch'],
                                                                                     total_epochs=data_args['epochs'])])
        keras.backend.clear_session()
        ret_model = bybit_run.Model(location=f'bin\\res\\models\\{model_name}__{model_name_args}.h5', model=active_model, seq=data_args["seq"],
                                    batch=data_args['batch'], future=data_args['future'], moving_avg=data_args["ma"], e_moving_avg=data_args["ema"],
                                    drop=drop, period=data_args['period'], layers=model_args)
        return raw_data, ret_model

    def _training_on(self):
        """Prints weather training will be with CPU or GPU, or tensorflow not installed"""
        try:
            if len(tf.config.list_physical_devices('GPU')) > 0:
                self.verbose_print('\tUSING GPU')
            else:
                self.verbose_print('\tUSING CPU')
        except ModuleNotFoundError:
            self.verbose_print('* Tensorflow Not Installed *')

    def verbose_print(self, text: str, max_val: int = 0, update: int = 0, reset: bool = False):
        """
        Print text to GUI if VERBOSE > 1 print to terminal if VERBOSE > 2 as long as MAX_VAL >= 0
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
                   ema: list, macd: bool, test_data_only: bool = False, prediction=False):
        """
        Drop un-needed columns, add moving averages, normalize and split data into sequences for training.
        Return sequences array and total number of featured in the data
        """
        if len(drop) > 0:
            for to_drop in drop:
                input_data.drop(columns=[col for col in input_data.columns if to_drop in col], inplace=True)
        input_data = self._add_avgs(input_data, moving_avg=moving_avg, ema=ema, madc=macd)
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

    def _add_avgs(self, input_data: pd.DataFrame, moving_avg, ema, madc):
        """Appends moving averages and exponential moving averages to end of dataframe"""
        for col in input_data.columns:
            if '_close' in col:
                if madc:
                    input_data[f'T1'] = input_data[col].ewm(span=12).mean() - input_data[col].ewm(span=26).mean()
                    input_data[f'DIF'] = input_data[f'T1'] - input_data[f'T1'].ewm(span=9).mean()
                    input_data[f'{col}_MACD'] = MinMaxScaler(feature_range=(-1, 1)).fit_transform(input_data[f'DIF'].values.reshape(-1, 1))
                    input_data.drop(columns=[f'T1', f'DIF'], inplace=True)
                if len(moving_avg) > 0:
                    self.verbose_print('Constructing MA & EMA...', max_val=1, reset=True)
                    for sub_mavg in moving_avg:
                        input_data[f'{col}_MA{sub_mavg}'] = input_data[col].rolling(window=sub_mavg).mean()
                if len(ema) > 0:
                    for e in ema:
                        input_data[f'{col}_EMA{e}'] = input_data[col].ewm(span=e, adjust=False).mean()
        input_data.dropna(inplace=True)
        self.verbose_print('', update=1, reset=True)
        return input_data

    def _set_targets(self, input_data: pd.DataFrame, target, future_p):
        """
        Prepares target data from passed df in groups of var FUTURE_PERIOD based on OUTCOME_TYPE
        if OUTCOME_TYPE == 'r' regression will be used else if OUTCOME_TYPE == 'c' classification will be used
        Returns dataframe with classification appended.
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
        """
        Replaces infinite values with nan then scales data.
        Drop rows containing nan values.
        Returns processed dataframe
        """
        self.verbose_print('Normalizing Data...', max_val=len(input_data.columns), reset=True)
        for col in tqdm(input_data.columns, unit=' column(s)', disable=not self.verbose > 1):
            if col not in ['time']:
                input_data.replace([np.inf, -np.inf], np.nan, inplace=True)
                input_data[col] = MinMaxScaler(feature_range=(-1, 1)).fit_transform(input_data[col].values.reshape(-1, 1))
            input_data[col] = pd.to_numeric(input_data[col], downcast='float')
            self.verbose_print('', update=1)
        input_data.dropna(inplace=True)
        return input_data

    def _sequence_data(self, input_data: pd.DataFrame, sequence_len: int):
        """
        Splits data into sequences for training then splits data and target.
        returns np.array of X (data) and y (target)
        """
        self.verbose_print(f'Constructing ~{len(input_data.values)} Sequences', max_val=len(input_data.values), reset=True)
        time.sleep(0.5)
        sequential_data = []
        seq = deque(maxlen=sequence_len)
        for i in tqdm(input_data.values, unit=' Sequences Constructed', disable=not self.verbose > 1):
            seq.append([n for n in i[:-1]])
            if len(seq) == sequence_len:
                sequential_data.append([np.array(seq), i[-1:]])
            self.verbose_print('', update=1)
        x, y = [], []
        self.verbose_print(f'Splitting {len(sequential_data)} Sequences', max_val=len(sequential_data), reset=True)
        for seq, target in tqdm(sequential_data, unit=' Target Seq Split', disable=not self.verbose > 1):
            x.append(seq)
            y.append(target)
            self.verbose_print('', update=1)
        return np.asarray(x), np.asarray(y)

    @staticmethod
    def _split_data(input_data):
        """
        Split data into training(70%), validation(20%), testing(10%)
        returns list of all dataframes
        """
        return [input_data.iloc[:int(len(input_data.index) * 0.7)],
                input_data.iloc[int(len(input_data.index) * 0.7):-int(len(input_data.index) * 0.1)],
                input_data.iloc[-int(len(input_data.index) * 0.1):]]

    @staticmethod
    def _gen_model(seq: int, features: int, model_args: list):
        """
        Generates model from passed sequence length, features, and model arguments.
        Returns compiled keras model
        """
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

    def _gen_model_name(self, model: keras.models, sym: str, b_s: int, fut_p: int, s_l: int, period: int, ma: list, ema: list, drop_col: list):
        """
        Generate file names based on model layout and data features trained on.
        Returns model name and model arguments string
        """
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
        m_name = f'{sym}_{name}B{b_s}_F{fut_p}_P{period}_{int(time.time())}'
        m_args = f"S{s_l}-MA{str(ma).replace(' ', '').replace(',', '_')}" \
                 f"-EMAS{str(ema).replace(' ', '').replace(',', '_')}" \
                 f"-DSYM[{' '.join(drop_col).strip().replace(' ', '_')}]"
        self.verbose_print(f'\n{m_name}\n{m_args}\n', max_val=-1)
        return m_name, m_args

    @staticmethod
    def _make_model(seq: int, features: int):
        """
        Used when training the model without the GUI.
        Model layers and optimizer and loss function can eb changed here.
        * NO BARING ON MODEL TRAINED THROUGH GUI *
        Returns compiled model.
        """
        m = Sequential([
            layers.LSTM(64, input_shape=(seq, features), return_sequences=False),
            layers.Dropout(0.15),
            layers.Dense(32),
            layers.Dense(32),
            layers.Dense(1, activation='tanh')
        ])
        m.summary()
        # lr = 0.0005
        lr = tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=0.001,  # start learning rate
                                                               first_decay_steps=1000,  # start learning rate decay after X steps
                                                               t_mul=2,  # each warm restart runs t_mul more steps
                                                               m_mul=0.8,  # on warm restart learning rate multiplied by m_mul
                                                               alpha=0.00001)  # min learning rate
        # OPTIMIZERS
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


if __name__ == '__main__':
    """
    When self started, used for training the model manually with custom parameters.
    """
    config = bybit_run.Config('res/config.yaml')
    api_key = config.key
    api_secret = config.secret
    if api_key is None or api_secret is None:
        exit(f'Please enter API Key and Secret in Config file at: "bin/res/config.yaml" ')
    Training(gui=None, parent=None, verbose=2, key=api_key,
             secret=api_secret).run(data_size=500,
                                    data_period=30,  # in min [1, 3, 5, 15, 30, 60, 120, 240, 360, 720]
                                    force_new_dataset=False,
                                    future=[1],
                                    targets=['BTC'],
                                    target_outcome=Training.Regression,
                                    drop_symbols=['ETH', 'XRP', 'EOS'],
                                    moving_avg=[[]],  # nested list
                                    e_moving_avg=[[]],  # nested list
                                    madc=[True],  # Moving Average Convergence Divergence
                                    sequence_length=[128, 256],
                                    batch_size=[64, 32],
                                    epochs=60,
                                    early_stop_patience=50)
