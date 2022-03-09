import os
import networkTraining

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # change tensorflow logs to not show.
from keras.models import load_model
from keras import layers
import gui
from bin import bybit_data_handler
import pandas as pd
from collections import deque
import random
import numpy as np
from bybit import bybit
import yaml
import threading
import os.path
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog, PhotoImage
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import ccxt

# ******************************
'''
Created by: Eric Kelm
    automatically trade cryptocurrencies from ByBit. with the help of a Neral network.
    set the amount to trade, when to cut your losses over a period of time.
    display your total balance as well as price and moving average of coin prices over different periods
    with a GUI.
'''


# ******************************

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


class Model:

    # data: pd.DataFrame, target: str, seq: int, future_p: int, drop: list, moving_avg: list, ema: list

    def __init__(self, **kwargs):
        self.location = kwargs.get('location')
        if self.location is not None and self.location.split('.')[-1] != 'h5':
            self.location = self.location + '.h5'
        self.model = kwargs.get('model')
        if self.location is not None and self.model is None:
            try:
                self.model = load_model(self.location)
            except (FileNotFoundError, OSError):
                print(bcolors.WARNING + f'WARNING: Model would not be found at: {self.location}')
                self.location = None
        self.seq = kwargs.get('seq')
        self.batch = kwargs.get('batch')
        self.future_p = kwargs.get('future')
        self.moving_avg = kwargs.get('moving_avg')
        self.e_moving_avg = kwargs.get('e_moving_avg')
        self.drop_symbols = kwargs.get('drop')

    def dump(self):
        if self.model is not None and self.location is not None:
            self.model.save(self.location)
        return {'location': self.location, 'seq': self.seq, 'batch': self.batch, 'future': self.future_p, 'moving_avg': self.moving_avg,
                'e_moving_avg': self.e_moving_avg, 'drop': self.drop_symbols}

    def move_model(self, n_location):
        import os
        os.rename(self.location, n_location)
        try:
            os.remove(self.location)
        except FileNotFoundError:
            pass
        self.location = n_location


class Config:

    def __init__(self, filepath=None):
        if filepath is None:
            c = self.default_config()
        else:
            c = self.load_file(filepath)
        self.c = c
        self.key = c.get('user').get('api_key')
        self.secret = c.get('user').get('api_secret')

    def load_file(self, filepath):
        """
        Loads config file. If the file is not found replace with a default one and passes values to GUI.
        If error occurs loading user config file more than 3 times it will be over written with the default config file.
            Old config file will be renamed to: CONFIG_FILE_NAME.yaml.old

        Exits after config files checked to be corrupt or Incorrect Structure.
        """
        global w, PAST_TRADED_CRYPTO
        global CONFIG_FILE_PATH

        try:
            with open(filepath, 'r') as file:
                conf = yaml.safe_load(file)
        except FileNotFoundError:
            print(bcolors.FAIL + f'ERROR: Config file not found at: {CONFIG_FILE_PATH}\nReplacing with default config file.')
            conf = self.default_config()
            self.write_config(conf)
        if not self._checksum_config(conf):
            print(bcolors.WARNING + 'WARNING: Config file has unexpected structure.' + bcolors.ENDC + ' This may cause errors or crashes.\n'
                                                                                                      f'Reset config file in the settings or modify it at: {CONFIG_FILE_PATH}')
        return conf

    def _checksum_config(self, conf: dict):
        """
        checksum_config(conf: dict)
        Compares check of config to default config for errors.

        Required Argument:
            config(dict):   Dictionary to checksum.

        :returns
            (str)   Hashed dictionary keys.
        """
        conf_sum = ''
        for key in self._rec_config(conf):
            conf_sum += key
        default_sum = ''
        for key in self._rec_config(self.default_config()):
            default_sum += key
        return hash(default_sum) == hash(conf_sum)

    def _rec_config(self, dictionary: dict):
        """
        rec_config(dictionary: dict)

        Recursively goes through provided dict to get all keys.

        Required Arguments:
            dictionary(dict): dict to be iterated over.
        :returns
            (str)   Keys from dictionary.
        """
        for key, value in dictionary.items():
            if type(value) is dict:
                yield key
                yield from self._rec_config(value)
            else:
                yield key

    def write_config(self, con=None):
        global MODELS
        if con is not None:
            c = con
        else:
            c = self.c
        for k, v in MODELS.items():
            if v is not None:
                c['settings'][f'{k.lower()}_model'] = v.dump()
            else:
                c['settings'][f'{k.lower()}_model'] = Model().dump()

        with open(CONFIG_FILE_PATH, 'w') as output:
            yaml.dump(c, output, sort_keys=False, default_flow_style=False)
        self.c = c
        print(bcolors.OKGREEN + 'SAVED')

    @staticmethod
    def default_config():
        return {'user': {'api_key': None, 'api_secret': None},
                'settings': {'def_coin': SYMBOLS[0], 'plot_display': TYPES_OF_PLOTS[0],
                             'balance_plot_period': list(BALANCE_PERIODS_DICT)[0],
                             'PMA': [0] * len(MOVING_AVG_DICT.values()),
                             'BMA': [0] * len(MOVING_AVG_BAL_DICT.values()),
                             'testing': True, 'trade_amount': 1, 'loss_amount': 5, 'loss_period': 10,
                             'btc_model': Model().dump(), 'eth_model': Model().dump(),
                             'xrp_model': Model().dump(), 'eos_model': Model().dump(),
                             'model_arguments': {'target': 'BTC', 'future': 1, 'size': 1000, 'period': 1,
                                                 'epochs': 10, 'seq': 128, 'batch': 64, 'ma': [5, 10], 'ema': [4, 8]},
                             'model_blueprint': [DEFAULT_BLUEPRINT_LAYER] * 3},
                'stats': {'total_profit': 0, 'last_cap_bal': 0, 'last_trade_profit': 0,
                          'total_prediction': 0, 'correct_prediction': 0}}  # default config file


class AsyncFetchBalance(threading.Thread):
    """
    AsyncFetchBalance(gui, key, secret, tag='fetch_balance')

    Thread to get balance from bybit using API info provided.
    Updates balance as well as throws: ccxt.errors.AuthenticationError if API key or secret is invalid.

    Prints duration thread run &
            value balance changed to console via: print_finish_thread()
    Prints Authentication to console.

    Required Arguments:
        gui (bybit_gui.Toplevel1): GUI interface.
        key (str): API key from bybit
        secret (str): API secret from bybit

    Optional Keyword Arguments:
        tag(str) = 'fetch_balance':    Unique tag to keep track of this thread in pool.
    """

    tag = 'fetch_balance'

    def __init__(self, gui: gui.Toplevel1, key: str, secret: str, tag='fetch_balance'):
        super().__init__()
        if self.tag != tag:
            self.tag = tag
        self.gui = gui
        self.key = key
        self.secret = secret
        self.st = 0
        # https://www.pythontutorial.net/tkinter/tkinter-thread/

    def run(self):
        global PAST_ACT_BAL, CONF, BALANCE_DF
        self.st = time.time()
        try:
            bybit_exchange = ccxt.bybit({'options': {'adjustForTimeDifference': True, },
                                         'apiKey': self.key, 'secret': self.secret, })
            crypto_balances = {s: bybit_exchange.fetch_balance().get(s)['free'] for s in SYMBOLS}
            balance_usd = {s: (crypto_balances[s] * bybit_exchange.fetch_ticker(f'{s}/USD')['close']) for s in SYMBOLS}
            s_bal = sum(balance_usd.values())
            if PAST_ACT_BAL == 0:
                change = sum(balance_usd.values()) - int(CONF.c['stats']['total_profit'])
                CONF.c['stats']['total_profit'] += s_bal - CONF.c['stats']['last_cap_bal']
            else:
                change = sum(balance_usd.values()) - PAST_ACT_BAL
                CONF.c['stats']['total_profit'] += s_bal - PAST_ACT_BAL
            PAST_ACT_BAL = s_bal
            CONF.c['stats']['last_cap_bal'] = s_bal
            data = [[int(time.time() - (time.time() % 60)), s_bal]]
            if len(BALANCE_DF.index) == 0:
                BALANCE_DF = pd.DataFrame(data, columns=['time', 'balance'])
            else:
                BALANCE_DF = pd.concat([BALANCE_DF, pd.DataFrame(data, columns=['time', 'balance'])])
            BALANCE_DF = BALANCE_DF[-525600:]

            for count, s in enumerate(SYMBOLS):
                self.gui.text_amount_arr[count]['text'] = crypto_balances[s]
                self.gui.text_usd_arr[count]['text'] = f'({conv_currency_str(balance_usd[s])} USD)'
            print_finish_thread(self.tag, self.st, [f'Change: ({conv_currency_str(change)})'])
        except ccxt.errors.AuthenticationError:
            print(bcolors.WARNING + '\n[!] Authentication Error\n\tCheck API key and secret and try again')
            for rb_label in self.gui.text_amount_arr:
                rb_label['text'] = ''
            for rb_label2 in self.gui.text_usd_arr:
                rb_label2['text'] = ''
            self.gui.text_amount_arr[0]['text'] = 'Authentication'
            self.gui.text_usd_arr[0]['text'] = 'Error'
            self.gui.text_amount_arr[2]['text'] = 'Check API Key'
            self.gui.text_usd_arr[2]['text'] = 'and Secret'


class AsyncDataHandler(threading.Thread):
    """
    AsyncDataHandler(gui, current_data, client, symbols, fetch_plot_delay=0.0, tag='data_handler')

    Thread to get price data from bybit servers using API info provided via client.
    After data collected will start: AsyncFetchPlot()  and if enabled will start: AsyncFetchPrediction()

    Prints duration thread run via: print_finish_thread() &
            avg api calls per second and minute via: bybit_data_handler.Handler()

    Required Arguments:
        gui(bybit_gui.Toplevel1):  GUI interface.
        current_data(pandas.Dataframe):    Crypto data pulled from ByBit.
        client(bybit):  Initialized bybit client with api key and secret.
        symbols(list):  List of crypto symbols to get from bybit api.

    Optional Keyword Arguments:
        fetch_plot_delay(float) = 0.0:  Start delay to be passed to: AsyncFetchPlot()
        tag(str) = 'data_handler':   Unique tag to keep track of this thread in pool.
    """

    tag = 'data_handler'

    def __init__(self, gui: gui.Toplevel1, current_data: pd.DataFrame, client: bybit, symbols: list,
                 fetch_plot_delay=0.0,
                 tag='data_handler'):
        super().__init__()
        if self.tag != tag:
            self.tag = tag
        self.handler = bybit_data_handler.Handler(current_data, client, symbols,
                                                  DATA_SIZE + list(MOVING_AVG_DICT)[-2])
        self.symbols = symbols
        self.gui = gui
        self.fetch_plot_delay = fetch_plot_delay
        self.st = 0
        # https://www.pythontutorial.net/tkinter/tkinter-thread/

    def run(self):
        global MAIN_DF, THREAD_POOL, trading_crypto, CONF, MODELS, TICK
        self.st = time.time()
        MAIN_DF = self.handler.get_data()
        print_finish_thread(self.tag, self.st)
        safe_start_thread(
            AsyncFetchPlot(self.gui, DATA_SIZE, trading_crypto.get(), MAIN_DF, delay=self.fetch_plot_delay))
        if int(CONF.c['settings']['trade_amount']) > 0 and IS_TRADING:
            print(bcolors.ENDC + '\nInit Trading ***')
            safe_start_thread(
                AsyncFetchPrediction(self.gui, trading_crypto.get(), CONF.c['settings']['trade_amount'],
                                     MAIN_DF, MODELS[self.symbols[trading_crypto.get()]]))


class AsyncFetchPlot(threading.Thread):
    """
    AsyncFetchPlot(gui, size, sel_sym, current_data, delay=0.0, tag='fetch_plot')

    Thread to create and display graphs for selected crypto price and past balance.

    Prints duration thread run & type of graph created via: print_finish_thread()

    Required Arguments:
        gui (bybit_gui.Toplevel1):  GUI interface.
        size (int): size or period of time in seconds of data to display.
        sel_sym(int):  position of symbol from SYMBOLS to be displayed.
        current_data(pandas.Dataframe): Crypto data pulled from ByBit.

    Optional Keyword Arguments:
        delay(float) = 0.0:  Delay before continues graphing data
        tag(str) = 'fetch_plot':   Unique tag to keep track of this thread in pool.
    """
    tag = 'fetch_plot'

    def __init__(self, gui: gui.Toplevel1, size: int, sel_sym: int, current_data: pd.DataFrame, delay=0.0,
                 tag='fetch_plot'):
        super().__init__()
        sns.set_theme()
        sns.set(rc={'figure.facecolor': "#E6E6E6"})  # good color: E6E6E6 | norm panel color: d9d9d9 | purp: E100E1
        pd.options.mode.chained_assignment = None
        if self.tag != tag:
            self.tag = tag
        self.delay = delay

        self.current_data = current_data
        self.gui = gui
        self.sel_sym = sel_sym
        if MODELS[SYMBOLS[self.sel_sym]].seq is not None and MODELS[SYMBOLS[self.sel_sym]].seq > 0:
            size = int(MODELS[SYMBOLS[self.sel_sym]].seq)
        self.size = size
        self.st = 0

    def run(self):
        time.sleep(self.delay)
        self.st = time.time()
        plot_size = {'x': 4, 'rely': 0.290, 'relheight': 0.650, 'width': 584}
        args = []
        to_rm = [child for child in self.gui.info_frame.winfo_children() if type(child) == tk.Canvas]
        if plot_type.get() == TYPES_OF_PLOTS[0]:
            dpi = 65
            place_moving_avg_btns(self.gui, MOVING_AVG_DICT)
            args.append(self.price_plot(plot_size, dpi))
        elif plot_type.get() == TYPES_OF_PLOTS[1]:
            dpi = 70
            place_moving_avg_btns(self.gui, MOVING_AVG_BAL_DICT)
            args.append(self.balance_plot(plot_size, dpi))
        elif plot_type.get() == TYPES_OF_PLOTS[2]:
            place_moving_avg_btns(self.gui, MOVING_AVG_DICT, m_offset=0)
            place_moving_avg_btns(self.gui, MOVING_AVG_BAL_DICT, m_offset=307)
            dpi = 62
            plot_size['relheight'] = (plot_size['relheight'] / 2)
            args.append(self.price_plot(plot_size, dpi))
            plot_size['rely'] = (plot_size['rely'] * 2.12)
            args.append(self.balance_plot(plot_size, dpi))
        [child.destroy() for child in to_rm]
        if len(args) == 0:
            args = None
        print_finish_thread(self.tag, self.st, args=args)

    def price_plot(self, plot_size, dpi):
        data = self.current_data[[f'{SYMBOLS[self.sel_sym]}_close']]
        data.rename(columns={f'{SYMBOLS[self.sel_sym]}_close': 'Price'}, inplace=True)
        data.sort_index(inplace=True)
        # data['Buy'] = sum([self.full_data[f'BTC_Buy_{c}'] for c in range(5)])
        # data['Sell'] = sum([self.full_data[f'BTC_Sell_{c}'] for c in range(5)])
        data['Time'] = self.current_data.index
        data.reset_index(drop=True, inplace=True)
        h_key = None
        for key, val in MOVING_AVG_DICT.items():
            if val.get() == 1:
                if key != 'EMA':
                    h_key = key
                    data[f'MA{key}'] = data['Price'].rolling(key, win_type='triang').mean()
                elif h_key is not None:
                    data[f'EMA{h_key}'] = data[f'MA{h_key}'].ewm(span=4).mean()

        data = data[-self.size:]
        data = pd.melt(data, 'Time')
        figure = plt.Figure(dpi=dpi)
        ax = figure.subplots()
        ax.yaxis.set_major_formatter(FuncFormatter(lambda z, pos: conv_currency_str(z)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda z, pos: None))
        ax.set(xlabel=f'$Time$', ylabel='Price')
        p = sns.lineplot(data=data, y='value', x='Time', hue='variable', ax=ax, legend=True)
        p.set_title(f'{SYMBOLS[self.sel_sym]} Price\nOver {self.size} min')
        ax.legend(loc='upper left')
        # ax2 = ax.twinx()
        # ax.yaxis.set_major_formatter(FuncFormatter(lambda z, pos: f'${z}'))
        # ax2.set(ylabel=f'Contracts ({m}s)')
        # sns.lineplot(data=t_data, y='value', x='Time', hue='variable', ax=ax2, palette=['g', 'r'], alpha=0.6, legend=True)
        # ax2.legend(loc='upper left')
        self.gui.canvas_open_close = FigureCanvasTkAgg(figure, self.gui.info_frame)
        self.gui.canvas_open_close.draw()
        self.gui.canvas_open_close.get_tk_widget().place(x=plot_size['x'], rely=plot_size['rely'],
                                                         relheight=plot_size['relheight'], width=plot_size['width'])
        if 'Loading' in self.gui.btn_start_stop['text']:
            self.gui.loading_plot_label.destroy()
            self.gui.btn_start_stop['text'] = START_STOP_TEXT['start']
            self.gui.btn_start_stop.state(['!disabled'])
        return 'PRICE'

    def balance_plot(self, plot_size, dpi):
        global BALANCE_DF, bal_period
        while AsyncFetchBalance.tag in [tag.tag for tag in THREAD_POOL]:
            time.sleep(0.3)
        data = BALANCE_DF[525600 - BALANCE_PERIODS_DICT.get(bal_period.get()):]
        data.rename(columns={'balance': 'Balance', 'time': 'Time'}, inplace=True)
        for key, val in MOVING_AVG_BAL_DICT.items():
            if val.get() == 1:
                data[f'MA{key}'] = data['Balance'].rolling(key, win_type='triang').mean()
        data = pd.melt(data, 'Time')
        figure_2 = plt.Figure(dpi=dpi)
        ax_2 = figure_2.subplots()
        ax_2.yaxis.set_major_formatter(FuncFormatter(lambda z, pos: conv_currency_str(z)))
        ax_2.xaxis.set_major_formatter(FuncFormatter(lambda z, pos: None))
        ax_2.set(xlabel=f'$Time$', ylabel='Balance')

        p = sns.lineplot(data=data, y='value', x='Time', hue='variable', ax=ax_2, legend=True)
        ax_2.legend(loc='upper left')
        p.set_title(f'Total Balance for last {bal_period.get()}')
        self.gui.canvas_open_close_2 = FigureCanvasTkAgg(figure_2, self.gui.info_frame)
        self.gui.canvas_open_close_2.draw()
        self.gui.canvas_open_close_2.get_tk_widget().place(x=plot_size['x'], rely=plot_size['rely'],
                                                           relheight=plot_size['relheight'], width=plot_size['width'])
        if 'Loading' in self.gui.btn_start_stop['text']:
            self.gui.loading_plot_label.destroy()
            self.gui.btn_start_stop['text'] = START_STOP_TEXT['start']
            self.gui.btn_start_stop.state(['!disabled'])
        return f'BALANCE [{bal_period.get()} period]'


class AsyncFetchPrediction(threading.Thread):
    """
    AsyncFetchPrediction(gui, sel_sym, current_data, model, tag='prediction')

    Thread to run model (or random value if testing) and place orders (or simulate orders if testing=True)
    Process data as via: bybit_data_handler.preprocess_data()
    Will sell previous held crypto if selection changes.
    Updates number of correct predictions.

    Prints duration thread run & status of trade ie. 'OK buy order | OK sell order | INSUFFICIENT FUNDS'
            via: print_finish_thread()
    Prints if real trade has been attempted.

    Required Arguments:
        gui(bybit_gui.Toplevel1):  GUI interface.
        sel_sym(int):  position of symbol from SYMBOLS to be traded.
        current_data(pandas.Dataframe): current unprocessed data from bybit
        model(dist):  dict holding model (keras.model) as well as data manipulation arguments.

    Optional Keyword Arguments:
        tag(str) = 'prediction':   Unique tag to keep track of this thread in pool.
    """
    tag = 'prediction'

    def __init__(self, gui: gui.Toplevel1, sel_sym: int, current_data: pd.DataFrame, model: Model, tag='prediction'):
        super().__init__()
        if self.tag != tag:
            self.tag = tag
        # self.exchange = EXCHANGE
        self.gui = gui
        self.model = model
        self.sel_sym = sel_sym
        self.size = model.seq
        self.current_data = current_data
        self.st = 0

    def run(self):
        global PREDICTION, CONF, HAS_POSITION, PAST_TRADED_CRYPTO
        self.st = time.time()
        status = 'OK'
        HAS_POSITION = {'position': 0, 'qty': 0}
        no_pos, buy_pos, sell_pos = (0, 1, -1)
        current_price = self.current_data[f'{SYMBOLS[self.sel_sym]}_close'].values[-1]

        processed_data = bybit_data_handler.preprocess_data(self.current_data, self.model.get_args())

        if ((PREDICTION == 1 and (current_price > self.current_data[f'{SYMBOLS[self.sel_sym]}_close'].values[-2])) or
                (PREDICTION == 0 and (
                        self.current_data[f'{SYMBOLS[self.sel_sym]}_close'].values[-2] > current_price))):
            CONF.c['stats']['correct_prediction'] += 1
        CONF.c['stats']['correct_prediction'] += 1

        if bool(CONF.c['settings']['testing']) or HARD_TESTING_LOCK or self.model.model is None:
            CONF.c['settings']['testing'] = True
            print(bcolors.ENDC + '*** Testing Prediction ***')
            PREDICTION = random.randint(0, 1)  # 0 == price down, 1 == price up
        else:
            pass
            # PREDICTION = int(self.model.predict([processed_data])[0][0])

        qty = int(CONF.c['settings']['trade_amount']) + (HAS_POSITION['qty'] * abs(HAS_POSITION['position']))
        # qty account for difference in previous trade and pulling out of last position as well as following new position
        if PAST_TRADED_CRYPTO != self.sel_sym and self._testing_check():
            if HAS_POSITION == buy_pos:  # Nullify position on past coin if coin selected differs from past
                # self.exchange.create_market_sell_order(symbol=f"{SYMBOLS[PAST_TRADED_CRYPTO]}/USD", amount=HAS_POSITION['qty'])  # SELL
                print(bcolors.FAIL + '\t[]*!*!* REAL TRADE ATTEMPTED *!*!*[]')
                print(bcolors.ENDC + 'ORDER: SELL :: CHANGED CRYPTO')
                HAS_POSITION['position'] = no_pos
            elif HAS_POSITION == sell_pos:  # Nullify position on past coin if coin selected differs from past
                # self.exchange.create_market_buy_order(symbol=f"{SYMBOLS[PAST_TRADED_CRYPTO]}/USD", amount=HAS_POSITION['qty'])  # BUY
                print(bcolors.FAIL + '\t[]*!*!* REAL TRADE ATTEMPTED *!*!*[]')
                print(bcolors.ENDC + 'ORDER: BUY :: CHANGED CRYPTO')
                HAS_POSITION['position'] = no_pos

        order = None
        HAS_POSITION['qty'] = qty
        try:
            if PREDICTION == 1 and HAS_POSITION['position'] != buy_pos:  # BUY ACTION as long as current held poition isnt buy
                HAS_POSITION['position'] = buy_pos
                if self._testing_check():
                    # order = self.exchange.create_market_buy_order(symbol=f"{self.sel_sym}/USD", amount=qty)
                    print(bcolors.FAIL + '\t[]*!*!* REAL TRADE ATTEMPTED *!*!*[]')
                status = 'OK buy order'
            elif PREDICTION == 0 and HAS_POSITION['position'] != sell_pos:
                HAS_POSITION['position'] = sell_pos
                if self._testing_check():
                    # order = self.exchange.create_market_sell_order(symbol=f"{self.sel_sym}/USD", amount=qty)
                    print(bcolors.FAIL + '\t[]*!*!* REAL TRADE ATTEMPTED *!*!*[]')
                status = 'OK sell order'
        except ccxt.errors.InsufficientFunds:
            global IS_TRADING
            IS_TRADING = False
            HAS_POSITION = no_pos
            self.gui.entry_trade_amount.delete(0, 'end')
            trigger_error_bar(['INSUFFICIENT FUNDS', 'Enter Lower Trade Amount'], 5)
            status = 'INSUFFICIENT FUNDS'
        print_finish_thread(self.tag, self.st, [status])

    @staticmethod
    def _testing_check():
        return not bool(CONF.c['settings']['testing']) and not HARD_TESTING_LOCK


class AsyncTestModel(threading.Thread):
    """
    AsyncTestModel(model_path, symbol, current_data, tag='test_model')

    Thread to load the model via 'model_path', test provided model on current data for errors,
        and assign provided model to model dist.

    Prints duration thread run & status of model ie. (PASSED | FAILED)

    Required Arguments:
        model_path(str):    path to model on disk.
        symbol(str):    Crypto symbol to assign the model if passes.
        current_data(pandas.DataFrame): Dataframe of current operating data.

    Optional Keyword Arguments:
        tag(str) = 'fetch_plot':   Unique tag to keep track of this thread in pool.
    """
    tag = 'test_model'

    def __init__(self, data: pd.DataFrame, testing_model: Model, target: str, verbose: int = 0, tag='test_model'):
        super().__init__()
        if self.tag != tag:
            self.tag = tag
        self.data = data
        self.verbose = verbose
        self.target = target
        self.model = testing_model
        self._return = 0
        self.st = 0

    def run(self):
        self.st = time.time()
        self._return = self.test_model()
        print_finish_thread(self.tag, self.st, ['eval:', round(self._return, 6)])

    def join(self, *args):
        threading.Thread.join(self, *args)
        return self._return

    def test_model(self):
        import networkTraining as nt
        test_x, test_y = nt.build_data(self.data, self.target, self.model.seq, self.model.future_p, self.model.drop_symbols,
                                       self.model.moving_avg, self.model.e_moving_avg, self.verbose, test_only=True)

        return self.model.model.evaluate(test_x, test_y, self.model.batch, verbose=self.verbose)


class AsyncTrainModel(threading.Thread):
    """
    AsyncTrainModel(model_path, symbol, current_data, tag='train_model')

    Thread to load the model via 'model_path', test provided model on current data for errors,
        and assign provided model to model dist.

    Prints duration thread run & status of model ie. (PASSED | FAILED)

    Required Arguments:
        model_path(str):    path to model on disk.
        symbol(str):    Crypto symbol to assign the model if passes.
        current_data(pandas.DataFrame): Dataframe of current operating data.

    Optional Keyword Arguments:
        tag(str) = 'fetch_plot':   Unique tag to keep track of this thread in pool.
        """
    tag = 'train_model'

    def __init__(self, gui: gui.Toplevel1, data_args: dict, model_args: list, verbose: int, force_new: bool, tag='train_model'):
        super().__init__()
        if self.tag != tag:
            self.tag = tag
        self.gui = gui
        self.data_args = data_args
        self.model_args = model_args
        self.verbose = verbose
        self.force_new = force_new
        self.st = 0
        self._kill = False

    def run(self):
        self.st = time.time()
        import networkTraining
        global MODELS, CONF
        nt = networkTraining
        raw_data, new_model = nt.train(self, self.gui, self.data_args, self.model_args, self.verbose, self.force_new,
                                       CONF.key, CONF.secret)
        old_model = MODELS[self.data_args.get('target')]
        if old_model.model is None:
            # assign new model to model
            MODELS[self.data_args.get('target')] = new_model
            mm = MODELS[self.data_args.get('target')]
            nt.update_progress('New Model Saved.', max_value=1, update=1, reset=True)
            status = 'No model found. New model saved.'
        else:
            nt.VERBOSE = 0
            nt.update_progress('Evaluating.', max_value=2, reset=True)

            eval1 = AsyncTestModel(raw_data.copy(), new_model, self.data_args.get('target')).test_model()
            eval2 = AsyncTestModel(raw_data.copy(), old_model, self.data_args.get('target')).test_model()
            dif = eval1
            status = f'Old model kept.\tEval split: {eval2 - eval1}'
            if eval1 < eval2:
                MODELS[self.data_args.get('target')] = new_model
                status = f'New model used.\tEval split: {eval1 - eval2}'
        time.sleep(2)
        nt.update_progress(reset=True)
        self.gui.btn_start_train['text'] = 'Start Training'
        self.gui.prog_bar_train['value'] = 0
        self.gui.label_train_status['text'] = ''
        print_finish_thread(self.tag, self.st, [status])

    def kill(self):
        self._kill = True

    def check_kill(self):
        if self._kill:
            self.gui.btn_start_train['text'] = 'Start Training'
            self.gui.prog_bar_train['value'] = 0
            self.gui.label_train_status['text'] = ''
            print(bcolors.WARNING + "WARNING: Thread: 'AsyncTrainModel'" + bcolors.FAIL + " Killed.")
            exit(2)


class AsyncBuildBalance(threading.Thread):
    """
    AsyncBuildBalance(tag='build_balance')

    Thread for the initial building of the balance dataframe.

    prints duration thread run & status of trade ie. 'OK | File Not Found' via: print_finish_thread()

    Optional Keyword Arguments:
        tag(str) = 'build_balance':   Unique tag to keep track of this thread in pool.
    """
    tag = 'build_balance'

    def __init__(self, tag='build_balance'):
        super().__init__()
        if self.tag != tag:
            self.tag = tag
        self.st = 0

    def run(self):
        global BALANCE_DF
        self.st = time.time()
        current_time_m = (time.time() - (time.time() % 60))
        try:
            status = 'OK'
            BALANCE_DF = pd.read_csv(BALANCE_DF_PATH, index_col=0)
            BALANCE_DF.sort_values('time', inplace=True)
            max_time = max(BALANCE_DF['time'])
            last_bal = BALANCE_DF['balance'].iloc[-1]
            data = []
            for v in range(1, int((current_time_m - max_time) / 60)):
                data.append([int(v * 60 + max_time), last_bal])
            BALANCE_DF = pd.concat([BALANCE_DF, pd.DataFrame(data, columns=['time', 'balance'])], ignore_index=True)
        except FileNotFoundError:
            status = 'File Not Found'
            pass
        except KeyError:
            status = 'Key Error'
            pass
        if len(BALANCE_DF.index) > 525600:  # length of 1 year in min
            BALANCE_DF = BALANCE_DF[:525600]
        elif len(BALANCE_DF.index) < 525600:
            def_dat_gen = ((int(current_time_m - (60 * off)), 0) for off in range(525600 - len(BALANCE_DF.index)))
            BALANCE_DF = pd.concat([pd.DataFrame(def_dat_gen, columns=['time', 'balance']), BALANCE_DF], axis=0,
                                   ignore_index=True)
        BALANCE_DF.sort_values('time', inplace=True)
        BALANCE_DF.reset_index(drop=True, inplace=True)
        print_finish_thread(self.tag, self.st, [status])


def print_finish_thread(tag: str, start_time: float, args: list = None):
    """
    print_finish_thread(tag, start_time, args = None)

    Print when thread finishes along with arguments and duration.

    Required Arguments:
        tag(str):   Tag from threat or preferred string.
        start_time(float):  Float of unix time started to calculate total time of thread.

    Optional Keyword Arguments:
        args(list) = None: A list of strings to be added to output.
    """
    to_app = ''
    if args is not None:
        for arg in args:
            to_app += str(arg) + '\t'
    print(bcolors.ENDC + f'{time.ctime()}\t' + bcolors.OKGREEN +
          f'Finished: {tag}:\t\t' + bcolors.ENDC + f'{to_app}Duration: {round(time.time() - start_time, 4)}sec')


def set_tk_var():
    """Set tkinter module global variables"""
    global trading_crypto
    trading_crypto = tk.IntVar()
    global MOVING_AVG_DICT
    MOVING_AVG_DICT = {key: tk.IntVar() for key in MOVING_AVG_DICT}
    global MOVING_AVG_BAL_DICT
    MOVING_AVG_BAL_DICT = {key: tk.IntVar() for key in MOVING_AVG_BAL_DICT}
    global plot_type
    plot_type = tk.StringVar()
    global bal_period
    bal_period = tk.StringVar()
    global train_symbol
    train_symbol = tk.StringVar()


def init(top, gui, *args, **kwargs):
    """
    Call load_config().
    Initialize vars (CLIENT, IND_IMAGE).
    Call root loops (refresh(), trading_loop()).
    Update GUI values with config values
    """
    global w, top_level, root, CLIENT, IND_IMAGE, BALANCE_DF, CONF, MODELS
    w = gui
    CONF = Config(filepath=CONFIG_FILE_PATH)
    CLIENT = bybit(test=False, api_key=CONF.c['user']['api_key'],
                   api_secret=CONF.c['user']['api_secret'])
    IND_IMAGE = {'green': PhotoImage(file='bin/res/green_ind_round.png'), 'red': PhotoImage(file='bin/res/red_ind_round.png'),
                 'yellow': PhotoImage(file='bin/res/yellow_ind_round.png')}
    for s in SYMBOLS:
        m_args = CONF.c['settings'][f'{s.lower()}_model']
        MODELS[s] = Model(location=m_args['location'], size=m_args['seq'],
                          moving_avg=m_args['moving_avg'], e_moving_avg=m_args['e_moving_avg'], drop=m_args['drop'])
    BALANCE_DF = pd.DataFrame()
    safe_start_thread(AsyncBuildBalance())
    top_level = top
    root = top
    set_gui_fields(gui)
    refresh()
    trading_loop()
    api_change()


def set_gui_fields(gui: gui.Toplevel1):
    global CONF
    gui.text_prediction['text'] = '-'
    gui.entry_trade_amount.delete(0, 'end')
    gui.entry_trade_amount.insert(0, CONF.c['settings']['trade_amount'])
    gui.entry_loss.delete(0, 'end')
    gui.entry_loss.insert(0, CONF.c['settings']['loss_amount'])
    gui.entry_loss_period.delete(0, 'end')
    gui.entry_loss_period.insert(0, CONF.c['settings']['loss_period'])

    if CONF.c['settings']['plot_display'] in TYPES_OF_PLOTS:
        gui.combo_box_plot_type.set(CONF.c['settings']['plot_display'])
    else:
        gui.combo_box_plot_type.set(TYPES_OF_PLOTS[0])

    if CONF.c['settings']['balance_plot_period'] in list(BALANCE_PERIODS_DICT):
        gui.combo_box_bal_period.set(CONF.c['settings']['balance_plot_period'])
    else:
        gui.combo_box_bal_period.set(list(BALANCE_PERIODS_DICT)[0])
    [gui.ma_btns[k].select() for k, v in enumerate(list(CONF.c['settings']['PMA'])) if v == 1]

    # {'target': 'BTC', 'future': 1, 'size': 1000, 'period': 1,
    #                    'epochs': 10, 'seq': 128, 'batch': 64, 'ma': [5, 10], 'ema': [4, 8]},

    model_args = CONF.c['settings']['model_arguments']
    gui.combo_box_target.current(SYMBOLS.index(model_args.get('target')))
    gui.entry_future_p.delete(0, 'end')
    gui.entry_future_p.insert(0, model_args.get('future'))
    gui.entry_data_size.delete(0, 'end')
    gui.entry_data_size.insert(0, model_args.get('size'))
    gui.entry_data_period.delete(0, 'end')
    gui.entry_data_period.insert(0, model_args.get('period'))
    gui.entry_epoch.delete(0, 'end')
    gui.entry_epoch.insert(0, model_args.get('epochs'))
    gui.entry_seq_len.delete(0, 'end')
    gui.entry_seq_len.insert(0, model_args.get('seq'))
    gui.entry_batch_size.delete(0, 'end')
    gui.entry_batch_size.insert(0, model_args.get('batch'))
    gui.entry_ma.delete(0, 'end')
    gui.entry_ma.insert(0, str(model_args.get('ma'))[1:-1].replace(' ', ''))
    gui.entry_ema.delete(0, 'end')
    gui.entry_ema.insert(0, str(model_args.get('ema'))[1:-1].replace(' ', ''))
    for layer in CONF.c['settings']['model_blueprint']:
        model_struct_row(layer)


def destroy_window():
    """Destroys window"""
    # Function which closes the window.
    global top_level
    top_level.destroy()
    top_level = None


def window_close():
    """Save file(s) and ends threads. Then calls destroy_window()"""
    global IS_QUITTING, BALANCE_DF_PATH, w
    IS_QUITTING = True
    CONF.write_config()
    BALANCE_DF.to_csv(BALANCE_DF_PATH)
    try:
        w.canvas_open_close.destroy()
    except AttributeError:
        pass
    for t in THREAD_POOL:
        t.join()
    destroy_window()


def conv_currency_str(value: float):
    """
    Format currency to proper decimal values depending on value

    Required Arguments:
        value(float):   Value to convert to currency string.

    :returns
        (str)   Converted to currency ie. $123.45
    """
    if value is None:
        return '0'
    elif value < 9.9999:
        return "${:,.4f}".format(value)
    elif value < 99.999:
        return "${:,.3f}".format(value)
    elif value > 99999999:
        return "${:,.0f}".format(value)
    else:
        return "${:,.2f}".format(value)


def obfuscate_api_info(val: str):
    """
    Hides API value passed only showing last 8 characters

    Required Arguments:
        val(str):   Value to be obfuscated.

    :returns
        (str)   Obfuscated value.
    """
    global w
    if val is None or len(val) < 8:
        return 'ERROR'
    else:
        return f'{"*" * (len(val) - 8) + val[-8:]}'


def safe_start_thread(t: callable):
    """
    Safely start a thread.
    Check is thread in running by checking if passed thread tag is in thread pool.
    Prints if thread started.

    Required Arguments:
        t(callable):    The thread callable to be checked and started.
    """
    global THREAD_POOL
    if AsyncBuildBalance.tag in [tag.tag for tag in THREAD_POOL]:
        print(bcolors.ENDC + f'Still Building Balance. Thread: "{t.tag}"' + bcolors.WARNING + ' NOT Started.')
        return
    elif t.tag in [tag.tag for tag in THREAD_POOL]:
        print(bcolors.ENDC + f'Thread: "{t.tag}" Running.' + bcolors.WARNING + ' NOT Started.')
        return
    t.start()
    THREAD_POOL.append(t)
    print(bcolors.ENDC + f'Started Thread: "{t.tag}"' + bcolors.OKGREEN + ' Safely.')


def api_change():
    """
    To be called after changing the api key or secret.
    Updates values for GUI as well as CLIENT and fetches new balance.
    """
    global MAIN_DF, CLIENT, w
    w.text_api_key['text'] = obfuscate_api_info(CONF.key)
    w.text_api_secret['text'] = obfuscate_api_info(CONF.secret)
    if obfuscate_api_info(CONF.key) == 'ERROR' or obfuscate_api_info(
            CONF.secret) == 'ERROR':
        trigger_error_bar(lines=['ERROR:  API Key/Secret', 'Try Updating API Key or Secret'], duration=10)

    safe_start_thread(
        AsyncFetchBalance(w, key=CONF.key, secret=CONF.secret))
    CLIENT = bybit(test=False, api_key=CONF.c['user']['api_key'],
                   api_secret=CONF.c['user']['api_secret'])
    safe_start_thread(AsyncDataHandler(w, MAIN_DF, CLIENT, SYMBOLS))


def trading_loop():
    """Main loop to call AsyncFetchBalance() thread and AsyncDataHandler() thread."""
    global TRADING_COUNT, w
    if TRADING_COUNT >= 1:
        safe_start_thread(
            AsyncFetchBalance(w, key=CONF.key, secret=CONF.secret))
        safe_start_thread(AsyncDataHandler(w, MAIN_DF, CLIENT, SYMBOLS, fetch_plot_delay=0.5))
    TRADING_COUNT += 1
    if not IS_QUITTING:
        root.after(int((60 - (time.time() % 60)) * 1000), lambda: trading_loop())


def refresh():
    """Main loop to refresh GUI elements and values."""
    global THREAD_POOL, trading_crypto, REFRESH_COUNT, w, TRADING_COUNT
    # refresh GUI every 1 seconds with updated values
    w.time_to_next_up['text'] = f'Update in: {int(60 - (time.time() % 60))}sec'
    if 'Loading' in w.btn_start_stop['text']:
        l_c = w.btn_start_stop['text'][w.btn_start_stop['text'].find('.'):]
        if len(l_c) >= 3:
            w.loading_plot_label['text'] = START_STOP_TEXT['loading']
            w.btn_start_stop['text'] = START_STOP_TEXT['loading']
        else:
            n_val = f'{" "}{w.btn_start_stop["text"]}{"."}'
            w.loading_plot_label['text'] = n_val
            w.btn_start_stop['text'] = n_val
    w.text_testing['text'] = str(bool(CONF.c['settings']['testing']))
    w.text_pred_total['text'] = CONF.c['stats']['total_prediction']
    w.text_pred_correct['text'] = CONF.c['stats']['correct_prediction']
    w.text_total_profit['text'] = conv_currency_str(CONF.c['stats']['total_profit'])
    w.text_last_trade_profit['text'] = conv_currency_str(CONF.c['stats']['last_trade_profit'])
    if int(CONF.c['stats']['last_trade_profit']) > 0:  # green text color
        w.text_last_trade_profit['foreground'] = '#25b100'
    elif int(CONF.c['stats']['last_trade_profit']) < 0:  # red text color
        w.text_last_trade_profit['foreground'] = '#b10000'
    else:  # black text color
        w.text_last_trade_profit['foreground'] = '#000000'

    if TRADING_COUNT >= 2 and IS_TRADING:
        if REFRESH_COUNT % 2 == 0:
            w.label_act_dact['text'] = 'Active'
            w.img_indicator_light.itemconfig(w.ind_img_container, image=IND_IMAGE['green'])
        else:
            w.label_act_dact['text'] = ''
            w.img_indicator_light.itemconfig(w.ind_img_container, image=None)
    elif IS_TRADING:
        w.label_act_dact['text'] = 'Initializing'
        w.img_indicator_light.itemconfig(w.ind_img_container, image=IND_IMAGE['yellow'])
    else:
        w.label_act_dact['text'] = 'Inactive'
        w.img_indicator_light.itemconfig(w.ind_img_container, image=IND_IMAGE['red'])

    try:
        w.text_pred_acc[
            'text'] = f'{(int(CONF.c["stats"]["correct_prediction"]) / int(CONF.c["stats"]["total_prediction"])) * 100}%'
    except ZeroDivisionError:
        w.text_pred_acc['text'] = '-%'

    if len(THREAD_POOL) > 0:
        THREAD_POOL = [t for t in THREAD_POOL if t.is_alive()]
    REFRESH_COUNT += 1
    if not IS_QUITTING:
        root.after(1000, lambda: refresh())


def trigger_error_bar(lines, duration, close=False):
    """
    When called triggers an error bar with contents of lines.

    Required Arguments:
        lines(list[str]):   What text to display with error.
        duration(float):    How long the error will display for in seconds.

    Optional Keyword Arguments:
        close=False:    To end error loop after displaying error (function use only.)

    """
    global w
    if close:
        w.text_error_bar['text'] = ''
        return
    f_text = ''
    for l in lines:
        f_text += f'**{l}**\n'
    print(bcolors.FAIL + f_text)
    w.text_error_bar['text'] = str(f_text)
    w.text_error_bar['foreground'] = '#880808'
    root.after(int(duration * 1000), lambda: trigger_error_bar(lines, duration, close=True))


def reset_stats():
    """Reset stats to '0' and trade settings to default."""
    global PAST_TRADE_VAL, MAX_LOSS_VAL, PAST_BAL_LOSS, w
    CONF.c['settings']['trade_amount'] = 1
    w.entry_trade_amount.delete(0, 'end')
    w.entry_trade_amount.insert(0, CONF.c['settings']['trade_amount'])
    PAST_TRADE_VAL = CONF.c['settings']['trade_amount']
    CONF.c['settings']['loss_amount'] = 5
    w.entry_loss.delete(0, 'end')
    w.entry_loss.insert(0, CONF.c['settings']['loss_amount'])
    MAX_LOSS_VAL = CONF.c['settings']['loss_amount']
    CONF.c['settings']['loss_period'] = 10
    w.entry_loss_period.delete(0, 'end')
    w.entry_loss_period.insert(0, CONF.c['settings']['loss_period'])
    t_p_b_l = PAST_BAL_LOSS
    PAST_BAL_LOSS = deque(maxlen=CONF.c['settings']['loss_period'])
    PAST_BAL_LOSS.append(t_p_b_l)
    CONF.c['stats']['total_prediction'] = 0
    CONF.c['stats']['correct_prediction'] = 0
    CONF.c['stats']['total_profit'] = 0
    CONF.c['stats']['last_trade_profit'] = 0
    CONF.write_config()


def save_settings():
    """Call when save settings button is clicked. Saves API key & secret to config then calls api_change()."""
    global w
    t_api_key = w.entry_api_key.get().strip()
    current_key = CONF.c['user']['api_key']
    t_api_secret = w.entry_api_secret.get().strip()
    current_secret = CONF.c['user']['api_secret']

    if not t_api_key or not t_api_secret or current_key == t_api_key or current_secret == t_api_secret:
        pass
    else:
        CONF.c['user']['api_key'] = t_api_key
        CONF.key = t_api_key
        CONF.c['user']['api_secret'] = t_api_secret
        CONF.secret = t_api_secret
        if os.path.exists(BALANCE_DF_PATH):
            os.remove(BALANCE_DF_PATH)
    safe_start_thread(AsyncBuildBalance())
    w.entry_api_key.delete(0, 'end')
    w.entry_api_secret.delete(0, 'end')
    CONF.write_config()
    api_change()


def testing_toggle():
    """When test button pressed Toggles test mode."""
    CONF.c['settings']['testing'] = not bool(CONF.c['settings']['testing'])


def place_moving_avg_btns(gui: gui.Toplevel1, var_set: dict, m_offset=0):
    """
    Called from GUI creation.
    Makes moving average buttons for graphs and removes previous buttons.

    Required Arguments:
        gui(bybit_gui.Toplevel1):   GUI Interface
        var_set(dict):  Dictionary of moving avr buttons.

    Optional Keyword Arguments:
        m_offset=0: Main offset if button set is placed next to another.
    """
    offset = 85 + m_offset
    if m_offset == 0 and len(gui.ma_btns) > 0:
        [btn.place_forget() for btn in gui.ma_btns]
        gui.ma_btns = []
    if m_offset != 0:
        gui.label_break_line.place(x=offset - 26, rely=0.25, height=20, width=16)
        gui.label_break_line.configure(activebackground="#f9f9f9", activeforeground="black", anchor='c',
                                       background="#d9d9d9",
                                       justify='center', disabledforeground="#a3a3a3",
                                       font="-family {Segoe UI} -size 16", foreground="#000000",
                                       highlightbackground="#d9d9d9", highlightcolor="black", text=' | ')
    else:
        try:
            gui.label_break_line.place_forget()
        except AttributeError:
            pass
    for count, (ma_key, ma_var) in enumerate(var_set.items()):
        text = f'MA{ma_key}'
        if 'EMA' == ma_key:
            text = f'{ma_key}'
        x_pos = offset + (55 * count)
        chk_btn_ma = tk.Checkbutton(gui.info_frame)
        chk_btn_ma.place(x=x_pos, rely=0.25, height=20, width=55)
        chk_btn_ma.configure(command=lambda: moving_avg_check_btn(), text=text, variable=ma_var, background="#d9d9d9")
        gui.ma_btns.append(chk_btn_ma)
    if list(MOVING_AVG_BAL_DICT) == list(var_set):
        gui.combo_box_bal_period.place(x=6 + offset + (len(var_set) * 55), rely=0.25, height=20, width=80)
    else:
        gui.combo_box_bal_period.place_forget()


def moving_avg_check_btn():
    """
    Checks if moving ave buttons pressed and calls AsyncFetchPlot() as long as time
    is greater than 15sec from refreshing.

    Required Arguments:
        event:  On click event.
    """
    CONF.c['settings']['PMA'] = [x.get() for x in MOVING_AVG_DICT.values()]
    CONF.c['settings']['BMA'] = [x.get() for x in MOVING_AVG_DICT.values()]
    if (60 - (time.time() % 60)) > 15:
        t = AsyncFetchPlot(w, DATA_SIZE, trading_crypto.get(), MAIN_DF, delay=4)
        safe_start_thread(t)


def plot_combo_box(event):
    """
    Checks if combo box for plot type is changed and calls AsyncFetchPlot() as long as time
    is greater than 15sec from refreshing.

    Required Arguments:
        event:  On click event.
    """
    CONF.c['settings']['plot_display'] = plot_type.get()
    if (60 - (time.time() % 60)) > 15:
        t = AsyncFetchPlot(w, DATA_SIZE, trading_crypto.get(), MAIN_DF, delay=4)
        safe_start_thread(t)
    w.info_frame.focus()


def bal_period_combo_box(event):
    """
    Checks if combo box for balance period is changed and calls AsyncFetchPlot() as long as time
    is greater than 15sec from refreshing.

    Required Arguments:
        event:  On click event.
    """
    CONF.c['settings']['plot_display'] = bal_period.get()
    if (60 - (time.time() % 60)) > 15:
        t = AsyncFetchPlot(w, DATA_SIZE, trading_crypto.get(), MAIN_DF, delay=4)
        safe_start_thread(t)
    w.info_frame.focus()


def load_model_btn(sym: str):
    """
    Opens OS file selector to locate model of type (hdf5, h5)
    and calls: AsyncTestModel() to make sure model is functional.
    No baring on accuracy of the model.

    Prints 'canceled' to console if no file is selected.

    Required Arguments:
        sym(str):   String of crypto symbol in witch the model is for.
    """
    global w
    w.TNotebook1.select(2)
    w.combo_box_target.current(SYMBOLS.index(sym))


def get_model_info(symbol: str, model, return_str=True):
    """
    Get neural network model info by calling summary on the model.
    If model passed is None, return.

    Required Arguments:
        symbol(str):    Symbol of crypto model to summarize.

    Optional Keyword Arguments:
        return_str(bool) = True:    If True will return a string containing parsed m.summary().
                                        Else open a dialog window displaying m.summary().
    :returns
        (str): If return_str is set to True else (None)
    """
    summary = []
    if model is None:
        print(bcolors.WARNING + f'WARNING: Model: {symbol} not loaded')
        return
    model.summary(line_length=100, print_fn=lambda x: summary.append(x))
    if not return_str:
        gui.new_model_info_window(gui.ModelInfoWindow, summary=summary)
    else:
        out = []
        longest = 0
        for line in summary:
            line = line.replace('_', '').replace('=', '')
            line = ' '.join(line.split())
            if not line or 'Layer (type) Output' in line:
                continue
            longest = max([longest, len(line)])
            out.append(line)

        border = '*' * (longest // 2)
        out.insert(0, border)
        out.append(border)
        return out


def update_trading_values():
    """Updates trading values to config file."""
    global PAST_TRADE_VAL, MAX_LOSS_VAL
    PAST_TRADE_VAL = CONF.c['settings']['trade_amount']
    CONF.c['settings']['trade_amount'] = w.entry_trade_amount.get()
    MAX_LOSS_VAL = CONF.c['settings']['loss_amount']
    CONF.c['settings']['loss_amount'] = w.entry_loss.get()
    CONF.c['settings']['loss_period'] = w.entry_loss_period.get()


def start_btn():
    """
    Called when start button pressed.
    Resets default prediction.
    Updates button position so display btn_update_trades and changes text on start button to
    show stop.
    """
    global IS_TRADING, TRADING_COUNT, PREDICTION
    PREDICTION = -1
    IS_TRADING = not IS_TRADING
    if IS_TRADING:
        w.btn_update_trades.place(x=140, rely=0.870, height=60, width=130)
        w.btn_start_stop['text'] = START_STOP_TEXT['stop']
        w.btn_start_stop.place(x=10, rely=0.870, height=60, width=130)
    else:
        w.btn_start_stop['text'] = START_STOP_TEXT['start']
        w.btn_update_trades.place(x=10, rely=0.800, height=30, width=0)
        w.btn_start_stop.place(x=10, rely=0.870, height=60, width=260)


def model_struct_row(layer=None):
    global w
    if layer is not None:
        y_pos = 32 + (5 * w.model_layer_lst['count']) + (20 * w.model_layer_lst['count'])
        if w.model_layer_lst['count'] >= 7:
            return
        w.model_layer_lst['count'] += 1

    else:
        if w.model_layer_lst['count'] > 0:
            w.model_layer_lst['count'] -= 1
        rm = w.model_layer_lst['list'].pop()
        for child in rm:
            child.destroy()
        return

    combo_box = ttk.Combobox(w.FrameTraining2)
    combo_box.place(x=5, y=y_pos, height=19, width=70)
    combo_box.configure(background="#d9d9d9", font="-family {Segoe UI} -size 9")
    combo_box['values'] = list(LAYER_OPTIONS)
    combo_box.current(list(LAYER_OPTIONS).index(layer['layer']))
    combo_box['state'] = 'readonly'

    n_label = ttk.Label(w.FrameTraining2)
    n_label.place(x=77, y=y_pos - 1, height=19, width=51)
    n_label.configure(background="#d9d9d9", foreground="#000000", font="-family {Segoe UI} -size 9", relief="flat",
                      anchor='w', justify='left', text='Neurons:')

    n_entry = tk.Entry(w.FrameTraining2)
    n_entry.place(x=129, y=y_pos, height=19, width=32)
    n_entry.configure(takefocus="", cursor="ibeam", font="-family {Segoe UI} -size 10")
    n_entry.insert(0, layer['n'])

    d_label = ttk.Label(w.FrameTraining2)
    d_label.place(x=162, y=y_pos - 1, height=19, width=50)
    d_label.configure(background="#d9d9d9", foreground="#000000", font="-family {Segoe UI} -size 9", relief="flat",
                      anchor='w', justify='left', text='Dropout:')

    d_entry = tk.Entry(w.FrameTraining2)
    d_entry.place(x=213, y=y_pos, height=19, width=28)
    d_entry.configure(takefocus="", cursor="ibeam", font="-family {Segoe UI} -size 10")
    d_entry.insert(0, layer['drop'])

    w.model_layer_lst['list'].append([combo_box, n_entry, d_entry, n_label, d_label])


def train_model():
    if 'stop' in w.btn_start_train['text'].lower():
        time.sleep(0.5)
        for th in THREAD_POOL:
            if th.tag == AsyncTrainModel.tag:
                th.kill()
        return
    data_args = {}
    args_key = ['target', 'future', 'size', 'period', 'epochs', 'seq', 'batch', 'ma', 'ema']
    for count, entry in enumerate(w.training_fields):
        try:
            if count == 0:
                if entry.get().strip() in SYMBOLS:
                    data_args[args_key[count]] = entry.get().strip()
                else:
                    print(bcolors.FAIL + f'ERROR: TargetError: Selected target not an option: {entry.get().strip()}')
                    return
            elif count > len(w.training_fields) - 3:
                data_args[args_key[count]] = [int(v.strip()) for v in entry.get().strip().split(',')]
            else:
                data_args[args_key[count]] = int(entry.get().strip())
        except ValueError:
            print(bcolors.FAIL + f'ERROR: ValueError: Model training field: #{count} value: {entry.get()} type: {type(entry.get())}')
            return

    for k in args_key[1:-2]:
        if data_args[k] <= 0:
            print(bcolors.WARNING + f'WARNING: Argument: "{k}" value: {data_args[k]} is invalid. Value must be greater than 0.')
            return

    model_args = []
    for layer in w.model_layer_lst['list']:
        try:
            sub_l = []
            for count, ll in enumerate(layer[:3]):
                if count == 0:
                    if ll.get().strip() in list(LAYER_OPTIONS):
                        sub_l.append(ll.get().strip())
                    else:
                        print(bcolors.FAIL + f'ERROR: LayerTypeError: layer type not acceptable: {ll.get().strip()}')
                        return
                else:
                    try:
                        sub_l.append(int(ll.get().strip()))
                    except ValueError:
                        f = float(ll.get().strip())
                        if f > 1:
                            f = 1.0
                        sub_l.append(f)
            model_args.append(sub_l)
        except ValueError:
            print(bcolors.FAIL + f'ERROR: ValueError: value: {ll.get()} model layer: {str(layer)}')
            return

    if len(model_args) <= 0:
        print(bcolors.WARNING + f'WARNING: Model Layout Invalid. Size: {len(model_args)} must be greater than 0.')
        return

    verbose = 1 if ('selected' in w.checkbutton_verbose.state() or 'alternate' in w.checkbutton_verbose.state()) else 0
    force_new = True if ('selected' in w.checkbutton_new_data.state() or 'alternate' in w.checkbutton_new_data.state()) else False

    w.btn_start_train['text'] = 'Stop Training'
    safe_start_thread(AsyncTrainModel(w, data_args=data_args, model_args=model_args, verbose=verbose, force_new=force_new))


# GLOBAL VARS

# none
CLIENT = None  # global var for bybit client

# bool
IS_TRADING = False  # if se tot trading mode or idle
HARD_TESTING_LOCK = True  # software lock to prevent accidental trading wail testing
IS_QUITTING = False  # when set to true to start the shutdown process.

# ints
DATA_SIZE = 120  # size of data for model. Changed when switching models.
PREDICTION = -1  # current buy or sell choice. -1 = invalid, 0 = sell, 1 = buy
TRADING_COUNT, REFRESH_COUNT, TICK = (0, 0, 0)  # counts for gui ticks
PAST_ACT_BAL = 0  # previous account balance to check value change in balance
MAX_LOSS_VAL = 0  # max value that can be lost before auto stop. retrived from config.
PAST_TRADED_CRYPTO = 0  # hold last traded crypto. in case user changed wail crypto is still being held.

# strings
CONFIG_FILE_PATH = 'bin/res/config.yaml'  # config file location
BALANCE_DF_PATH = 'bin/res/b_df'  # user balance dataframe
VERSION = '0.0.2'  # version number

# lists
SYMBOLS = ['BTC', 'ETH', 'XRP', 'EOS']  # cryptocurrency that can be traded.
TYPES_OF_PLOTS = ['Price', 'Balance', 'Both']  # types of graphs that can be displayed
THREAD_POOL = []  # pool of all active threads
LAYER_OPTIONS = {'LSTM': layers.LSTM, 'GRU': layers.GRU, 'RNN': layers.RNN, 'SimpleRNN': layers.SimpleRNN,
                 'Dense': layers.Dense}  # options for layer types
MODELS = {s: None for s in SYMBOLS}  # list of models

# dictionary's
HAS_POSITION = {'position': 0, 'qty': 0}  # hold last traded position and quantity
ENT_PRICE = {'price': 0, 'side': 'none'}  # hold price entered in trade and buy or sell side
BALANCE_PERIODS_DICT = {'10 min': 10, '60 Min': 60, '1 Day': 1440, '10 Day': 14400, '1 Month': 43800, '3 Month': 131400,
                        '6 Month': 262800, '1 Year': 525600}  # periods in graph that can be displayed
MOVING_AVG_DICT = {5: 0, 10: 0, 20: 0, 30: 0, 'EMA': 0}  # moving average options for price = 0 off, 1 on
MOVING_AVG_BAL_DICT = {10: 0, 30: 0}  # moving option averages for price 0 off, 1 on
DEFAULT_BLUEPRINT_LAYER = {'layer': list(LAYER_OPTIONS)[4], 'n': 16, 'drop': 0.0}  # default layer type

IND_IMAGE = {'green': None, 'red': None, 'yellow': None}  # indicator images
START_STOP_TEXT = {'start': 'Start Trading', 'stop': 'Stop Trading',
                   'loading': 'Loading'}  # different text options for button

# misc
PAST_BAL_LOSS = deque(maxlen=2)  # hold past 2 balance changes
MAIN_DF = pd.DataFrame()  # dataframe holding the coin price data
BALANCE_DF = pd.DataFrame()  # dataframe holding the user balance price data
CONF = Config()  # config file being used. set to default until initialized

if __name__ == '__main__':
    exit('START BY RUNNING "gui.py"')
