import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # change tensorflow logs to not show.
from keras.models import load_model
import gui
from bin import bybit_data_handler
import pandas as pd
from collections import deque
import random
import numpy as np
from bybit import bybit
import yaml
from threading import Thread
import os.path
import tkinter as tk
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
CONFIG_FILE_PATH = 'res/config.yaml'  # config file location
BALANCE_DF_PATH = 'res/b_df'
VERSION = '0.0.2'  # version number

# lists
SYMBOLS = ['BTC', 'ETH', 'XRP', 'EOS']  # cryptocurrency that can be traded.
TYPES_OF_PLOTS = ['Price', 'Balance', 'Both']  # types of graphs that can be displayed
THREAD_POOL = []  # pool of all active threads

# dictionary's
HAS_POSITION = {'position': 0, 'qty': 0}  # hold last traded position and quantity
ENT_PRICE = {'price': 0, 'side': 'none'}  # hold price entered in trade and buy or sell side
BALANCE_PERIODS_DICT = {'10 min': 10, '60 Min': 60, '1 Day': 1440, '10 Day': 14400, '1 Month': 43800, '3 Month': 131400,
                        '6 Month': 262800, '1 Year': 525600}  # periods in graph that can be displayed
MOVING_AVG_DICT = {5: 0, 10: 0, 20: 0, 30: 0, 'EMA': 0}  # moving average options for price = 0 off, 1 on
MOVING_AVG_BAL_DICT = {10: 0, 30: 0}  # moving option averages for price 0 off, 1 on

MODELS = {s: {'model': None, 'args': {'data_size': None, 'ma': [], 'ema_span': [], 'symbols': []}}
          for s in SYMBOLS}  # loaded model in dict with symbol as key and data manipulation arguments

IND_IMAGE = {'green': None, 'red': None, 'yellow': None}  # indicator images
START_STOP_TEXT = {'start': 'Start Trading', 'stop': 'Stop Trading',
                   'loading': 'Loading'}  # diffrent text options for button
DEFAULT_CONFIG = {'config': {'user': {'api_key': None, 'api_secret': None},
                             'settings': {'def_coin': SYMBOLS[0], 'plot_display': TYPES_OF_PLOTS[0],
                                          'balance_plot_period': list(BALANCE_PERIODS_DICT.keys())[0],
                                          'PMA': [0] * len(MOVING_AVG_DICT.values()),
                                          'BMA': [0] * len(MOVING_AVG_BAL_DICT.values()),
                                          'testing': True, 'trade_amount': 1, 'loss_amount': 5, 'loss_period': 10,
                                          'btc_model_path': None,
                                          'eth_model_path': None, 'xrp_model_path': None, 'eos_model_path': None,
                                          'st_count': 0, 'st_count_re': 3},
                             'stats': {'total_profit': 0, 'last_cap_bal': 0, 'last_trade_profit': 0,
                                       'total_prediction': 0, 'correct_prediction': 0}}}  # default config file

# misc
PAST_BAL_LOSS = deque(maxlen=2)  # hold past 2 balance changes
MAIN_DF = pd.DataFrame()  # dataframe holding the coin price data
BALANCE_DF = pd.DataFrame()  # dataframe holding the user balance price data
CONF = DEFAULT_CONFIG  # config file being used. set to default until initialized


class AsyncFetchBalance(Thread):
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

    def __init__(self, gui: bybit_gui.Toplevel1, key: str, secret: str, tag='fetch_balance'):
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
                change = sum(balance_usd.values()) - int(CONF['config']['stats']['total_profit'])
                CONF['config']['stats']['total_profit'] += s_bal - CONF['config']['stats']['last_cap_bal']
            else:
                change = sum(balance_usd.values()) - PAST_ACT_BAL
                CONF['config']['stats']['total_profit'] += s_bal - PAST_ACT_BAL
            PAST_ACT_BAL = s_bal
            CONF['config']['stats']['last_cap_bal'] = s_bal
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
            print('\n[!] Authentication Error\n\tCheck API key and secret and try again')
            for rb_label in self.gui.text_amount_arr:
                rb_label['text'] = ''
            for rb_label2 in self.gui.text_usd_arr:
                rb_label2['text'] = ''
            self.gui.text_amount_arr[0]['text'] = 'Authentication'
            self.gui.text_usd_arr[0]['text'] = 'Error'
            self.gui.text_amount_arr[2]['text'] = 'Check API Key'
            self.gui.text_usd_arr[2]['text'] = 'and Secret'
            vv = ''


class AsyncDataHandler(Thread):
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

    def __init__(self, gui: bybit_gui.Toplevel1, current_data: pd.DataFrame, client: bybit, symbols: list,
                 fetch_plot_delay=0.0,
                 tag='data_handler'):
        super().__init__()
        if self.tag != tag:
            self.tag = tag
        self.handler = bybit_data_handler.Handler(current_data, client, symbols,
                                                  DATA_SIZE + list(MOVING_AVG_DICT.keys())[-2])
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
        if int(CONF['config']['settings']['trade_amount']) > 0 and IS_TRADING:
            print('\nInit Trading ***')
            safe_start_thread(
                AsyncFetchPrediction(self.gui, trading_crypto.get(), CONF['config']['settings']['trade_amount'],
                                     MAIN_DF, MODELS[self.symbols[trading_crypto.get()]]))


class AsyncFetchPlot(Thread):
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

    def __init__(self, gui: bybit_gui.Toplevel1, size: int, sel_sym: int, current_data: pd.DataFrame, delay=0.0,
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
        if MODELS[SYMBOLS[self.sel_sym]]['args']['data_size'] is not None and MODELS[SYMBOLS[self.sel_sym]]['args']['data_size'] > 0:
            size = int(MODELS[SYMBOLS[self.sel_sym]]['args']['data_size'])
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


class AsyncFetchPrediction(Thread):
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

    def __init__(self, gui: bybit_gui.Toplevel1, sel_sym: int, current_data: pd.DataFrame, model: dict, tag='prediction'):
        super().__init__()
        if self.tag != tag:
            self.tag = tag
        # self.exchange = EXCHANGE
        self.gui = gui
        self.model = model
        self.sel_sym = sel_sym
        if MODELS[self.sel_sym]['data_size'] is not None and MODELS[self.sel_sym]['data_size'] > 0:
            self.size = int(MODELS[self.sel_sym]['data_size'])
        else:
            self.size = DATA_SIZE
        self.current_data = current_data
        self.st = 0

    def run(self):
        global PREDICTION, CONF, HAS_POSITION, PAST_TRADED_CRYPTO
        self.st = time.time()
        status = 'OK'
        HAS_POSITION = {'position': 0, 'qty': 0}
        no_pos, buy_pos, sell_pos = (0, 1, -1)
        current_price = self.current_data[f'{SYMBOLS[self.sel_sym]}_close'].values[-1]

        processed_data = bybit_data_handler.preprocess_data(self.current_data, SYMBOLS, self.model['args'])
        self.model = self.model['model']

        if ((PREDICTION == 1 and (current_price > self.current_data[f'{SYMBOLS[self.sel_sym]}_close'].values[-2])) or
                (PREDICTION == 0 and (
                        self.current_data[f'{SYMBOLS[self.sel_sym]}_close'].values[-2] > current_price))):
            CONF['config']['stats']['correct_prediction'] += 1
        CONF['config']['stats']['correct_prediction'] += 1

        if bool(CONF['config']['settings']['testing']) or HARD_TESTING_LOCK or self.model is None:
            CONF['config']['settings']['testing'] = True
            print('*** Testing Prediction ***')
            PREDICTION = random.randint(0, 1)  # 0 == price down, 1 == price up
        else:
            PREDICTION = int(self.model.predict([processed_data])[0][0])

        qty = int(CONF['config']['settings']['trade_amount']) + (HAS_POSITION['qty'] * abs(HAS_POSITION['position']))
        # qty account for difference in previous trade and pulling out of last position as well as following new position
        if PAST_TRADED_CRYPTO != self.sel_sym and self._testing_check():
            if HAS_POSITION == buy_pos:  # Nullify position on past coin if coin selected differs from past
                # self.exchange.create_market_sell_order(symbol=f"{SYMBOLS[PAST_TRADED_CRYPTO]}/USD", amount=HAS_POSITION['qty'])  # SELL
                print('\t[]*!*!* REAL TRADE ATTEMPTED *!*!*[]')
                print('ORDER: SELL :: CHANGED CRYPTO')
                HAS_POSITION['position'] = no_pos
            elif HAS_POSITION == sell_pos:  # Nullify position on past coin if coin selected differs from past
                # self.exchange.create_market_buy_order(symbol=f"{SYMBOLS[PAST_TRADED_CRYPTO]}/USD", amount=HAS_POSITION['qty'])  # BUY
                print('\t[]*!*!* REAL TRADE ATTEMPTED *!*!*[]')
                print('ORDER: BUY :: CHANGED CRYPTO')
                HAS_POSITION['position'] = no_pos

        order = None
        HAS_POSITION['qty'] = qty
        try:
            if PREDICTION == 1 and HAS_POSITION['position'] != buy_pos:  # BUY ACTION as long as current held poition isnt buy
                HAS_POSITION['position'] = buy_pos
                if self._testing_check():
                    # order = self.exchange.create_market_buy_order(symbol=f"{self.sel_sym}/USD", amount=qty)
                    print('\t[]*!*!* REAL TRADE ATTEMPTED *!*!*[]')
                status = 'OK buy order'
            elif PREDICTION == 0 and HAS_POSITION['position'] != sell_pos:
                HAS_POSITION['position'] = sell_pos
                if self._testing_check():
                    # order = self.exchange.create_market_sell_order(symbol=f"{self.sel_sym}/USD", amount=qty)
                    print('\t[]*!*!* REAL TRADE ATTEMPTED *!*!*[]')
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
        return not bool(CONF['config']['settings']['testing']) and not HARD_TESTING_LOCK


class AsyncTestModel(Thread):
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

    def __init__(self, model_path: str, symbol: str, tag='test_model'):
        super().__init__()
        if self.tag != tag:
            self.tag = tag
        self.model_path = model_path
        self.symbol = symbol
        # self.current_data = current_data
        self.st = 0

    def run(self):
        try_count, max_count = 0, 22
        while len(MAIN_DF) == 0:
            if try_count > max_count:
                print('ERROR: Data not loaded. Could not test model.')
                exit()
            try_count += 1
            if try_count % 5 == 0:
                print(f'Thread: {self.tag} held. Waiting for data.  {try_count}/{max_count}')
            time.sleep(3)

        self.st = time.time()
        status_lst = ['PASSED']
        process_time = 0
        try:
            mod = load_model(self.model_path)
            summary = get_model_info(self.symbol, mod, return_str=True)
            for s in summary:
                print(s)
            model_args = self.model_path.split('/')[-1].split('__')[-1].split('-')
            model_args = {'size': int(model_args[0][1:]),
                          'ma': [int(v) for v in model_args[1].split('[')[-1].replace(']', '').split('_')],
                          'ema': [int(v) for v in model_args[2].split('[')[-1].replace(']', '').split('_')],
                          'drop_sym': [v for v in model_args[3].split('[')[-1].split(']')[0].split('_')]}
            process_time = time.time()
            data = bybit_data_handler.preprocess_data(MAIN_DF, data_mod_args=model_args)
            process_time = time.time() - process_time
            x = 1
            data = np.asarray(data)
            print(data.shape)
            try:
                pre = mod.predict(data)
                x = 'hello'
            except Exception as e:
                print(e)
        except (AttributeError, ValueError, TypeError) as e:
            status_lst = [str(type(e)), 'FAILED']
        finally:
            if process_time != 0:
                status_lst.append(f'data process time: {round(process_time,4)}')
            print_finish_thread(self.tag, self.st, status_lst)


class AsyncBuildBalance(Thread):
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
            to_app += arg + '\t'
    print(f'Finished: {tag}:\t\t{to_app}Duration: {round(time.time() - start_time, 4)}sec')


def set_tk_var():
    """Set tkinter module global variables"""
    global trading_crypto
    trading_crypto = tk.IntVar()
    global api_key_entry
    api_key_entry = tk.StringVar()
    global api_secret_entry
    api_secret_entry = tk.StringVar()
    global MOVING_AVG_DICT
    MOVING_AVG_DICT = {key: tk.IntVar() for key in MOVING_AVG_DICT.keys()}
    global MOVING_AVG_BAL_DICT
    MOVING_AVG_BAL_DICT = {key: tk.IntVar() for key in MOVING_AVG_BAL_DICT.keys()}
    global plot_type
    plot_type = tk.StringVar()
    global bal_period
    bal_period = tk.StringVar()


def init(top, gui, *args, **kwargs):
    """
    Call load_config().
    Initialize vars (CLIENT, IND_IMAGE).
    Call root loops (refresh(), trading_loop()).
    Update GUI values with config values
    """
    global w, top_level, root, CLIENT, IND_IMAGE, BALANCE_DF
    w = gui
    load_config()
    CLIENT = bybit(test=False, api_key=CONF['config']['user']['api_key'],
                   api_secret=CONF['config']['user']['api_secret'])
    BALANCE_DF = pd.DataFrame()
    safe_start_thread(AsyncBuildBalance())
    top_level = top
    root = top
    IND_IMAGE = {'green': PhotoImage(file='res/green_ind_round.png'), 'red': PhotoImage(file='res/red_ind_round.png'),
                 'yellow': PhotoImage(file='res/yellow_ind_round.png')}
    refresh()
    trading_loop()
    api_change()
    w.text_prediction['text'] = '-'
    gui.entry_trade_amount.delete(0, 'end')
    gui.entry_trade_amount.insert(0, CONF['config']['settings']['trade_amount'])
    gui.entry_loss.delete(0, 'end')
    gui.entry_loss.insert(0, CONF['config']['settings']['loss_amount'])
    gui.entry_loss_period.delete(0, 'end')
    gui.entry_loss_period.insert(0, CONF['config']['settings']['loss_period'])
    if CONF['config']['settings']['plot_display'] in TYPES_OF_PLOTS:
        gui.combo_box_plot_type.set(CONF['config']['settings']['plot_display'])
    else:
        gui.combo_box_plot_type.set(TYPES_OF_PLOTS[0])

    if CONF['config']['settings']['balance_plot_period'] in list(BALANCE_PERIODS_DICT.keys()):
        gui.combo_box_bal_period.set(CONF['config']['settings']['balance_plot_period'])
    else:
        gui.combo_box_bal_period.set(list(BALANCE_PERIODS_DICT.keys())[0])
    [gui.ma_btns[k].select() for k, v in enumerate(list(CONF['config']['settings']['PMA'])) if v == 1]


def destroy_window():
    """Destroys window"""
    # Function which closes the window.
    global top_level
    top_level.destroy()
    top_level = None


def window_close():
    """Writes to config file and ends threads. Then calls destroy_window()"""
    global IS_QUITTING, w
    write_config(CONF)
    IS_QUITTING = True
    try:
        w.canvas_open_close.destroy()
    except AttributeError:
        pass
    for t in THREAD_POOL:
        t.join()
    destroy_window()


def quitting_text():
    """Displays quitting text in appropriate areas"""
    global w
    l_c = w.loading_plot_label['text']
    if ('Quitting' in l_c) and len(l_c[l_c.find('.'):]) <= 3:
        w.loading_plot_label['text'] = f'{" "}{w.btn_start_stop["text"]}{"."}'
    else:
        w.loading_plot_label['text'] = 'Quitting'
    root.after(1000, lambda: quitting_text())


def rec_config(dictionary: dict):
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
            yield from rec_config(value)
        else:
            yield key


def checksum_config(conf: dict):
    """
    checksum_config(conf: dict)
    Compares check of config to default config for errors.

    Required Argument:
        config(dict):   Dictionary to checksum.

    :returns
        (str)   Hashed dictionary keys.
    """
    conf_sum = ''
    for key in rec_config(conf):
        conf_sum += key
    default_sum = ''
    for key in rec_config(DEFAULT_CONFIG):
        default_sum += key
    return hash(default_sum) == hash(conf_sum)


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
        print(f'Still Building Balance. Thread: "{t.tag}" NOT Started.')
        return
    elif t.tag in [tag.tag for tag in THREAD_POOL]:
        print(f'Thread: "{t.tag}" Running. NOT Started.')
        return
    t.start()
    THREAD_POOL.append(t)
    print(f'Started Thread: "{t.tag}" Safely.')


def load_config():
    f"""
    Loads config file. If the file is not found replace with a default one and passes values to GUI.
    If error occurs loading user config file more than 3 times it will be over written with the default config file.
        Old config file will be renamed to: CONFIG_FILE_NAME.yaml.old

    Exits after config files checked to be corrupt or Incorrect Structure.
    """
    global w, PAST_TRADED_CRYPTO
    # load config file. if file not found replace with default.
    # if error loading config more than 3 times temp old config and replace with default config then
    # load default values into GUI
    global CONFIG_FILE_PATH, DEFAULT_CONFIG, CONF
    try:
        with open(CONFIG_FILE_PATH, 'r') as file:
            CONF = yaml.safe_load(file)
    except FileNotFoundError:
        print(f'ERROR:: Config file not found at: {CONFIG_FILE_PATH}\nReplacing with default config file.')
        write_config(DEFAULT_CONFIG)
        CONF = DEFAULT_CONFIG
    if not checksum_config(CONF):
        out_text = ''
        # check if config file and default config have the same keys in them via checksum
        settings_conf = CONF["config"]["settings"]
        out_text += f'ERROR:: Config file corrupt or Incorrect Structure please DELETE or REVERT file at: {CONFIG_FILE_PATH}\n' \
                    f'Expected:{DEFAULT_CONFIG.keys()}\n' \
                    f'Received:{CONF.keys()}'
        out_text += f'\nAttempts {settings_conf["st_count"]}/{settings_conf["st_count_re"]}'
        CONF['config']['settings']['st_count'] += 1
        if settings_conf["st_count"] >= settings_conf["st_count_re"]:
            config_rename = f'{CONFIG_FILE_PATH}.OLD'
            print(f'Writing Default Config File. Old file saved as {config_rename}')
            os.rename(CONFIG_FILE_PATH, config_rename)
            write_config(DEFAULT_CONFIG)
        else:
            write_config(CONF)
            exit(
                f'ERROR:: Config file corrupt or Incorrect Structure please DELETE or REVERT file at: {CONFIG_FILE_PATH}\n'
                f'Attempt(s) {settings_conf["st_count"]}/{settings_conf["st_count_re"]}')

    conf_settings = CONF['config']['settings']
    w.text_testing['text'] = conf_settings['testing']
    if conf_settings['def_coin'] == 'ETH':
        PAST_TRADED_CRYPTO = 1
        w.rb_cont_arr[1].invoke()
    elif conf_settings['def_coin'] == 'XRP':
        PAST_TRADED_CRYPTO = 2
        w.rb_cont_arr[2].invoke()
    elif conf_settings['def_coin'] == 'EOS':
        PAST_TRADED_CRYPTO = 3
        w.rb_cont_arr[3].invoke()
    else:
        PAST_TRADED_CRYPTO = 0
        w.rb_cont_arr[0].invoke()


def write_config(conf_out: dict):
    """
    Write passed config_out to config file & safe balance_df then print saved to console.

    Required Arguments:
        config_out(dict):   Config dictionary to wright to yaml
    """
    # write given file to yaml as config file
    global CONFIG_FILE_PATH
    BALANCE_DF.to_csv(BALANCE_DF_PATH)
    with open(CONFIG_FILE_PATH, 'w') as output:
        yaml.dump(conf_out, output, sort_keys=False, default_flow_style=False)
    print('SAVED')


def api_change():
    """
    To be called after changing the api key or secret.
    Updates values for GUI as well as CLIENT and fetches new balance.
    """
    global MAIN_DF, CLIENT, w
    w.text_api_key['text'] = obfuscate_api_info(CONF['config']['user']['api_key'])
    w.text_api_secret['text'] = obfuscate_api_info(CONF['config']['user']['api_secret'])
    if obfuscate_api_info(CONF['config']['user']['api_key']) == 'ERROR' or obfuscate_api_info(
            CONF['config']['user']['api_secret']) == 'ERROR':
        trigger_error_bar(lines=['ERROR:  API Key/Secret', 'Try Updating API Key or Secret'], duration=10)

    safe_start_thread(
        AsyncFetchBalance(w, key=CONF['config']['user']['api_key'], secret=CONF['config']['user']['api_secret']))
    CLIENT = bybit(test=False, api_key=CONF['config']['user']['api_key'],
                   api_secret=CONF['config']['user']['api_secret'])
    safe_start_thread(AsyncDataHandler(w, MAIN_DF, CLIENT, SYMBOLS))


def trading_loop():
    """Main loop to call AsyncFetchBalance() thread and AsyncDataHandler() thread."""
    global TRADING_COUNT, w
    if TRADING_COUNT >= 1:
        safe_start_thread(
            AsyncFetchBalance(w, key=CONF['config']['user']['api_key'], secret=CONF['config']['user']['api_secret']))
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
    w.text_testing['text'] = str(bool(CONF['config']['settings']['testing']))
    w.text_pred_total['text'] = CONF['config']['stats']['total_prediction']
    w.text_pred_correct['text'] = CONF['config']['stats']['correct_prediction']
    w.text_total_profit['text'] = conv_currency_str(CONF['config']['stats']['total_profit'])
    w.text_last_trade_profit['text'] = conv_currency_str(CONF['config']['stats']['last_trade_profit'])
    if int(CONF['config']['stats']['last_trade_profit']) > 0:  # green text color
        w.text_last_trade_profit['foreground'] = '#25b100'
    elif int(CONF['config']['stats']['last_trade_profit']) < 0:  # red text color
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
            'text'] = f'{(int(CONF["config"]["stats"]["correct_prediction"]) / int(CONF["config"]["stats"]["total_prediction"])) * 100}%'
    except ZeroDivisionError:
        w.text_pred_acc['text'] = '-%'

    if len(THREAD_POOL) > 0:
        THREAD_POOL = [t for t in THREAD_POOL if t.is_alive()]
    REFRESH_COUNT += 1
    if IS_QUITTING:
        root.after(500, lambda: quitting_text())
    else:
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
    print(f'* ERROR: {f_text}*')
    w.text_error_bar['text'] = str(f_text)
    w.text_error_bar['foreground'] = '#880808'
    root.after(int(duration * 1000), lambda: trigger_error_bar(lines, duration, close=True))


def reset_stats():
    """Reset stats to '0' and trade settings to default."""
    global PAST_TRADE_VAL, MAX_LOSS_VAL, PAST_BAL_LOSS, w
    CONF['config']['settings']['trade_amount'] = 1
    w.entry_trade_amount.delete(0, 'end')
    w.entry_trade_amount.insert(0, CONF['config']['settings']['trade_amount'])
    PAST_TRADE_VAL = CONF['config']['settings']['trade_amount']
    CONF['config']['settings']['loss_amount'] = 5
    w.entry_loss.delete(0, 'end')
    w.entry_loss.insert(0, CONF['config']['settings']['loss_amount'])
    MAX_LOSS_VAL = CONF['config']['settings']['loss_amount']
    CONF['config']['settings']['loss_period'] = 10
    w.entry_loss_period.delete(0, 'end')
    w.entry_loss_period.insert(0, CONF['config']['settings']['loss_period'])
    t_p_b_l = PAST_BAL_LOSS
    PAST_BAL_LOSS = deque(maxlen=CONF['config']['settings']['loss_period'])
    PAST_BAL_LOSS.append(t_p_b_l)
    CONF['config']['stats']['total_prediction'] = 0
    CONF['config']['stats']['correct_prediction'] = 0
    CONF['config']['stats']['total_profit'] = 0
    CONF['config']['stats']['last_trade_profit'] = 0
    write_config(CONF)


def save_settings():
    """Call when save settings button is clicked. Saves API key & secret to config then calls api_change()."""
    t_api_key = api_key_entry.get().strip()
    con_key = CONF['config']['user']['api_key']
    t_api_secret = api_secret_entry.get().strip()
    con_secret = CONF['config']['user']['api_secret']
    if (not t_api_key or not t_api_secret) and (con_key != t_api_key or con_secret != t_api_secret):
        CONF['config']['user']['api_key'] = t_api_key
        CONF['config']['user']['api_secret'] = t_api_secret
        if os.path.exists(BALANCE_DF_PATH):
            os.remove(BALANCE_DF_PATH)
        safe_start_thread(AsyncBuildBalance())
    w.entry_api_key.delete(0, 'end')
    w.entry_api_secret.delete(0, 'end')
    write_config(CONF)
    api_change()


def testing_toggle():
    """When test button pressed Toggles test mode."""
    CONF['config']['settings']['testing'] = not bool(CONF['config']['settings']['testing'])


def place_moving_avg_btns(gui: bybit_gui.Toplevel1, var_set: dict, m_offset=0):
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
    if list(MOVING_AVG_BAL_DICT.keys()) == list(var_set.keys()):
        gui.combo_box_bal_period.place(x=6 + offset + (len(var_set.keys()) * 55), rely=0.25, height=20, width=80)
    else:
        gui.combo_box_bal_period.place_forget()


def moving_avg_check_btn():
    """
    Checks if moving ave buttons pressed and calls AsyncFetchPlot() as long as time
    is greater than 15sec from refreshing.

    Required Arguments:
        event:  On click event.
    """
    CONF['config']['settings']['PMA'] = [x.get() for x in MOVING_AVG_DICT.values()]
    CONF['config']['settings']['BMA'] = [x.get() for x in MOVING_AVG_DICT.values()]
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
    CONF['config']['settings']['plot_display'] = plot_type.get()
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
    CONF['config']['settings']['plot_display'] = bal_period.get()
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
    selected_file = filedialog.askopenfilename(initialdir=os.getcwd(), title=f"Select {sym} Model File",
                                               filetypes=(("h5 model files", "*.h5"), ("hdf5 model files", "*.hdf5"),
                                                          ("all files", "*.*")))
    if selected_file != '':
        safe_start_thread(AsyncTestModel(model_path=selected_file.strip(), symbol=sym))
    else:
        print('canceled')


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
        print(f'model: {symbol} not loaded')
        return
    model.summary(line_length=100, print_fn=lambda x: summary.append(x))
    if not return_str:
        bybit_gui.new_model_info_window(bybit_gui.ModelInfoWindow, summary=summary)
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
    PAST_TRADE_VAL = CONF['config']['settings']['trade_amount']
    CONF['config']['settings']['trade_amount'] = w.entry_trade_amount.get()
    MAX_LOSS_VAL = CONF['config']['settings']['loss_amount']
    CONF['config']['settings']['loss_amount'] = w.entry_loss.get()
    CONF['config']['settings']['loss_period'] = w.entry_loss_period.get()


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


if __name__ == '__main__':
    import tempbybit_gui

    tempbybit_gui.vp_start_gui()
