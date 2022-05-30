import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # change tensorflow logs to not show.
from keras.models import load_model
from keras import layers
import pandas as pd

pd.options.mode.chained_assignment = None
from math import ceil
import random
from bybit import bybit
import yaml
import threading
import os.path
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import PhotoImage
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import ccxt

'''
Created by: Eric Kelm
    automatically trade cryptocurrencies from ByBit. with the help of a Neral network.
    Trade Derivative "Inverse Perpetual" contracts on bybit.
    (more info here under "Perpetual Swaps": https://learn.bybit.com/trading/what-is-crypto-derivatives-trading-how-does-it-work/#The_Types_of_Derivatives_Trading)
    set the amount to trade, when to stop automatically trading based on loss over a period of time.
    display your total balance as well as price and moving average of coin prices over different periods
    with a GUI and build and train simple models within the GUI
'''


class Bcolors:
    """Terminal color codes"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class PlotTypes:
    """Types of plots that can be created"""
    price = 'Price'
    balance = 'Balance'
    both = 'Both'
    all = [price, balance, both]


class DataPeriods:
    """Periods data can be loaded in from bybit"""
    min1, min3, min5, min15, min30 = 1, 3, 5, 15, 30
    hour1, hour2, hour4, hour6, hour12 = 60, 120, 240, 360, 720
    all = {'1 min': min1, '3 min': min3, '5 min': min5, '15 min': min15, '30 min': min30,
           '1 hour': hour1, '2 hours': hour2, '4 hours': hour4, '6 hours': hour6, '12 hours': hour12}


class StartStopBtnText:
    """Different text options for start trade button"""
    start = 'Start Trading'
    stop = 'Stop Trading'
    loading = 'Loading'


class LocalBybit:
    """Bybit client object"""

    @staticmethod
    def set_client(key, secret):
        """Create bybit client object with api key and secret"""
        return bybit(test=False, api_key=key, api_secret=secret)

    @staticmethod
    def set_leverage(symbol, leverage):
        """Set bybit account leverage"""
        global CLIENT
        return CLIENT.Positions.Positions_saveLeverage(symbol=f'{symbol}USD', leverage=str(leverage))


class Exchange:
    """CCXT client object"""

    @staticmethod
    def set_client(key, secret):
        """Create ccxt bybit client with api key and secret"""
        return ccxt.bybit({'options': {'adjustForTimeDifference': True, },
                           'apiKey': key, 'secret': secret, })


class Symbols:
    """Symbols that can be traded via bybit with list option"""
    BTC = 'BTC'
    ETH = 'ETH'
    XRP = 'XRP'
    EOS = 'EOS'
    all = [BTC, ETH, XRP, EOS]


class Position:
    """
    Holds current session trading position
    https://learn.bybit.com/bybit-guide/bybit-trading-fees/
    """
    current_qty, past_qty = 0, 0
    last_order_value = ['none', 0]
    p = 'hold'
    maker_fee = 0.006
    taker_fee = 0.05

    def __init__(self, symbol=None, ):
        """init position object"""
        if symbol is None:
            symbol = Symbols.BTC
        self.symbol = symbol
        self.past_symbol = symbol
        self.past_leverage = 1
        self.last_order_value = ['none', 0]

    def update(self, symbol=None, qty=None, position=None, leverage=None):
        """Update position valued and past position values"""
        if symbol is not None:
            self.set_symbol(symbol)
        if qty is not None:
            self.set_qty(qty)
        if leverage is not None:
            self.past_leverage = leverage
        self.p = position

    def set_symbol(self, new_symbol):
        """Updates symbol att and past symbol att"""
        global CONF
        self.past_symbol = self.symbol
        self.symbol = new_symbol
        CONF.def_coin = new_symbol

    def set_qty(self, qty):
        """Updates quantity att and past quantity att"""
        self.past_qty = self.current_qty
        self.current_qty = qty

    def place_trade(self, side, symbol, amount):
        """
        Checks if testing, updates leverage, and places a trade to bybit exchange for given side,symbol,and amount.
        If testing updated testing values
        """
        global EXCHANGE, TESTING_PROFIT, CONF, LEVERAGE, CLIENT
        out = 'TESTING LOCK'
        if self.past_leverage != CONF.leverage:
            LocalBybit.set_leverage(symbol, CONF.leverage)
        if not (HARD_TESTING_LOCK or CONF.testing):
            out = 'Hold'
            self.p = side
            if side == 'buy':
                out = EXCHANGE.create_market_buy_order(symbol=f"{symbol}/USD", amount=amount)['id']
            elif side == 'sell':
                out = EXCHANGE.create_market_sell_order(symbol=f"{symbol}/USD", amount=amount)['id']
            if side in ['buy', 'sell']:
                print(Bcolors.OKCYAN + f'{side.upper()} {symbol} with qty of: {amount}')
        else:
            new_order_value = amount / EXCHANGE.fetch_ticker(f'{symbol}/USD')['close'] * CONF.leverage
            if self.last_order_value[1] != 0:
                change = self.last_order_value[1] - new_order_value
                if self.last_order_value[0] == 'sell' and change < 0:
                    change = abs(change)
                TESTING_PROFIT += change - ((new_order_value * self.taker_fee) * CONF.leverage)
            self.last_order_value = [side, new_order_value]
        return Bcolors.OKCYAN + out + Bcolors.ENDC

    def cancel_trades(self):
        """Cancels any trades that are outstanding in any symbol"""
        order = 'No orders canceled.'
        if not (HARD_TESTING_LOCK or CONF.testing):
            if EXCHANGE.fetch_balance()[self.symbol]['used'] > 0.0:
                if POSITION.p == 'buy':
                    order = self.place_trade(side='sell', symbol=POSITION.symbol, amount=POSITION.current_qty)
                elif POSITION.p == 'sell':
                    order = self.place_trade(side='buy', symbol=POSITION.symbol, amount=POSITION.current_qty)
            if self.past_symbol != self.symbol and EXCHANGE.fetch_balance()[self.past_symbol]['used'] > 0.0:
                if self.p == 'buy':
                    order += ' | ' + self.place_trade(side='sell', symbol=self.past_symbol, amount=self.past_qty)
                elif self.p == 'sell':
                    order += ' | ' + self.place_trade(side='buy', symbol=self.past_symbol, amount=self.past_qty)
            self.__init__(self.symbol)
        return order


class LayerOptions:
    """Layer options for model generation and dictionary and default layer"""
    LSTM = layers.LSTM
    GRU = layers.GRU
    RNN = layers.RNN
    simpleRNN = layers.SimpleRNN
    dense = layers.Dense
    layer_dict = {'LSTM': LSTM, 'GRU': GRU, 'RNN': RNN, 'SimpleRNN': simpleRNN, 'Dense': dense}
    DEFAULT_BLUEPRINT_LAYER = {'layer': 'Dense', 'n': 16, 'drop': 0.0}  # default layer type


class Model:
    """
    Model object.
    information about the model needed to train and prep the data
    to be predicted.
    """

    def __init__(self, **kwargs):
        """init model class and load model from disk if passed location"""
        self.location = kwargs.get('location')
        if self.location is not None and self.location.split('.')[-1] == 'h5':
            self.model = kwargs.get('model')
            if self.model is not None and self.location is not None:
                self.model.save(self.location)
            elif self.location is not None:
                try:
                    self.model = load_model(self.location)
                except (FileNotFoundError, OSError):
                    print(Bcolors.WARNING + f'WARNING: Model could not be found at: {self.location}')
                    self.location = None
            self.seq = kwargs.get('seq')
            self.batch = kwargs.get('batch')
            self.future_p = kwargs.get('future')
            self.moving_avg = kwargs.get('moving_avg')
            self.e_moving_avg = kwargs.get('e_moving_avg')
            self.drop_symbols = kwargs.get('drop')
            self.data_period = kwargs.get('period')
            self.layers = kwargs.get('layers')
        else:
            self.model = None
            self.seq = None
            self.batch = None
            self.future_p = None
            self.moving_avg = None
            self.e_moving_avg = None
            self.drop_symbols = None
            self.data_period = None
            self.layers = None

    def dump(self):
        """Returns dict of model atts"""
        if self.model is not None and self.location is not None:
            self.model.save(self.location)
        return {'location': self.location, 'seq': self.seq, 'batch': self.batch, 'future': self.future_p, 'moving_avg': self.moving_avg,
                'e_moving_avg': self.e_moving_avg, 'drop': self.drop_symbols, 'period': self.data_period, 'layers': self.layers}

    def move_model(self, n_location):
        """Moves model file to a new location on disk"""
        import os
        os.rename(self.location, n_location)
        try:
            os.remove(self.location)
        except FileNotFoundError:
            pass
        self.location = n_location

    def delete_model(self):
        """Delete model from disk"""
        if self.location is not None and os.path.isfile(self.location):
            os.remove(self.location)
            self.location = None

    def clear(self):
        """Delete model and re initialize class"""
        self.delete_model()
        self.__init__()

    def prediction(self, data: pd.DataFrame(), symbol):
        """
        Gets new data from bybit and makes prediction with model att.
        Returns 'buy' if prediction is higher than last closing price, and 'sell' vise versa
        """
        from bin.networkTraining import Training
        import numpy as np
        global CONF
        processed_data = Training(CONF.key, CONF.secret).build_data(data.copy(), target=symbol, seq=self.seq, future_p=self.future_p,
                                                                    drop=self.drop_symbols, moving_avg=self.moving_avg,
                                                                    ema=self.e_moving_avg, prediction=True)
        last_close = processed_data[f'{symbol}_close'].values[-1]
        for layer in self.layers:
            if layer['layer'] in ['LSTM', 'GRU', 'RNN', 'SimpleRNN']:
                processed_data = np.asarray([processed_data])  # 2 wraps... [[x.xxx]]
                break
        if self.model.predict(processed_data)[0][0] > last_close:
            return 'buy'
        else:
            return 'sell'

    def get_max_datasize(self):
        """Returns max data size needed to train the model. if model class not fully initialize will return DATA_SIZE"""
        if self.seq is not None:
            if self.moving_avg is not None and self.e_moving_avg is not None \
                    and len(self.moving_avg) + len(self.e_moving_avg) > 0:
                return self.seq + max(*self.moving_avg, *self.e_moving_avg)
            return self.seq
        return DATA_SIZE


class Config:
    """
    Configuration file object.
    load and write yaml file to and from disk.
    """

    DATA_SIZE = 120  # default data size
    DEFAULT_CONFIG_FILEPATH = 'bin/res/config.yaml'  # config file location

    def __init__(self, filepath=DEFAULT_CONFIG_FILEPATH):
        """init config class and loads default config from disk"""
        if filepath is None:
            c = self.default_config()
        else:
            c = self.load_file(filepath)
        try:
            self.key = c.get('user').get('api_key')
            self.secret = c.get('user').get('api_secret')
            settings = c.get('settings')
            if settings.get('def_coin') in Symbols.all:
                self.def_coin = settings.get('def_coin')
            else:
                raise ValueError
            if not (1 <= int(c.get('settings').get('leverage')) <= 100):
                c['settings']['leverage'] = 1
            self.leverage = int(c.get('settings').get('leverage'))
            if settings.get('plot_display') in PlotTypes.all:
                self.plot_display = settings.get('plot_display')
            else:
                raise ValueError
            self.balance_plot_period = settings.get('balance_plot_period')
            self.plot_moving_avg = settings.get('PMA')
            self.balance_moving_avg = settings.get('BMA')
            self.testing = bool(settings.get('testing'))
            self.trade_amount = int(settings.get('trade_amount'))
            self.loss_amount = int(settings.get('loss_amount'))
            self.loss_period = int(settings.get('loss_period'))
            self.models = {Symbols.BTC: Model(**settings.get('btc_model')),
                           Symbols.ETH: Model(**settings.get('eth_model')),
                           Symbols.XRP: Model(**settings.get('xrp_model')),
                           Symbols.EOS: Model(**settings.get('eos_model'))}
            self.default_model_arguments = settings.get('default_model_arguments')
            self.model_blueprint = settings.get('model_blueprint')
            stats = c.get('stats')
            self.total_profit = float(stats.get('total_profit'))
            self.last_cap_bal = float(stats.get('last_cap_bal'))
            self.total_prediction = int(stats.get('total_prediction'))
            self.correct_prediction = int(stats.get('correct_prediction'))
        except (ValueError, TypeError) as e:
            print(e)
            print(Bcolors.FAIL + f'ERROR: Config file ValueError | TypeError.\tDelete Config file at: {CONFIG_FILE_PATH}')
            exit(-1)

        self.write_config(filepath)

    def load_file(self, filepath):
        """Loads config yaml file from disk"""
        global w
        global CONFIG_FILE_PATH
        try:
            with open(filepath, 'r') as file:
                conf = yaml.safe_load(file)
        except FileNotFoundError:
            print(Bcolors.FAIL + f'ERROR: Config file not found at: {CONFIG_FILE_PATH}\nReplacing with default config file.')
            conf = self.default_config()
        if not self._checksum_config(conf):
            print(Bcolors.WARNING + 'WARNING: Config file has unexpected structure.' + Bcolors.ENDC + ' This may cause errors or crashes.\n'
                                                                                                      f'Reset config file in the settings or modify '
                                                                                                      f'it at: {CONFIG_FILE_PATH}')
        return conf

    def _checksum_config(self, conf: dict):
        """Compares keys in default config file to passed config file"""
        conf_sum = ''
        for key in self._rec_config(conf):
            conf_sum += key
        default_sum = ''
        for key in self._rec_config(self.default_config()):
            default_sum += key
        return hash(default_sum) == hash(conf_sum)

    def _rec_config(self, dictionary: dict):
        """recursively iterate through dictionary"""
        for key, value in dictionary.items():
            if type(value) is dict:
                yield key
                yield from self._rec_config(value)
            else:
                yield key

    def write_config(self, filepath=DEFAULT_CONFIG_FILEPATH):
        """Write config file to disk"""
        c = {'user': {'api_key': self.key, 'api_secret': self.secret},
             'settings': {'def_coin': self.def_coin, 'leverage': self.leverage, 'plot_display': self.plot_display,
                          'balance_plot_period': self.balance_plot_period,
                          'PMA': self.plot_moving_avg, 'BMA': self.balance_moving_avg,
                          'testing': self.testing, 'trade_amount': self.trade_amount, 'loss_amount': self.loss_amount,
                          'loss_period': self.loss_period,
                          'btc_model': None, 'eth_model': None, 'xrp_model': None, 'eos_model': None,
                          'default_model_arguments': self.default_model_arguments,
                          'model_blueprint': self.model_blueprint},
             'stats': {'total_profit': self.total_profit, 'last_cap_bal': self.last_cap_bal,
                       'total_prediction': self.total_prediction, 'correct_prediction': self.correct_prediction}}
        for k, v in self.models.items():
            if v is not None:
                c['settings'][f'{k.lower()}_model'] = v.dump()
            else:
                c['settings'][f'{k.lower()}_model'] = Model().dump()
        with open(filepath, 'w') as output:
            yaml.dump(c, output, sort_keys=False, default_flow_style=False)

    @staticmethod
    def default_config():
        """Default config file. incase user config file is corupt or missing"""
        return {'user': {'api_key': None, 'api_secret': None},
                'settings': {'def_coin': Symbols.BTC, 'leverage': 1, 'plot_display': PlotTypes.price,
                             'balance_plot_period': list(BALANCE_PERIODS_DICT)[0],
                             'PMA': [0] * len(MOVING_AVG_DICT.values()),
                             'BMA': [0] * len(MOVING_AVG_BAL_DICT.values()),
                             'testing': True, 'trade_amount': 1, 'loss_amount': 5, 'loss_period': 10,
                             'btc_model': Model().dump(), 'eth_model': Model().dump(),
                             'xrp_model': Model().dump(), 'eos_model': Model().dump(),
                             'default_model_arguments': {'target': 'BTC', 'future': 1, 'size': 1000, 'period': 1,
                                                         'epochs': 10, 'seq': 128, 'batch': 64, 'ma': [5, 10], 'ema': [4, 8]},
                             'model_blueprint': [LayerOptions.DEFAULT_BLUEPRINT_LAYER] * 3},
                'stats': {'total_profit': 0, 'last_cap_bal': 0, 'total_prediction': 0, 'correct_prediction': 0}}


class AsyncFetchBalance(threading.Thread):
    """
    Thread
    Get account balance from bybit with provided api key and secret.
    Display and print to console a warning if api key or secret is invalid
    """

    tag = 'fetch_balance'

    def __init__(self, key: str, secret: str):
        """init fetch balance thread"""
        super().__init__()
        self.key = key
        self.secret = secret
        self.st = 0

    def run(self):
        """
        Get account balance associated with api key and secret. Display error on gui if Authentication Error
        Print tag, thread duration, and passed arguments when thread is done.
        """
        global CONF, BALANCE_DF, EXCHANGE, TESTING_PROFIT
        self.st = time.time()
        args = []
        try:
            t_count = 0
            while True:
                try:
                    crypto_balances = {s: EXCHANGE.fetch_balance().get(s)['free'] for s in Symbols.all}
                    break
                except ccxt.errors.NetworkError:
                    t_count += 1
                    if t_count > 10:
                        args = [f'CCXT can not connect to servers. Please try again later.']
                        raise ccxt.errors.NetworkError
                    time.sleep(0.5)
            balance_usd = {s: (crypto_balances[s] * EXCHANGE.fetch_ticker(f'{s}/USD')['close']) for s in Symbols.all}
            s_bal = round(sum(balance_usd.values()), 10)
            change = 0
            if float(CONF.last_cap_bal) != 0:
                change = s_bal - float(CONF.last_cap_bal)
                CONF.total_profit += change
            CONF.last_cap_bal = s_bal
            d_time = int(time.time() - (time.time() % 60))
            if len(BALANCE_DF.index) > 0 and BALANCE_DF['Time'].values[-1] != d_time:
                BALANCE_DF = pd.concat(
                    [BALANCE_DF, pd.DataFrame([[d_time, s_bal, s_bal + TESTING_PROFIT]], columns=['Time', 'Balance', 'Test Balance'])])
            BALANCE_DF = BALANCE_DF[-525600:]
            BALANCE_DF.to_csv(BALANCE_DF_PATH)
            for count, s in enumerate(Symbols.all):
                w.text_amount_arr[count]['text'] = crypto_balances[s]
                w.text_usd_arr[count]['text'] = f'({conv_currency_str(balance_usd[s])} USD)'
            args = [f'Change: {conv_currency_str(change)}']
        except ccxt.errors.AuthenticationError:
            for rb_label in w.text_amount_arr:
                rb_label['text'] = ''
            for rb_label2 in w.text_usd_arr:
                rb_label2['text'] = ''
            w.text_amount_arr[0]['text'] = 'Authentication'
            w.text_usd_arr[0]['text'] = 'Error'
            w.text_amount_arr[2]['text'] = 'Check API Key'
            w.text_usd_arr[2]['text'] = 'and Secret'
            args = ['[!] Authentication Error\n\tCheck API key and secret and try again']
        finally:
            print_finish_thread(self.tag, self.st, args=args)


class AsyncDataHandler(threading.Thread):
    """
    Thread
    collect price data from bybit server
    then calls 'AsyncFetchPlot' with a delay if set via 'fetch_plot_delay'
    and calls 'AsyncPlaceTrade' if currently trading
    """

    tag = 'data_handler'

    def __init__(self, verbose=0, fetch_plot_delay=0.0):
        """init data handler thread"""
        super().__init__()
        self.fetch_plot_delay = fetch_plot_delay
        self.verbose = verbose
        self.st = 0

    def run(self):
        """
        Run data handler. Set 'MAIN_DF' with new price data from bybit
        Print tag, thread duration, and passed arguments
        """
        from bin import data_handler
        global MAIN_DF, CONF, DATA_SIZE, TRADING_COUNT, CLIENT
        self.st = time.time()
        symbol = Symbols.all[TRADING_CRYPTO.get()]
        DATA_SIZE = CONF.models.get(symbol).get_max_datasize()
        MAIN_DF = data_handler.Constructor(None, client=CLIENT, data_s=ceil(DATA_SIZE / 200), force_new=True, save_file=False).get_data()[-DATA_SIZE:]
        print_finish_thread(self.tag, self.st)
        safe_start_thread(AsyncFetchPlot(delay=self.fetch_plot_delay))
        if int(CONF.trade_amount) > 0 and IS_TRADING:
            safe_start_thread(AsyncPlaceTrade(symbol))  # place trade thread


class AsyncFetchPlot(threading.Thread):
    """
    Thread
    Constructs price and balance graphs for gui
    can be delayed for x seconds by setting 'delay' variable
    """
    tag = 'fetch_plot'

    def __init__(self, delay=0.0):
        """init fetch plots thread"""
        super().__init__()
        global DATA_SIZE, CONF
        sns.set_theme()
        sns.set(rc={'figure.facecolor': "#E6E6E6"})
        self.delay = delay
        self.current_data = MAIN_DF
        self.sel_sym = Symbols.all[TRADING_CRYPTO.get()]
        DATA_SIZE = CONF.models[self.sel_sym].get_max_datasize()
        self.size = CONF.models[self.sel_sym].seq
        if self.size is None:
            self.size = DATA_SIZE
        self.st = 0

    def run(self):
        """
        Run fetch balance thread and populate price and/or balance graph
        Print tag, thread duration, and passed arguments
        """
        global w
        time.sleep(self.delay)
        self.st = time.time()
        plot_size = {'x': 4, 'rely': 0.290, 'relheight': 0.650, 'width': 584}
        args = []
        to_rm = [child for child in w.info_frame.winfo_children() if type(child) == tk.Canvas]
        if PLOT_TYPE.get() == PlotTypes.price:
            dpi = 65
            place_moving_avg_btns(w, MOVING_AVG_DICT)
            args.append(self.price_plot(plot_size, dpi))
        elif PLOT_TYPE.get() == PlotTypes.balance:
            dpi = 70
            place_moving_avg_btns(w, MOVING_AVG_BAL_DICT)
            args.append(self.balance_plot(plot_size, dpi))
        elif PLOT_TYPE.get() == PlotTypes.both:
            place_moving_avg_btns(w, MOVING_AVG_DICT, m_offset=0)
            place_moving_avg_btns(w, MOVING_AVG_BAL_DICT, m_offset=307)
            dpi = 62
            plot_size['relheight'] = (plot_size['relheight'] / 2)
            args.append(self.price_plot(plot_size, dpi))
            plot_size['rely'] = (plot_size['rely'] * 2.12)
            args.append(self.balance_plot(plot_size, dpi))
        [child.destroy() for child in to_rm]
        # if len(args) == 0:
        #    args = None
        print_finish_thread(self.tag, self.st, args=args)

    def price_plot(self, plot_size, dpi):
        """populate price graph"""
        try:
            data = self.current_data[[f'{self.sel_sym}_close']]
        except KeyError:
            return
        data.rename(columns={f'{self.sel_sym}_close': 'Price'}, inplace=True)
        data.sort_index(inplace=True)
        data['Time'] = self.current_data.index
        data.reset_index(drop=True, inplace=True)
        for key, val in MOVING_AVG_DICT.items():
            if val.get() == 1:
                if key.isdigit():
                    data[f'MA{key}'] = data['Price'].rolling(key, win_type='triang').mean()
                elif key == 'EMA':
                    data[f'EMA'] = data['Price'].ewm(span=8, adjust=False).mean()
        data = data[-self.size:]
        data = pd.melt(data, 'Time')
        figure = plt.Figure(dpi=dpi)
        ax = figure.subplots()
        ax.yaxis.set_major_formatter(FuncFormatter(lambda z, pos: conv_currency_str(z)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda z, pos: None))
        ax.set(xlabel=f'$Time$', ylabel='Price')
        p = sns.lineplot(data=data, y='value', x='Time', hue='variable', ax=ax, legend=True)
        p.set_title(f'{self.sel_sym} Price\nOver {self.size} min')
        ax.legend(loc='upper left')
        w.canvas_open_close = FigureCanvasTkAgg(figure, w.info_frame)
        w.canvas_open_close.draw()
        w.canvas_open_close.get_tk_widget().place(x=plot_size['x'], rely=plot_size['rely'],
                                                  relheight=plot_size['relheight'], width=plot_size['width'])
        if StartStopBtnText.loading in w.btn_start_stop['text']:
            w.loading_plot_label.destroy()
            w.btn_start_stop['text'] = StartStopBtnText.start
            w.btn_start_stop.state(['!disabled'])
        return 'PRICE'

    def balance_plot(self, plot_size, dpi):
        """populate balance graph"""
        global BALANCE_DF, BAL_PERIOD
        while AsyncFetchBalance.tag in [tag.tag for tag in THREAD_POOL]:
            time.sleep(0.5)
        data = BALANCE_DF[525600 - int(BALANCE_PERIODS_DICT.get(BAL_PERIOD.get())):]
        if not CONF.testing:
            data.drop(columns=['Test Balance'], inplace=True)
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
        p.set_title(f'Total Balance for last {BAL_PERIOD.get()}')
        w.canvas_open_close_2 = FigureCanvasTkAgg(figure_2, w.info_frame)
        w.canvas_open_close_2.draw()
        w.canvas_open_close_2.get_tk_widget().place(x=plot_size['x'], rely=plot_size['rely'],
                                                    relheight=plot_size['relheight'], width=plot_size['width'])
        if StartStopBtnText.loading in w.btn_start_stop['text']:
            w.loading_plot_label.destroy()
            w.btn_start_stop['text'] = StartStopBtnText.start
            w.btn_start_stop.state(['!disabled'])
        return f'BALANCE [{BAL_PERIOD.get()} period]'


class AsyncPlaceTrade(threading.Thread):
    """
    Thread
    check's loss safety before canceling previous trades if crypto changed.
    as well as updates gui model accuracy.
    """
    tag = 'trade'

    def __init__(self, symbol: str):
        """init place trade thread"""
        super().__init__()
        self.symbol = symbol
        self.current_data = pd.DataFrame()
        self.st = 0

    def run(self):
        """
        Run Place trade thread. sleep 10 sec to allow bybit to have proper data.
        Updates prediction accuracy and position object.
        Prints a warning if no model is found and switches to testing.
        Stops trading if balance is insufficient
        """
        global CONF, PREDICTION, POSITION, LEVERAGE, TRADING_COUNT, CLIENT
        time.sleep(10)
        self.st = time.time()
        TRADING_COUNT += 1
        if TRADING_COUNT % int(CONF.models.get(self.symbol).data_period) != 0:
            print_finish_thread(self.tag, self.st,
                                [f'{int(CONF.models.get(self.symbol).data_period) - TRADING_COUNT % int(CONF.models.get(self.symbol).data_period)} '
                                 f' more count till trade'])
            return
        else:
            from bin import data_handler
            self.current_data = data_handler.Constructor(None, client=CLIENT, data_s=ceil((DATA_SIZE + int(DATA_SIZE * 0.1)) / 200), save_file=False,
                                                         data_p=CONF.models.get(self.symbol).data_period, force_new=True).get_data()[-DATA_SIZE:]
        CONF.leverage = int(LEVERAGE.get())
        POSITION.update(symbol=self.symbol, qty=int(w.entry_trade_amount.get()), position=PREDICTION)
        status, order = 'OK', 'none'
        self.check_past_prediction()
        loss, loss_period = int(w.entry_loss.get()), int(w.entry_loss_period.get())
        if self.check_balance_trend(loss, loss_period):
            if CONF.models.get(POSITION.symbol).model is None:
                testing_toggle(True)
                print(Bcolors.WARNING + 'WARNING: No Model Found.' + Bcolors.ENDC + ' Testing with random predictions.')
                PREDICTION = random.choice(['buy', 'sell'])
            else:
                PREDICTION = CONF.models.get(POSITION.symbol).prediction(self.current_data, POSITION.symbol)
            # sell past order if last traded symbol is different from currently trading.
            if POSITION.past_symbol != POSITION.symbol:
                # trading new crypto. nullify last trade
                if POSITION.p == 'buy':
                    # if position for previous crypto was BUY, then SELL it
                    order = POSITION.place_trade(side='sell', symbol=POSITION.past_symbol, amount=POSITION.past_qty)
                elif POSITION.p == 'sell':
                    # if position for previous crypto was SELL, then BUY it
                    order = POSITION.place_trade(side='buy', symbol=POSITION.past_symbol, amount=POSITION.past_qty)
                # place trade in the direction of the prediction. as well as sell/buy all previous orders.
            try:
                if PREDICTION == 'buy' and POSITION.p != 'buy':
                    order = POSITION.place_trade(side='buy', symbol=POSITION.symbol,
                                                 amount=POSITION.past_qty + POSITION.current_qty if POSITION.p == 'sell' else POSITION.current_qty)
                    status += ' buy order'
                elif PREDICTION == 'sell' and POSITION.p != 'sell':
                    order = POSITION.place_trade(side='sell', symbol=POSITION.symbol,
                                                 amount=POSITION.past_qty + POSITION.current_qty if POSITION.p == 'buy' else POSITION.current_qty)
                    status += ' sell order'
                else:
                    status += ' hold order'
                w.text_prediction['text'] = PREDICTION
            except ccxt.errors.InsufficientFunds:
                order = self.end_trading(error_bar=['INSUFFICIENT FUNDS', 'Enter Lower Trade Amount'])
                status = Bcolors.WARNING + f'WARNING: Insufficient Funds. Lower trade amount. Halting Trading.'
            POSITION.update(position=PREDICTION, leverage=CONF.leverage)
        else:
            order = self.end_trading(error_bar=['STOP LIMIT REACHED', 'Halting Trading'])
            status = Bcolors.WARNING + f'WARNING: Loss stop limit: ${loss} in {loss_period}min reached. Halting Trading.'
        print_finish_thread(self.tag, self.st, [status, 'ID: ' + order])

    def check_past_prediction(self):
        """Checks past prediction for model accuracy"""
        global PREDICTION
        current_price = self.current_data[f'{POSITION.past_symbol}_close'].values[-1]
        previous_price = self.current_data[f'{POSITION.past_symbol}_close'].values[-2]
        if (PREDICTION == 'buy' and current_price > previous_price) or (PREDICTION == 'sell' and (previous_price > current_price)):
            CONF.correct_prediction += 1
        if PREDICTION in ['buy', 'sell']:
            CONF.total_prediction += 1

    @staticmethod
    def check_balance_trend(loss, loss_period):
        """Check balance trend for automatic stop"""
        b_df = BALANCE_DF.copy()
        if loss_period == 0 or b_df['Balance'].values[-loss_period] == 0:
            return True
        if (b_df['Balance'].values[-1] - b_df['Balance'].values[-loss_period]) > -abs(loss):
            return True
        return False

    @staticmethod
    def end_trading(error_bar: list):
        """Close all open trade positions and end trading"""
        global IS_TRADING
        IS_TRADING = False
        order = POSITION.cancel_trades()
        w.text_prediction['text'] = '-'
        trigger_error_bar(error_bar, 5)
        return order


class AsyncTestModel(threading.Thread):
    """
    Thread
    gets data and evaluates passes model to verify its functional
    """
    tag = 'test_model'

    def __init__(self, data: pd.DataFrame, testing_model: Model, target: str, verbose: int = 0):
        """init test model Thread"""
        super().__init__()
        self.data = data
        self.verbose = verbose
        self.target = target
        self.model = testing_model
        self._return = 0
        self.st = 0

    def run(self):
        """
        Run test model thread. Takes passed model and runs keras evaluate function on it with model data arguments
        to check for any errors.
        Prints tag, thread duration, and passed arguments
        """
        self.st = time.time()
        self._return = self.test_model()
        print_finish_thread(self.tag, self.st, ['eval:', round(self._return, 6)])

    def join(self, *args):
        """Override join function"""
        threading.Thread.join(self, *args)
        return self._return

    def test_model(self):
        """
        Run test model thread. Takes passed model and runs keras evaluate function on it with model data arguments
        to check for any errors.
        """
        from bin.networkTraining import Training
        global CONF
        test_x, test_y = Training(CONF.key, CONF.secret).build_data(self.data, self.target, self.model.seq, self.model.future_p,
                                                                    self.model.drop_symbols, self.model.moving_avg, self.model.e_moving_avg,
                                                                    test_data_only=True)
        return self.model.model.evaluate(test_x, test_y, self.model.batch, verbose=self.verbose)


class AsyncTrainModel(threading.Thread):
    """
    Thread
    Trains model via networkTraining.py the  compares ot to existing model if it is better and should be replaced
    """
    tag = 'train_model'

    def __init__(self, data_args: dict, model_args: dict, verbose: int, force_new: bool):
        """init training model thread"""
        super().__init__()
        self.data_args = data_args
        self.model_args = model_args
        self.verbose = verbose
        self.force_new = force_new
        self.st = 0
        self._kill = False

    def run(self):
        """
        Run train model thread. Downloads data from bybit, then parses data, then fits model to data.
        After fitting the new model is compared to the previous model if exists.
        Prints tag, thread duration, and passed arguments.
        """
        self.st = time.time()
        from bin.networkTraining import Training
        global CONF, CLIENT
        trainer = Training(CONF.key, CONF.secret, gui=w, parent=self, verbose=self.verbose)
        raw_data, new_model = trainer.train(CLIENT, self.data_args, self.model_args, self.force_new, save_file=True)
        old_model = CONF.models.get(self.data_args.get('target'))
        if old_model.model is None:
            CONF.models.get(self.data_args.get('target')).delete_model()
            CONF.models[self.data_args.get('target')] = new_model
            trainer.verbose_print('New Model Saved.', max_val=1, update=1, reset=True)
            status = 'No model found. New model saved.'
        else:
            trainer.verbose_print('Evaluating.', max_val=2, reset=True)
            time.sleep(1)
            evaluations = [AsyncTestModel(raw_data.copy(), new_model, self.data_args.get('target')).test_model(),
                           AsyncTestModel(raw_data.copy(), old_model, self.data_args.get('target')).test_model()]
            if evaluations[0] < evaluations[1]:
                CONF.models[self.data_args.get('target')].delete_model()
                CONF.models[self.data_args.get('target')] = new_model
            else:
                new_model.delete_model()
                del new_model
            evaluations.sort()
            status = f'Old model kept.\tEvaluation difference: {round((evaluations[0] / evaluations[1]) * 100, 2)}%'
        time.sleep(2)
        trainer.verbose_print('', reset=True)
        w.btn_start_train['text'] = 'Start Training'
        w.prog_bar_train['value'] = 0
        w.label_prog_percent['text'] = ''
        update_model_gui_names()
        print_finish_thread(self.tag, self.st, [Bcolors.OKCYAN + status + Bcolors.ENDC])

    def kill(self):
        """Set kill attribute to True to kill thread"""
        self._kill = True

    def check_kill(self):
        """Checks kill attribute and ends thread if true"""
        if self._kill:
            w.btn_start_train['text'] = 'Start Training'
            w.prog_bar_train['value'] = 0
            w.label_prog_percent['text'] = ''
            print(Bcolors.WARNING + "WARNING: Thread: 'AsyncTrainModel'" + Bcolors.FAIL + " Killed.")
            exit(2)


class AsyncBuildBalanceDf(threading.Thread):
    """
    Thread
    builds balance df if df doesn't exist.
    """
    tag = 'build_balance'

    def __init__(self):
        """init build balance thread"""
        super().__init__()
        self.st = 0

    def run(self):
        """
        Run build balance thread. Loads csv or balance, if file can't be loaded generate 1 year worth
        of balance the value being 0.
        After the csv is loaded the difference between time of loading and when last saved is filled with
        the last recorded value.
        Prints tag, thread duration, and passed arguments.
        """
        global BALANCE_DF, CONF
        self.st = time.time()
        current_time_m = (time.time() - (time.time() % 60))
        try:
            status = 'OK'
            BALANCE_DF = pd.read_csv(BALANCE_DF_PATH, index_col=0)
            BALANCE_DF.sort_values('Time', inplace=True)
            max_time = max(BALANCE_DF['Time'])
            last_bal = BALANCE_DF['Balance'].iloc[-1]
            data = []
            for v in range(1, int((current_time_m - max_time) / 60) + 1):
                data.append([int(v * 60 + max_time), last_bal])
            BALANCE_DF = pd.concat([BALANCE_DF, pd.DataFrame(data, columns=['Time', 'Balance'])], ignore_index=True)
        except FileNotFoundError:
            status = 'File Not Found'
        except KeyError:
            status = 'Key Error'
        if len(BALANCE_DF.index) > 525600:  # length of 1 year in min
            BALANCE_DF = BALANCE_DF[:525600]
        elif len(BALANCE_DF.index) < 525600:
            def_dat_gen = ((int(current_time_m - (60 * off)), 0, 0) for off in range(525600 - len(BALANCE_DF.index)))
            BALANCE_DF = pd.concat([pd.DataFrame(def_dat_gen, columns=['Time', 'Balance', 'Test Balance']), BALANCE_DF], axis=0,
                                   ignore_index=True)
        BALANCE_DF.sort_values('Time', inplace=True)
        BALANCE_DF['Test Balance'] = BALANCE_DF['Balance']
        BALANCE_DF.reset_index(drop=True, inplace=True)
        BALANCE_DF.to_csv(BALANCE_DF_PATH)
        if status != 'OK':
            CONF.total_profit = 0
            CONF.last_cap_bal = 0
        print_finish_thread(self.tag, self.st, [status])


def print_finish_thread(tag: str, start_time: float, args: list = None):
    """
    Print that the thread finished and how long it was running along with any additional
    arguments that were passed.
    """
    to_app = ''
    if args is not None and len(args) > 0:
        for arg in args:
            to_app += str(arg) + '\t'
    print(Bcolors.ENDC + f'{time.ctime()}\t' + Bcolors.OKGREEN +
          f'Finished: {tag}:\t\t' + Bcolors.ENDC + f'{to_app}Duration: {round(time.time() - start_time, 4)}sec')


def set_tk_var():
    """Set tkinter module global variables"""
    global TRADING_CRYPTO
    TRADING_CRYPTO = tk.IntVar()
    global MOVING_AVG_DICT
    MOVING_AVG_DICT = {key: tk.IntVar() for key in MOVING_AVG_DICT}
    global MOVING_AVG_BAL_DICT
    MOVING_AVG_BAL_DICT = {key: tk.IntVar() for key in MOVING_AVG_BAL_DICT}
    global GEN_MODEL_DICT
    GEN_MODEL_DICT = {key: tk.StringVar() for key in Symbols.all}
    global MODEL_INFO_DICT
    MODEL_INFO_DICT = {key: tk.StringVar() for key in Symbols.all}
    global PLOT_TYPE
    PLOT_TYPE = tk.StringVar()
    global BAL_PERIOD
    BAL_PERIOD = tk.StringVar()
    global DATA_PERIOD
    DATA_PERIOD = tk.StringVar()
    global TRAIN_SYMBOL
    TRAIN_SYMBOL = tk.StringVar()
    global LEVERAGE
    LEVERAGE = tk.StringVar()


def init(top, gui, *args, **kwargs):
    """init bybit_run with default variables and gui loops."""
    global w, top_level, root, CLIENT, BALANCE_DF, CONF, DATA_SIZE, EXCHANGE
    global POSITION, INDICATOR_LIGHT_COLORS, PREDICTION, TESTING_PROFIT
    w = gui
    CONF = Config()
    CLIENT = LocalBybit.set_client(CONF.key, CONF.secret)
    EXCHANGE = Exchange.set_client(key=CONF.key, secret=CONF.secret)
    POSITION = Position(symbol=CONF.def_coin)
    PREDICTION = 'hold'
    INDICATOR_LIGHT_COLORS = {'green': PhotoImage(file='bin/res/green_ind_round.png'),
                              'red': PhotoImage(file='bin/res/red_ind_round.png')}
    TESTING_PROFIT = 0
    BALANCE_DF = pd.DataFrame()
    safe_start_thread(AsyncBuildBalanceDf())
    top_level = top
    root = top
    set_gui_fields()
    refresh()
    minute_loop()
    api_change()


def set_gui_fields():
    """Populate gui objects with values from config file"""
    global CONF
    w.text_prediction['text'] = '-'
    w.rb_cont_arr[Symbols.all.index(POSITION.symbol)].invoke()
    w.entry_trade_amount.delete(0, 'end')
    w.entry_trade_amount.insert(0, CONF.trade_amount)
    w.entry_loss.delete(0, 'end')
    w.entry_loss.insert(0, CONF.loss_amount)
    w.entry_loss_period.delete(0, 'end')
    w.entry_loss_period.insert(0, CONF.loss_period)
    testing_toggle(toggle_to=CONF.testing)
    if CONF.plot_display in PlotTypes.all:
        w.combo_box_plot_type.set(CONF.plot_display)
    else:
        w.combo_box_plot_type.set(PlotTypes.price)
    if CONF.balance_plot_period in list(BALANCE_PERIODS_DICT.keys()):
        w.combo_box_bal_period.set(CONF.balance_plot_period)
    else:
        w.combo_box_bal_period.set(list(BALANCE_PERIODS_DICT.keys())[0])
    [w.ma_btns[k].select() for k, v in enumerate(list(CONF.plot_moving_avg)) if v == 1]
    model_args = CONF.default_model_arguments
    w.combo_box_target.current(Symbols.all.index(model_args.get('target')))
    w.entry_future_p.delete(0, 'end')
    w.entry_future_p.insert(0, model_args.get('future'))
    w.entry_data_size.delete(0, 'end')
    w.entry_data_size.insert(0, model_args.get('size'))
    w.combo_box_data_period.current(list(DataPeriods.all.values()).index(model_args.get('period')))
    w.entry_epoch.delete(0, 'end')
    w.entry_epoch.insert(0, model_args.get('epochs'))
    w.entry_seq_len.delete(0, 'end')
    w.entry_seq_len.insert(0, model_args.get('seq'))
    w.entry_batch_size.delete(0, 'end')
    w.entry_batch_size.insert(0, model_args.get('batch'))
    w.entry_ma.delete(0, 'end')
    w.entry_ma.insert(0, str(model_args.get('ma'))[1:-1].replace(' ', ''))
    w.entry_ema.delete(0, 'end')
    w.entry_ema.insert(0, str(model_args.get('ema'))[1:-1].replace(' ', ''))
    w.slider_leverage.set(CONF.leverage)
    LocalBybit.set_leverage(Symbols.all.index(model_args.get('target')), leverage=LEVERAGE.get())
    for layer in CONF.model_blueprint:
        model_struct_row(layer)
    update_model_gui_names()


def destroy_window():
    """Destroys window"""
    # Function which closes the window.
    global top_level
    top_level.destroy()
    top_level = None


def window_close():
    """
    Saves config balance_df to disk.
    End threads then calls destroy_window()
    """
    global IS_QUITTING, BALANCE_DF_PATH, w
    IS_QUITTING = True
    CONF.write_config()
    POSITION.cancel_trades()
    for th in THREAD_POOL:
        if th.tag == AsyncTrainModel.tag:
            th.kill()
    for t in THREAD_POOL:
        t.join()
    destroy_window()


def conv_currency_str(value: float):
    """Formats precision of currency to set decimal place"""
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
    Hides api passed and returns only last 8 characters for gui or 'ERROR'
    if passed value is less than 8 characters.
    """
    global w
    if val is None or len(val) < 8:
        return 'ERROR'
    else:
        return f'{"*" * (len(val) - 8) + val[-8:]}'


def safe_start_thread(t: callable):
    """
    Check is thread in running by checking if thread tag is in thread pool.
    Prints if thread has or hasn't started.
    """
    global THREAD_POOL
    out = ''
    if AsyncBuildBalanceDf.tag in [tag.tag for tag in THREAD_POOL]:
        out = Bcolors.ENDC + f'Still Building Balance. Thread: "{t.tag}"' + Bcolors.WARNING + ' NOT Started.' + Bcolors.ENDC
    if t.tag not in [tag.tag for tag in THREAD_POOL]:
        try:
            t.start()
            out = Bcolors.ENDC + f'Started Thread: "{t.tag}"' + Bcolors.OKGREEN + ' Safely.' + Bcolors.ENDC
            THREAD_POOL.append(t)
        except RuntimeError:
            pass
    else:
        out = Bcolors.ENDC + f'Thread: "{t.tag}"' + Bcolors.WARNING + ' NOT Started.'
    print(out)


def api_change():
    """
    To be called after changing the api key or secret.
    Update values for GUI as well as CLIENT and fetches new balance.
    """
    global MAIN_DF, CLIENT, EXCHANGE, w
    w.text_api_key['text'] = obfuscate_api_info(CONF.key)
    w.text_api_secret['text'] = obfuscate_api_info(CONF.secret)
    if obfuscate_api_info(CONF.key) == 'ERROR' or obfuscate_api_info(CONF.secret) == 'ERROR':
        trigger_error_bar(lines=['ERROR:  API Key/Secret', 'Try Updating API Key or Secret'], duration=10)
        return
    safe_start_thread(AsyncFetchBalance(key=CONF.key, secret=CONF.secret))
    CLIENT = LocalBybit.set_client(key=CONF.key, secret=CONF.secret)
    EXCHANGE = Exchange.set_client(key=CONF.key, secret=CONF.secret)
    safe_start_thread(AsyncDataHandler(fetch_plot_delay=1))


def minute_loop():
    """Main loop to call AsyncFetchBalance() thread and AsyncDataHandler() thread."""
    global TRADING_COUNT, CONF
    if REFRESH_COUNT > 5:
        safe_start_thread(AsyncFetchBalance(key=CONF.key, secret=CONF.secret))
        safe_start_thread(AsyncDataHandler(fetch_plot_delay=0.5))
    if not IS_QUITTING:
        root.after(int((60 - (time.time() % 60)) * 1000), lambda: minute_loop())
    else:
        TRADING_COUNT = 0


def refresh():
    """Main loop to refresh GUI elements and values."""
    global THREAD_POOL, TRADING_CRYPTO, REFRESH_COUNT, w, TRADING_COUNT, TESTING_PROFIT
    w.time_to_next_up['text'] = f'Update in: {int(60 - (time.time() % 60))}sec'
    if 'Loading' in w.btn_start_stop['text']:
        l_c = w.btn_start_stop['text'][w.btn_start_stop['text'].find('.'):]
        if len(l_c) >= 3:
            w.loading_plot_label['text'] = StartStopBtnText.loading
            w.btn_start_stop['text'] = StartStopBtnText.loading
        else:
            n_val = f'{" "}{w.btn_start_stop["text"]}{"."}'
            w.loading_plot_label['text'] = n_val
            w.btn_start_stop['text'] = n_val
    w.text_testing['text'] = str(bool(CONF.testing))
    w.text_pred_total['text'] = CONF.total_prediction
    w.text_pred_correct['text'] = CONF.correct_prediction
    try:
        w.text_pred_acc['text'] = f'{round(int(CONF.correct_prediction) / int(CONF.total_prediction) * 100, 2)}%'
    except ZeroDivisionError:
        w.text_pred_acc['text'] = '-%'
    w.text_total_profit['text'] = conv_currency_str(CONF.total_profit)
    if CONF.testing:
        w.text_testing_profit['text'] = conv_currency_str(TESTING_PROFIT)
    if IS_TRADING:
        if REFRESH_COUNT % 2 == 0:
            w.label_act_dact['text'] = 'Active'
            w.img_indicator_light.itemconfig(w.ind_img_container, image=INDICATOR_LIGHT_COLORS.get('green'))
        else:
            w.label_act_dact['text'] = ''
            w.img_indicator_light.itemconfig(w.ind_img_container, image=None)
    else:
        w.label_act_dact['text'] = 'Inactive'
        w.img_indicator_light.itemconfig(w.ind_img_container, image=INDICATOR_LIGHT_COLORS.get('red'))
    w.label_fees['text'] = f'Maker Fee: {POSITION.maker_fee}%    |    Taker Fee: {POSITION.taker_fee}%'
    if len(THREAD_POOL) > 0:
        THREAD_POOL = [t for t in THREAD_POOL if t.is_alive()]
    REFRESH_COUNT += 1
    if not IS_QUITTING:
        root.after(1000, lambda: refresh())


def trigger_error_bar(lines: list, duration: int, close=False):
    """When called triggers an error bar with contents of lines var"""
    global w
    if close:
        w.text_error_bar['text'] = ''
        return
    f_text = ''
    for l in lines:
        f_text += f'**{l}**\n'
    print(Bcolors.FAIL + f_text)
    w.text_error_bar['text'] = str(f_text)
    w.text_error_bar['foreground'] = '#880808'
    root.after(int(duration * 1000), lambda: trigger_error_bar(lines, duration, close=True))


def reset_stats():
    """Reset stats to and trade settings to default."""
    global w, TESTING_PROFIT, BALANCE_DF
    TESTING_PROFIT = 0
    BALANCE_DF['Test Balance'] = BALANCE_DF['Balance']
    CONF.trade_amount = 1
    w.entry_trade_amount.delete(0, 'end')
    w.entry_trade_amount.insert(0, CONF.trade_amount)
    CONF.loss_amount = 5
    w.entry_loss.delete(0, 'end')
    w.entry_loss.insert(0, CONF.loss_amount)
    CONF.loss_period = 10
    w.entry_loss_period.delete(0, 'end')
    w.entry_loss_period.insert(0, CONF.loss_period)
    CONF.total_prediction = 0
    CONF.correct_prediction = 0
    CONF.total_profit = 0
    CONF.last_cap_bal = 0
    CONF.write_config()


def change_trading_crypto():
    """Called when current trading crypto is changed to update price graph"""
    global TRADING_COUNT
    TRADING_COUNT = 0
    if (60 - (time.time() % 60)) > 10:
        safe_start_thread(AsyncFetchPlot(delay=4))


def save_settings():
    """Call when save settings button is clicked. Saves API key & secret to config then calls api_change()."""
    global w
    t_api_key = w.entry_api_key.get().strip()
    current_key = CONF.key
    t_api_secret = w.entry_api_secret.get().strip()
    current_secret = CONF.secret
    if not t_api_key or not t_api_secret or current_key == t_api_key or current_secret == t_api_secret:
        pass
    else:
        CONF.key = t_api_key
        CONF.secret = t_api_secret
        if os.path.exists(BALANCE_DF_PATH):
            os.remove(BALANCE_DF_PATH)
    safe_start_thread(AsyncBuildBalanceDf())
    w.entry_api_key.delete(0, 'end')
    w.entry_api_secret.delete(0, 'end')
    CONF.write_config()
    api_change()


def testing_toggle(toggle_to: bool = None):
    """When test button pressed Toggles test mode."""
    global TESTING_PROFIT, CONF
    if toggle_to is not None:
        CONF.testing = toggle_to
    else:
        CONF.testing = not bool(CONF.testing)
    if CONF.testing:
        w.btn_testing_on_off['text'] = 'Turn Off'
        w.label_testing_profit['text'] = 'Testing Profit:'
        w.text_testing_profit['text'] = conv_currency_str(TESTING_PROFIT)
    else:
        TESTING_PROFIT = 0
        reset_testing_balance()
        w.btn_testing_on_off['text'] = 'Turn On'
        w.label_testing_profit['text'] = ''
        w.text_testing_profit['text'] = ''


def place_moving_avg_btns(interface, var_set: dict, m_offset=0):
    """Place moving avg buttons on the gui"""
    offset = 85 + m_offset
    if m_offset == 0 and len(interface.ma_btns) > 0:
        [btn.place_forget() for btn in interface.ma_btns]
        interface.ma_btns = []
    if m_offset != 0:
        interface.label_break_line.place(x=offset - 26, rely=0.25, height=20, width=16)
        interface.label_break_line.configure(activebackground="#f9f9f9", activeforeground="black", anchor='c', background="#d9d9d9",
                                             justify='center', disabledforeground="#a3a3a3", font="-family {Segoe UI} -size 16",
                                             foreground="#000000", highlightbackground="#d9d9d9", highlightcolor="black", text=' | ')
    else:
        try:
            interface.label_break_line.place_forget()
        except AttributeError:
            pass
    for count, (ma_key, ma_var) in enumerate(var_set.items()):
        text = f'MA{ma_key}'
        if 'EMA' == ma_key:
            text = f'{ma_key}'
        x_pos = offset + (55 * count)
        chk_btn_ma = tk.Checkbutton(interface.info_frame)
        chk_btn_ma.place(x=x_pos, rely=0.25, height=20, width=55)
        chk_btn_ma.configure(command=lambda: moving_avg_check_btn(), text=text, variable=ma_var, background="#d9d9d9")
        interface.ma_btns.append(chk_btn_ma)
    if list(MOVING_AVG_BAL_DICT) == list(var_set):
        interface.combo_box_bal_period.place(x=6 + offset + (len(var_set) * 55), rely=0.25, height=20, width=80)
    else:
        interface.combo_box_bal_period.place_forget()


def moving_avg_check_btn():
    """
    Checks if moving ave buttons pressed and calls AsyncFetchPlot() as long as time
    is greater than 10 sec from refreshing.
    """
    CONF.plot_moving_avg = [x.get() for x in MOVING_AVG_DICT.values()]
    CONF.balance_moving_avg = [x.get() for x in MOVING_AVG_DICT.values()]
    if (60 - (time.time() % 60)) > 10:
        safe_start_thread(AsyncFetchPlot(delay=4))


def plot_combo_box(event):
    """
    Checks if combo box for plot type is changed and calls AsyncFetchPlot() as long as time
    is greater than 10 sec from refreshing.
    """
    CONF.plot_display = PLOT_TYPE.get()
    if (60 - (time.time() % 60)) > 10:
        safe_start_thread(AsyncFetchPlot(delay=4))
    w.info_frame.focus()


def bal_period_combo_box(event):
    """
    Checks if combo box for balance period is changed and calls AsyncFetchPlot() as long as time
    is greater than 10 sec from refreshing.

    """
    CONF.balance_plot_period = BAL_PERIOD.get()
    if (60 - (time.time() % 60)) > 10:
        safe_start_thread(AsyncFetchPlot(delay=4))
    w.info_frame.focus()


def gen_model_btn(symbol: str):
    """
    Moves to model training page and populates fields with models values if not none
    to be retrained on or adjusted.
    """
    global w, CONF
    model = CONF.models.get(symbol)
    w.TNotebook1.select(2)
    w.combo_box_target.current(Symbols.all.index(symbol))
    if model.model is not None:
        w.entry_future_p.delete(0, 'end')
        w.entry_future_p.insert(0, str(model.future_p))
        w.entry_seq_len.delete(0, 'end')
        w.entry_seq_len.insert(0, str(model.seq))
        w.entry_batch_size.delete(0, 'end')
        w.entry_batch_size.insert(0, str(model.batch))
        w.entry_ma.delete(0, 'end')
        w.entry_ma.insert(0, str(model.moving_avg)[1:-1].replace(' ', ''))
        w.entry_ema.delete(0, 'end')
        w.entry_ema.insert(0, str(model.e_moving_avg)[1:-1].replace(' ', ''))
        if model.layers is not None:
            # remove all old gui network layers and reset count
            [[child.destroy() for child in row] for row in w.model_layer_lst]
            w.model_layer_lst = []
            for layer in model.layers:
                model_struct_row(layer)


def get_model_info(symbol: str, display_gui=True):
    """
    returns a summary of the model.
    If 'display_gui=True' open a new window with model summary such as layers and
    types and number of nodes. has an option to display.
    """
    from gui import popup_bonus
    global CONF
    if CONF.models[symbol].model is None:
        print(Bcolors.WARNING + f'WARNING: Model: {symbol} not loaded')
        return
    model = CONF.models[symbol].model
    if not display_gui:
        model.summary()
        return
    total_params, p_neurons, count = (0, 0, 0)
    out_args = {'title': f'{symbol} Model', 'location': CONF.models[symbol].location, 'layers': []}
    while True:
        try:
            layer = model.get_layer(index=count)
            name = layer.name.capitalize() if layer.name[0] == 'd' else layer.name.upper()
            try:
                neurons = layer.units
            except AttributeError:
                neurons = layer.rate
            if count == 0:
                # get input shape and features.
                out_args['seq'] = layer.input_shape[1]
                out_args['features'] = layer.input_shape[2]
                no_params = neurons * layer.input_shape[2] + neurons
            else:
                no_params = neurons * p_neurons + neurons
            out_args['layers'].append([name, neurons, no_params])
            total_params += no_params
            p_neurons = neurons
            count += 1
        except ValueError:
            break
    out_args['total_param'] = total_params
    popup_bonus(**out_args)


def update_model_gui_names():
    """Updated model location in gui"""
    global w, CONF
    for count, model in enumerate(CONF.models.values()):
        loc = '...'
        if model.location is not None:
            loc = model.location.split('\\')[-1]
        w.text_model_name_arr[count]['text'] = loc
    CONF.write_config()


def start_btn():
    """
    Called when start button pressed.
    Resets default prediction.
    Updates button position so display btn_update_trades and changes text on start button to
    show stop.
    """
    global IS_TRADING, PREDICTION, POSITION, TRADING_CRYPTO
    POSITION = Position(symbol=Symbols.all[TRADING_CRYPTO.get()])
    IS_TRADING = not IS_TRADING
    if IS_TRADING:
        w.btn_update_trades.place(x=140, rely=0.870, height=60, width=130)
        w.btn_start_stop['text'] = StartStopBtnText.stop
        w.btn_start_stop.place(x=10, rely=0.870, height=60, width=130)
    else:
        AsyncPlaceTrade.end_trading(['TRADING HALTED', 'User request', 'Closing open trades'])
        w.btn_start_stop['text'] = StartStopBtnText.start
        w.btn_update_trades.place(x=10, rely=0.800, height=30, width=0)
        w.btn_start_stop.place(x=10, rely=0.870, height=60, width=260)


def model_struct_row(layer=None):
    """Populates rows in model training field on gui"""
    import gui
    if layer is not None:
        y_pos = 32 + (5 * len(w.model_layer_lst) - 1) + (20 * len(w.model_layer_lst) - 1)
        if len(w.model_layer_lst) >= 8:
            return
    else:
        if len(w.model_layer_lst) > 0:
            rm = w.model_layer_lst.pop()
            for child in rm:
                child.destroy()
            return
    combo_box = ttk.Combobox(w.FrameTraining2)
    combo_box.place(x=5, y=y_pos, height=19, width=70)
    combo_box.configure(background="#d9d9d9", font="-family {Segoe UI} -size 9")
    combo_box['values'] = list(LayerOptions.layer_dict)
    combo_box.current(list(LayerOptions.layer_dict).index(layer['layer']))
    combo_box['state'] = 'readonly'
    n_label = gui.struct_label(w.FrameTraining2, 'Neurons:', x=77, y=y_pos - 1, height=19, width=51, size=9)
    n_entry = gui.struct_entry(w.FrameTraining2, x=129, y=y_pos, height=19, width=32, size=10)
    n_entry.insert(0, layer['n'])
    d_label = gui.struct_label(w.FrameTraining2, 'Dropout:', x=162, y=y_pos - 1, height=19, width=50, size=9)
    d_entry = gui.struct_entry(w.FrameTraining2, x=213, y=y_pos, height=19, width=28, size=10)
    d_entry.insert(0, layer['drop'])
    w.model_layer_lst.append([combo_box, n_entry, d_entry, n_label, d_label])


def train_model():
    """
    Takes values entered on training page and places them into a dictionary.
    Then saves the model layout to config and begins training via AsyncTrainModel thread
    """
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
                if entry.get().strip() in Symbols.all:
                    data_args[args_key[count]] = entry.get().strip()
                else:
                    print(Bcolors.FAIL + f'ERROR: TargetError: Selected target not an option: {entry.get().strip()}')
                    return
            elif args_key[count] == 'period':
                data_args[args_key[count]] = DataPeriods.all.get(entry.get())
            elif count > len(w.training_fields) - 3:
                try:
                    data_args[args_key[count]] = [int(v.strip()) for v in entry.get().strip().split(',')]
                except ValueError:
                    data_args[args_key[count]] = []
            else:
                data_args[args_key[count]] = int(entry.get().strip())
        except ValueError:
            print(Bcolors.FAIL + f'ERROR: ValueError: Model training field: #{count} value: {entry.get()} type: {type(entry.get())}')
            return
    for k in args_key[1:-2]:
        if data_args[k] <= 0:
            print(Bcolors.WARNING + f'WARNING: Argument: "{k}" value: {data_args[k]} is invalid. Value must be greater than 0.')
            return
    CONF.default_model_arguments = data_args
    model_args = []
    for layer in w.model_layer_lst:
        try:
            sub_l = {}
            for count, ll in enumerate(layer[:3]):
                if count == 0:
                    if ll.get().strip() in list(LayerOptions.layer_dict):
                        sub_l['layer'] = ll.get().strip()
                    else:
                        print(Bcolors.FAIL + f'ERROR: LayerTypeError: layer type not acceptable: {ll.get().strip()}')
                        return
                elif count == 2:
                    f = float(ll.get().strip())
                    while f > 1:
                        f = f / 10
                    ll.delete(0, 'end')
                    ll.insert(0, f)
                    sub_l['drop'] = f
                else:
                    try:
                        sub_l['n'] = int(ll.get().strip())
                    except ValueError:
                        print(Bcolors.FAIL + f'ERROR: ValueError: layer value not acceptable: {ll.get().strip()}')
                        return
            model_args.append(sub_l)
        except ValueError:
            print(Bcolors.FAIL + f'ERROR: ValueError: value: {ll.get()} model layer: {str(layer)}')
            return
    if len(model_args) <= 0:
        print(Bcolors.WARNING + f'WARNING: Model Layout Invalid. Size: {len(model_args)} must be greater than 0.')
        return
    CONF.model_blueprint = model_args
    CONF.write_config()
    verbose = 2 if ('selected' in w.checkbutton_verbose.state() or 'alternate' in w.checkbutton_verbose.state()) else 1
    force_new = True if ('selected' in w.checkbutton_new_data.state() or 'alternate' in w.checkbutton_new_data.state()) else False
    w.btn_start_train['text'] = 'Stop Training'
    safe_start_thread(AsyncTrainModel(data_args=data_args, model_args=model_args, verbose=verbose, force_new=force_new))


def reset_testing_balance():
    """Resets testing balance dataframe to balance dataframe."""
    global BALANCE_DF
    BALANCE_DF['Test Balance'] = BALANCE_DF['Balance']


def delete_model(symbol: str):
    """Delete model from disk and clears gui label when 'delete' button is pressed"""
    CONF.models[symbol].clear()
    update_model_gui_names()


# GLOBAL VARS
IS_TRADING = False  # if se tot trading mode or idle
HARD_TESTING_LOCK = True  # software lock to prevent accidental trading while testing
IS_QUITTING = False  # when set to true to start the shutdown process.
TRADING_COUNT, REFRESH_COUNT, TICK = 0, 0, 0  # counts for gui ticks
BALANCE_DF_PATH = 'bin/res/b_df.csv'  # user balance dataframe
VERSION = '1.0.0'  # version number
TESTING_PROFIT = 0
DATA_SIZE = Config.DATA_SIZE
BALANCE_PERIODS_DICT = {'10 min': 10, '60 Min': 60, '1 Day': 1440, '10 Day': 14400, '1 Month': 43800, '3 Month': 131400,
                        '6 Month': 262800, '1 Year': 525600}  # periods in graph that can be displayed
MOVING_AVG_DICT = {5: 0, 10: 0, 20: 0, 30: 0, 'EMA': 0}  # moving average options for price = 0 off, 1 on
MOVING_AVG_BAL_DICT = {10: 0, 30: 0}  # moving option averages for price 0 off, 1 on
THREAD_POOL = []  # pool of all active threads
POSITION = Position()
MAIN_DF = pd.DataFrame()  # dataframe holding the coin price data
BALANCE_DF = pd.DataFrame()  # dataframe holding the user balance price data
no_buy, no_sell = 0, 0

if __name__ == '__main__':
    exit('START BY RUNNING "gui.py"')
