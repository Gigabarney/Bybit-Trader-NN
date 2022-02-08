import concurrent.futures
import time
import os.path
from tqdm import tqdm
import pandas as pd
from bybit import bybit
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import selenium.common.exceptions
from selenium.webdriver.chrome.options import Options
import config
from collections import deque
import nltk
from dateutil import parser
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np

SYMBOLS = ['BTC', 'ETH', 'XRP', "EOS"]
ITERATIONS = 1
DATA_PERIOD = 1
THREADS = 1
QUICK = False
PRICE_SPREAD = 5


def clean_data(data):
    columns = []
    dataset = []
    for count, itemDict in enumerate(data):
        temp = []
        for key, val in itemDict.items():
            if key not in ['symbol', 'interval']:
                temp.append(val)
                if count == 0:
                    columns.append(key)
        dataset.append(temp)
    return dataset, columns


def pull_crypto(client, sym, itter, prog_bar):
    # amount of data = 200(items) * itter(5) = 1000 *
    # amount of ratio = 5min intervul max 500 items
    temp = pd.DataFrame()
    for count in range(itter):
        prog_bar.update(1 * 200)
        time_offset = (200 * (count + 1)) * (60 * DATA_PERIOD)
        try:
            dataset, columns = clean_data(client.Kline.Kline_get(symbol=f'{sym}USD', interval=f"{str(DATA_PERIOD)}",
                                                                 **{'from': int(float(
                                                                     client.Common.Common_getTime().result()[0][
                                                                         'time_now'])) - (
                                                                                time_offset)}).result()[0]['result'])
        except ConnectionError:
            time.sleep(0.5)
            dataset, columns = clean_data(client.Kline.Kline_get(symbol=f'{sym}USD', interval="1",
                                                                 **{'from': int(float(
                                                                     client.Common.Common_getTime().result()[0][
                                                                         'time_now'])) - (
                                                                                time_offset)}).result()[0]['result'])
        time.sleep(0.05)
        # ['open_time', 'open', 'high', 'low', 'close', 'volume', 'turnover']   COLUMNS
        for c, col in enumerate(columns):
            if col != 'open_time':
                columns[c] = f'{sym}_{col}'
        if len(temp) == 0:
            temp = pd.DataFrame(dataset, columns=columns)
        else:
            temp = temp.append(pd.DataFrame(dataset, columns=columns), ignore_index=True)
    temp.rename(columns={'open_time': 'time'}, inplace=True)
    # temp.set_index('time', inplace=True)
    return temp


def get_crypto(iters, file_name):
    try:
        df = pd.read_csv(file_name, index_col=0)
        df.sort_values('time')
        print(f'Loaded crypto file: {file_name}  ...\n\tItems: {len(df.index)}\n')
    except FileNotFoundError:
        print(f'"{file_name}" Not Found.\n\tCreating New Data File')
        client = bybit(test=False, api_key=config.api_key, api_secret=config.api_secret)
        df = pd.DataFrame()
        time.sleep(1)
        with tqdm(total=iters * len(SYMBOLS) * 200, unit=' Price Iterations') as crypto_prog_bar:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(pull_crypto, client, symbol, iters, crypto_prog_bar) for symbol in SYMBOLS]
                for f in concurrent.futures.as_completed(futures):
                    if len(df.index) == 0:
                        df = pd.DataFrame(f.result().set_index('time'))
                    else:
                        df = df.join(f.result().set_index('time'))

        df = df.astype(float)
        df['time'] = df.index
        df.reset_index(drop=True, inplace=True)
        df.sort_values(by=['time'], inplace=True)
        df.to_csv(path_or_buf=file_name)
        print(f'Created crypto file: {file_name}  ...\n\tItems: {len(df.index)}\n')
    return df


def pull_trading_records(client, sym: str, oldest_time, offset=None):
    global trading_r_master_count, display_info
    tz_offset = time.timezone
    pulled = {}
    count = 0
    error_count = 0
    arguments = {'symbol': f'{sym}USD', 'limit': 1000, 'from': 0}
    trading_record_loop = True
    error_loop = True
    while trading_record_loop:
        result = []
        while error_loop:
            try:
                result = client.Market.Market_tradingRecords(**arguments).result()[0]['result']
                if result is not None:
                    break
                else:
                    raise ConnectionError
            except ConnectionError:
                error_count += 1
                if error_count >= 25:
                    time.sleep(60)
                time.sleep(0.5)
        if arguments['from'] != 0:
            result.reverse()
        # arguments['from'] = result[-1]['id'] - arguments['limit']
        trading_r_master_count += arguments['limit']
        if offset is not None and count >= offset[1]:
            arguments['from'] = result[-1]['id'] - (arguments['limit'] * offset[0])
        else:
            arguments['from'] = result[-1]['id'] - arguments['limit']

        for cnt, r in enumerate(result):
            parsed_time = int(time.mktime(parser.parse(r['time'], fuzzy=True).timetuple())) - tz_offset
            rounded_time = parsed_time - (parsed_time % 60)
            if cnt == 0:
                display_info[sym] = parsed_time - oldest_time
                print(f'\r{sym}  dif: {parsed_time - oldest_time}     ', end=' ')
            if parsed_time > oldest_time:
                if rounded_time not in pulled.keys():
                    pulled[rounded_time] = {'Buy': {}, 'Sell': {}}

                if r['price'] not in pulled[rounded_time][r['side']].keys():
                    pulled[rounded_time][r['side']][r['price']] = 0
                pulled[rounded_time][r['side']][r['price']] += r['qty']
            else:
                trading_record_loop = False
                break
        count += 1
    print(f'\nparsing {sym} data...')
    columns = ['time', *(f'{sym}_Buy_{c}' for c in range(PRICE_SPREAD - 1, -1, -1)), *(f'{sym}_Sell_{c2}' for c2 in range(PRICE_SPREAD))]
    temp_df = pd.DataFrame()
    for item in pulled.items():
        buys = [item[1]['Buy'][i] for i in sorted(item[1]['Buy'], reverse=True)][-PRICE_SPREAD:]
        while len(buys) < 5:
            buys.insert(0, 0)
        sells = [item[1]['Sell'][i] for i in sorted(item[1]['Sell'], reverse=False)][:PRICE_SPREAD]
        while len(sells) < 5:
            sells.append(0)
        # n_buy_sell = [(float(i) / sum([*buys, *sells])) for i in [*buys, *sells]]  # normalize data in row
        data = [[item[0], *buys, *sells]]
        if len(temp_df.index) == 0:
            temp_df = pd.DataFrame(data, columns=columns)
        else:
            temp_df = temp_df.append(pd.DataFrame(data, columns=columns))
    temp_df.sort_values(by=['time'], inplace=True)
    return sym, temp_df


def get_trading_records(file_name, oldest_date):
    global trading_r_master_count, display_info
    display_info = {s: 0 for s in SYMBOLS}
    trading_r_master_count = 0
    try:
        df = pd.read_csv(file_name, index_col=0)
        df.sort_values('time')
        print(f'Loaded Trading Record file: {file_name}  ...\n\tItems: {len(df.index)}\n')
    except FileNotFoundError:
        client = bybit(test=False, api_key=config.api_key, api_secret=config.api_secret)
        df = pd.DataFrame()

        extra_th_symbols = ['BTC', 'ETH']
        extra_dfs = {s: pd.DataFrame() for s in extra_th_symbols}
        time.sleep(1)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for symbol in SYMBOLS:
                offsets = list(range(int(THREADS)))
                if symbol in extra_th_symbols and len(offsets) > 0:
                    for offset in offsets:
                        futures.append(executor.submit(pull_trading_records, client, symbol, int(oldest_date), [len(offsets), offset]))
                else:
                    futures.append(executor.submit(pull_trading_records, client, symbol, int(oldest_date)))
            for f in concurrent.futures.as_completed(futures):
                if f.result()[0] in extra_th_symbols:
                    if extra_dfs[f.result()[0]] is None or len(extra_dfs[f.result()[0]]) == 0:
                        extra_dfs[f.result()[0]] = pd.DataFrame(f.result()[-1].set_index('time'))
                    else:
                        extra_dfs[f.result()[0]] = extra_dfs[f.result()[0]].append(f.result()[-1].set_index('time'))
                else:
                    if len(df.index) == 0:
                        df = pd.DataFrame(f.result()[-1].set_index('time'))
                    else:
                        df = df.join(f.result()[-1].set_index('time'))
        for extra_df_val in extra_dfs.values():
            if len(df.index) == 0:
                df = extra_df_val
            else:
                df = df.join(extra_df_val)
        df['time'] = df.index
        df.reset_index(drop=True, inplace=True)
        df.sort_values(by=['time'], inplace=True)
        df.to_csv(path_or_buf=file_name)
        print(f'\nCreated Trading Record file: {file_name}  ...\n\tItems: {len(df.index)}\n')
    return df


def merge_df(df1, df2):
    return pd.merge_asof(df1.sort_values('time'), df2.sort_values('time'), on='time',
                         direction='nearest', tolerance=(60 * DATA_PERIOD) - 1)


def get_data(iteration_count: int = ITERATIONS, data_period: int = DATA_PERIOD, threads: int = THREADS, force_new: bool = False,
             price_spread: int = PRICE_SPREAD):
    global ITERATIONS  # Number or loops of data go get from ByBit. 1 iteration is 60 min worth or prices
    ITERATIONS = iteration_count
    global THREADS  # Number fo threads to use when pulling article and parsing article data
    THREADS = threads
    global PRICE_SPREAD
    PRICE_SPREAD = price_spread
    global DATA_PERIOD
    p = [1, 3, 5, 15, 30, 60, 120, 240, 360, 720]
    if data_period not in p:
        print(f'\n\t*** Value: {data_period} is invalid.'
              f'\n\tValue must be as follows: {p}'
              f'\n\t"date_period" set to: 1')
        data_period = 1
    DATA_PERIOD = data_period

    crypto_file = f'data\\crypto_data_{"-".join(SYMBOLS)}_iter_{ITERATIONS}.csv'
    trading_r_file = f'data\\trading_records-{ITERATIONS}.csv'
    news_file = f'data\\news_data_{ITERATIONS}.csv'
    whale_alert_file = f'data\\whale_alert_{ITERATIONS}.csv'
    merge_file = f'data\\FINAL_CRYPTO_{ITERATIONS}.csv'

    if os.path.isfile(merge_file) and not force_new:
        final_crypto = pd.read_csv(merge_file, dtype={'text': str, 'amount': str, 'trans': str}, index_col=0)
        final_crypto.sort_index(inplace=True)
        print(f'Loaded FINAL data file: {merge_file}\n\n')
        return SYMBOLS, final_crypto
    else:
        final_df = get_crypto(file_name=crypto_file, iters=ITERATIONS)

        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #    article_futures = [executor.submit(get_trading_records, trading_r_file, final_df['time'][0])]
        #    for f in concurrent.futures.as_completed(article_futures):
        #        final_df = merge_df(final_df, f.result())

        final_df.drop_duplicates(subset=['time'], inplace=True)
        final_df.set_index('time', inplace=True)
        final_df.sort_index(inplace=True)
        final_df.to_csv(path_or_buf=merge_file)
        return SYMBOLS, final_df


if __name__ == '__main__':
    get_data()
