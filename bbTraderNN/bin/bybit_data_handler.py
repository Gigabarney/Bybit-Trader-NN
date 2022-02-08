import concurrent.futures
import pandas as pd
from dateutil import parser
import time
from sklearn import preprocessing
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class Handler:
    def __init__(self, current_data, client, symbols, size):
        self.current_data = current_data
        self.client = client
        self.symbols = symbols
        self.size = size

    def pop_test_data(self, tick):
        data = pd.read_csv('data/FINAL_CRYPTO_500.csv', index_col=0)
        if len(data.index) < self.size + tick:
            tick = 0
        data = data.iloc[tick, self.size + tick]
        tick += 1
        return tick, data

    def get_data(self):
        if len(self.current_data.index) == 0:
            # fill current data full up to size requested
            crypto_price = self._get_crypto(self.size)
            # trading_records = self._get_trading_records(crypto_price['time'][0])
        else:
            crypto_price = self._get_crypto(1)
            # trading_records = self._get_trading_records(self.current_data.index[-1])
        # final_df = pd.merge_asof(crypto_price.sort_values('time'), trading_records.sort_values('time'), on='time',
        #                         direction='nearest', tolerance=59)
        final_df = crypto_price
        if len(self.current_data.index) == 0:
            final_df.set_index('time', inplace=True)
            final_df.sort_index(inplace=True)
            final_df.dropna(inplace=True)
            self.current_data = final_df
        else:
            self.current_data = pd.concat([self.current_data, final_df.set_index('time')])

        if len(self.current_data.index) > self.size:
            self.current_data.drop(self.current_data.head(len(self.current_data.index) - self.size).index, inplace=True)

        if len(self.current_data.index) < self.size:
            self.current_data = pd.concat([self.current_data,
                                           pd.DataFrame(data=[[0 for _ in range(len(self.current_data.columns))]
                                                              for __ in range(self.size - len(self.current_data.index))],
                                                        columns=self.current_data.columns)])
        return self.current_data

    def _get_crypto(self, no_items):
        df = pd.DataFrame()
        api_calls_made = 0
        delta_times = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self._pull_crypto, self.client, s, no_items) for s in self.symbols]
            for f in concurrent.futures.as_completed(futures):
                api_calls_made += f.result()[1]
                delta_times.append(f.result()[2])
                if len(df.index) == 0:
                    df = pd.DataFrame(f.result()[0].set_index('time'))
                else:
                    df = df.join(f.result()[0].set_index('time'))
        df = df.astype(float)
        df['time'] = df.index
        df.reset_index(drop=True, inplace=True)
        df.sort_values(by=['time'], inplace=True)
        avg_delta_time = (sum(delta_times) / len(delta_times))
        print(f'Crypto Price:\t{api_calls_made} API call(s) in {round(avg_delta_time, 2)}sec'
              f'  ({round((api_calls_made / avg_delta_time) * 60, 2)}/min)  ({round(api_calls_made / avg_delta_time, 2)}/sec)')
        return df

    def _pull_crypto(self, client, sym, itter):
        temp = pd.DataFrame()
        dataset = []
        columns = []
        api_calls_made = 0
        p_c_start_time = time.time()
        if itter < 200:
            try:
                dataset, columns = self._clean_data(client.Kline.Kline_get(symbol=f'{sym}USD', interval="1", **{'from': int(float(
                    client.Common.Common_getTime().result()[0]['time_now'])) - (itter * 60)}).result()[0]['result'])
                api_calls_made += 1
            except ConnectionError:
                time.sleep(0.5)
                dataset, columns = self._clean_data(client.Kline.Kline_get(symbol=f'{sym}USD', interval="1", **{'from': int(float(
                    client.Common.Common_getTime().result()[0]['time_now'])) - (itter * 60)}).result()[0]['result'])
                api_calls_made += 1
        else:
            for count in range(itter):
                time_offset = (200 * (count + 1)) * 60
                try:
                    dataset, columns = self._clean_data(client.Kline.Kline_get(symbol=f'{sym}USD', interval="1",
                                                                               **{'from': int(float(
                                                                                   client.Common.Common_getTime().result()[0][
                                                                                       'time_now'])) - time_offset}).result()[0]['result'])
                    api_calls_made += 1
                except ConnectionError:
                    time.sleep(0.5)
                    dataset, columns = self._clean_data(client.Kline.Kline_get(symbol=f'{sym}USD', interval="1",
                                                                               **{'from': int(float(
                                                                                   client.Common.Common_getTime().result()[0][
                                                                                       'time_now'])) - time_offset}).result()[0]['result'])
                    api_calls_made += 1
        p_c_time_delta = time.time() - p_c_start_time
        # ['open_time', 'open', 'high', 'low', 'close', 'volume', 'turnover']   COLUMNS
        for c, col in enumerate(columns):
            if col != 'open_time':
                columns[c] = f'{sym}_{col}'
        if len(temp) == 0:
            temp = pd.DataFrame(dataset, columns=columns)
        else:
            temp = temp.append(pd.DataFrame(dataset, columns=columns))
        temp.rename(columns={'open_time': 'time'}, inplace=True)
        # temp.set_index('time', inplace=True)
        return temp, api_calls_made, p_c_time_delta

    @staticmethod
    def _clean_data(data):
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

    def _get_trading_records(self, oldest_date):
        df = pd.DataFrame()
        api_calls_made = 0
        delta_times = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for s in self.symbols:
                futures.append(executor.submit(self._pull_trading_records, self.client, s, int(oldest_date)))
            for f in concurrent.futures.as_completed(futures):
                api_calls_made += f.result()[1]
                delta_times.append(f.result()[2])
                if len(df.index) == 0:
                    df = pd.DataFrame(f.result()[0].set_index('time'))
                else:
                    df = df.join(f.result()[0].set_index('time'))
        df['time'] = df.index
        df.reset_index(drop=True, inplace=True)
        df.sort_values(by=['time'], inplace=True)
        avg_delta_time = (sum(delta_times) / len(delta_times))
        print(f'Trading Records:\t{api_calls_made} API call(s) in {round(avg_delta_time, 2)}sec'
              f'  ({round((api_calls_made / avg_delta_time) * 60, 2)}/min)  ({round(api_calls_made / avg_delta_time, 2)}/sec)')
        return df

    def _pull_trading_records(self, client, sym: str, oldest_time):
        tz_offset = time.timezone
        api_calls_made = 0
        pulled = {}
        count = 0
        error_count = 0
        arguments = {'symbol': f'{sym}USD', 'limit': 1000, 'from': 0}
        trading_record_loop = True
        error_loop = True
        p_t_start_time = time.time()
        while trading_record_loop:
            result = []
            while error_loop:
                try:
                    r_ = client.Market.Market_tradingRecords(**arguments).result()
                    # print(r_[1])
                    result = r_[0]['result']
                    api_calls_made += 1
                    break
                except ConnectionError:
                    error_count += 1
                    if error_count >= 25:
                        raise ConnectionError
                    time.sleep(0.5)
            if arguments['from'] != 0:
                result.reverse()
            arguments['from'] = result[-1]['id'] - arguments['limit']

            for r in result:
                parsed_time = int(time.mktime(parser.parse(r['time'], fuzzy=True).timetuple())) - tz_offset
                rounded_time = parsed_time - (parsed_time % 60)
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
        p_t_time_delta = time.time() - p_t_start_time
        columns = ['time', *(f'{sym}_Buy_{c}' for c in range(self.price_spread - 1, -1, -1)),
                   *(f'{sym}_Sell_{c2}' for c2 in range(self.price_spread))]
        temp_df = pd.DataFrame()
        for item in pulled.items():
            buys = [item[1]['Buy'][i] for i in sorted(item[1]['Buy'], reverse=True)][-self.price_spread:]
            while len(buys) < 5:
                buys.insert(0, 0)
            sells = [item[1]['Sell'][i] for i in sorted(item[1]['Sell'], reverse=False)][:self.price_spread]
            while len(sells) < 5:
                sells.append(0)
            # n_buy_sell = [(float(i) / sum([*buys, *sells])) for i in [*buys, *sells]]  # normalize data in row
            data = [[item[0], *buys, *sells]]
            if len(temp_df.index) == 0:
                temp_df = pd.DataFrame(data, columns=columns)
            else:
                temp_df = pd.concat([temp_df, pd.DataFrame(data, columns=columns)])
        temp_df.sort_values(by=['time'], inplace=True)
        return temp_df, api_calls_made, p_t_time_delta


def preprocess_data(df, data_mod_args: dict):
    # use data_mod_args to change size of data to correct size:: "df[-size:]"
    df.drop(columns=[col for col in df.columns if 'turnover' in col], inplace=True)
    if not data_mod_args['drop_sym']:
        for to_drop in data_mod_args['drop_sym']:
            df.drop(columns=[col for col in df.columns if to_drop in col], inplace=True)

    if len(data_mod_args['ma']) > 0:
        for col in df.columns:
            if '_close' in col:
                for m in data_mod_args['ma']:
                    df[f'{col}_MA{m}'] = df[col].rolling(window=m).mean()
                    if len(data_mod_args['ema']) > 0:
                        for e in data_mod_args['ema']:
                            df[f'{col}_{m}EMA{e}'] = df[f'{col}_MA{m}'].ewm(span=e).mean()
    for col in df.columns:
        if col not in ['time']:
            df[col] = df[col].pct_change()
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df[col].dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)
        df[col] = pd.to_numeric(df[col], downcast='integer')
        df[col] = pd.to_numeric(df[col], downcast='float')

    df.dropna(inplace=True)
    df.sort_index(inplace=True)
    # df = df[-data_mod_args['size']:]
    chunk = []
    # chunk2 = [[n for n in i] for i in df.values]
    # chunk2 = chunk2[-data_mod_args['size']:]
    print(df.columns)
    print(len(df.columns))
    for i in df.values:
        chunk.append([n for n in i])
        if len(chunk) >= data_mod_args['size']:
            break
    return chunk
    # return df


def _process_trading_records(df, symbols, price_spread):
    pd.options.mode.chained_assignment = None
    d_columns = [f'{s}_{d}_{i}' for s in symbols for d in ['Buy', 'Sell'] for i in range(price_spread)]
    temp_df = df[[f'{s}_{d}_{i}' for s in symbols for d in ['Buy', 'Sell'] for i in range(price_spread)]]
    temp_df.replace(np.nan, 0, inplace=True)
    for sym in symbols:
        columns = [*(f'{sym}_Buy_{c}' for c in range(price_spread - 1, -1, -1)), *(f'{sym}_Sell_{c2}' for c2 in range(price_spread))]
        for i, row in temp_df[columns].iterrows():
            try:
                temp_df.at[i, columns] = [(float(x) / sum(row)) for x in row]
            except ZeroDivisionError:
                pass
    temp_df = temp_df.apply(pd.to_numeric, downcast='float')
    temp_df.index = df.index
    df.drop(columns=d_columns, inplace=True)
    df = pd.concat([df, temp_df], axis=1)
    return df
