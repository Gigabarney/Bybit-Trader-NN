import concurrent.futures
from bin import networkTraining
import time
import os.path
from tqdm import tqdm
import pandas as pd
from bybit import bybit


class Constructor:
    symbols = ['BTC', 'ETH', 'XRP', "EOS"]

    def __init__(self, outer_instance: networkTraining.Training, data_s=1, data_p=1, threads=1,
                 client=None, key=None, secret=None, force_new: bool = False, save_file=True):
        self.outer_instance = outer_instance
        self.client = client
        if self.client is None:
            self.client = bybit(test=False, api_key=key, api_secret=secret)
        self.data_size = data_s
        if outer_instance is not None:
            self.verbose = outer_instance.verbose
        else:
            self.verbose = 0
        self.data_period = data_p
        self.threads = threads
        self.force_new = force_new
        self.save_file = save_file

    def v_print(self, text=None, max_val=0, update=0, reset=False):
        if self.outer_instance is not None:
            self.outer_instance.verbose_print(text, max_val, update, reset)

    def get_data(self):
        crypto_file = f'data\\t_crypto_data_{"-".join(self.symbols)}_iter_{self.data_size}.csv'
        merge_file = f'data\\FINAL_CRYPTO_{self.data_size}_P{self.data_period}.csv'

        if os.path.isfile(merge_file) and self.force_new is False:
            final_crypto = pd.read_csv(merge_file, dtype={'text': str, 'amount': str, 'trans': str}, index_col=0)
            final_crypto.sort_index(inplace=True)
            self.v_print(f'Loaded FINAL data file: {merge_file}\n\n', max_val=-1)
            return final_crypto
        else:
            final_df = self.get_crypto(file_name=crypto_file)
            final_df.drop_duplicates(subset=['time'], inplace=True)
            final_df.set_index('time', inplace=True)
            final_df.sort_index(inplace=True)
            try:
                os.remove(merge_file)
            except FileNotFoundError:
                pass
            if self.save_file:
                final_df.to_csv(path_or_buf=merge_file)
            return final_df

    def get_crypto(self, file_name=None):
        if file_name is not None:
            try:
                df = pd.read_csv(file_name, index_col=0)
                df.sort_values('time', inplace=True)
                self.v_print(f'Loaded crypto file: {file_name}  ...\n\tItems: {len(df.index)}\n', max_val=-1)
                return df
            except FileNotFoundError:
                self.v_print(f'Creating New Data', max_val=1)
        df = pd.DataFrame()
        self.v_print(f'Building {self.data_size * len(self.symbols) * 200} Datapoints',
                     max_val=self.data_size * len(self.symbols) * 200, reset=True)
        time.sleep(0.1)
        with tqdm(total=self.data_size * len(self.symbols) * 200, unit=' Price Iterations',
                  disable=not self.verbose > 1) as crypto_prog_bar:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self._pull_crypto, self.client, symbol, self.data_size, crypto_prog_bar) for symbol in
                           self.symbols]
                for f in concurrent.futures.as_completed(futures):
                    if len(df.index) == 0:
                        df = pd.DataFrame(f.result().set_index('time'))
                    else:
                        df = df.join(f.result().set_index('time'))
        df = df.astype(float)
        df['time'] = df.index
        df.reset_index(drop=True, inplace=True)
        df.sort_values(by=['time'], inplace=True)
        self.v_print(f'Created crypto file: {file_name}  ...\n\tItems: {len(df.index)}\n', max_val=-1)
        return df

    def _pull_crypto(self, client, sym, itter, prog_bar):
        # amount of data = 200(items) * itter(5) = 1000 *
        # amount of ratio = 5min intervul max 500 items
        temp = pd.DataFrame()
        for count in range(itter):
            self.v_print('', update=200)
            prog_bar.update(200)
            time_offset = (200 * (count + 1)) * (60 * self.data_period)
            try:
                dataset, columns = self._clean_data(client.Kline.Kline_get(symbol=f'{sym}USD', interval=f"{str(self.data_period)}",
                                                                           **{'from': int(float(
                                                                               client.Common.Common_getTime().result()[0][
                                                                                   'time_now'])) - (
                                                                                          time_offset)}).result()[0]['result'])
            except ConnectionError:
                time.sleep(0.5)
                dataset, columns = self._clean_data(client.Kline.Kline_get(symbol=f'{sym}USD', interval="1",
                                                                           **{'from': int(float(
                                                                               client.Common.Common_getTime().result()[0][
                                                                                   'time_now'])) - (
                                                                                          time_offset)}).result()[0]['result'])
            for c, col in enumerate(columns):
                if col != 'open_time':
                    columns[c] = f'{sym}_{col}'
            temp = pd.concat([temp, pd.DataFrame(dataset, columns=columns)])
        temp.rename(columns={'open_time': 'time'}, inplace=True)
        return temp

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
