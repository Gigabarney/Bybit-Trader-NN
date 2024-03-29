import sys
import tkinter as tk
import tkinter.ttk as ttk
from bin import bybitRun as bybit_gui_support
import os.path

root = None


def vp_start_gui():
    '''Starting point when module is the main routine.'''
    global val, w, root
    global prog_location
    prog_call = sys.argv[0]
    prog_location = os.path.split(prog_call)[0]
    root = tk.Tk()
    bybit_gui_support.set_tk_var()
    top = Toplevel1(root)
    bybit_gui_support.init(root, top)
    root.protocol('WM_DELETE_WINDOW', bybit_gui_support.window_close)
    root.iconphoto(False, tk.PhotoImage(file='bin/res/icon.png'))
    root.mainloop()


w = None


def create_Toplevel1(rt, *args, **kwargs):
    '''Starting point when module is imported by another module.
       Correct form of call: 'create_Toplevel1(root, *args, **kwargs)' .'''
    global w, w_win, root
    global prog_location
    prog_call = sys.argv[0]
    prog_location = os.path.split(prog_call)[0]
    root = rt
    w = tk.Toplevel(root)
    bybit_gui_support.set_tk_var()
    top = Toplevel1(w)
    bybit_gui_support.init(w, top, *args, **kwargs)
    return w, top


def destroy_Toplevel1():
    global w
    w.destroy()
    w = None


def struct_label(parent, text, x=None, y=None, height=None, width=None, relx=None, rely=None, relheight=None, relwidth=None,
                 justify='left', background="#d9d9d9", size=0, bold=False, relief='flat'):
    jf = {'left': 'w', 'right': 'e', 'center': 'c'}
    font = '-family {Segoe UI}' + f' -size {size}'
    if size == 0:
        font = None
    if bold:
        font += ' -weight bold'
    label = ttk.Label(parent, background=background, font=font, relief=relief, justify=justify, anchor=jf.get(justify),
                      text=text)
    # foreground="#000000"
    c = {'x': x, 'y': y, 'height': height, 'width': width,
         'relx': relx, 'rely': rely, 'relheight': relheight, 'relwidth': relwidth}
    label.place(**{k: v for k, v in c.items() if v is not None})
    return label


def struct_frame(parent, x=None, y=None, height=None, width=None, relx=None, rely=None, relheight=None, relwidth=None,
                 padx="0", pady="0", relief='groove', borderwidth="0", bg="#d9d9d9", hl_bg="#d9d9d9", hl_color="black"):
    frame = tk.Frame(parent, relief=relief, borderwidth=borderwidth, background=bg,
                     highlightbackground=hl_bg, highlightcolor=hl_color, padx=padx, pady=pady)
    c = {'x': x, 'y': y, 'height': height, 'width': width,
         'relx': relx, 'rely': rely, 'relheight': relheight, 'relwidth': relwidth}
    frame.place(**{k: v for k, v in c.items() if v is not None})
    return frame


def struct_entry(parent, x=None, y=None, height=None, width=None, relx=None, rely=None, relheight=None, relwidth=None, size=0):
    font = "-family {Segoe UI}"
    if size == 0:
        font = None
    else:
        font += f' -size {size}'
    entry = ttk.Entry(parent, takefocus="", cursor="ibeam", font=font)
    c = {'x': x, 'y': y, 'height': height, 'width': width,
         'relx': relx, 'rely': rely, 'relheight': relheight, 'relwidth': relwidth}
    entry.place(**{k: v for k, v in c.items() if v is not None})
    return entry


def struct_button(parent, text, x=None, y=None, height=None, width=None, relx=None, rely=None, relheight=None, relwidth=None, command=None):
    btn = ttk.Button(parent, takefocus='', text=text, command=command)
    c = {'x': x, 'y': y, 'height': height, 'width': width,
         'relx': relx, 'rely': rely, 'relheight': relheight, 'relwidth': relwidth}
    btn.place(**{k: v for k, v in c.items() if v is not None})
    return btn


class Toplevel1:
    def __init__(self, top=None):
        '''This class configures and populates the toplevel window.
           top is the toplevel containing window.'''
        _bgcolor = '#d9d9d9'  # X11 color: 'gray85'
        _fgcolor = '#000000'  # X11 color: 'black'
        _compcolor = '#d9d9d9'  # X11 color: 'gray85'
        _ana1color = '#d9d9d9'  # X11 color: 'gray85'
        _ana2color = '#ececec'  # Closest X11 color: 'gray92'
        self.style = ttk.Style()
        if sys.platform == "win32":
            self.style.theme_use('winnative')
        self.style.configure('.', background=_bgcolor)
        self.style.configure('.', foreground=_fgcolor)
        self.style.configure('.', font="TkDefaultFont")
        self.style.map('.', background=[('selected', _compcolor), ('active', _ana2color)])

        top.geometry("900x650+321+54")
        top.minsize(900, 650)
        top.maxsize(900, 650)
        top.resizable(1, 1)
        top.title("Neural Network Trader")
        top.configure(background="#d9d9d9")
        top.configure(highlightbackground="#d9d9d9")
        top.configure(highlightcolor="black")

        self.main_frame = struct_frame(top, relx=0.0, rely=0.0, relheight=1.0, relwidth=1.0, padx="5", pady="5", relief='flat',
                                       borderwidth='2')
        self.side_frame = struct_frame(self.main_frame, relx=0.0, rely=0.0, relheight=1.0, relwidth=0.333, padx="5", pady="5", relief='flat',
                                       borderwidth='2')
        self.style.configure('TNotebook.Tab', background=_bgcolor)
        self.style.configure('TNotebook.Tab', foreground=_fgcolor)
        self.style.map('TNotebook.Tab', background=[('selected', _compcolor), ('active', _ana2color)])
        self.TNotebook1 = ttk.Notebook(self.side_frame)
        self.TNotebook1.place(relx=0.0, rely=0.0, relheight=1.0, relwidth=1.0)
        self.TNotebook1.configure(takefocus="")
        self.TNotebook1_t1 = tk.Frame(self.TNotebook1)
        self.TNotebook1.add(self.TNotebook1_t1, padding=3)
        self.TNotebook1.tab(0, text="Trading", compound="left", underline="-1", )
        self.TNotebook1_t1.configure(background="#d9d9d9", highlightbackground="#d9d9d9", highlightcolor="black")
        self.TNotebook1_t2 = tk.Frame(self.TNotebook1)
        self.TNotebook1.add(self.TNotebook1_t2, padding=3)
        self.TNotebook1.tab(1, text="Options", compound="left", underline="-1", )
        self.TNotebook1_t2.configure(background="#d9d9d9", highlightbackground="#d9d9d9", highlightcolor="black")
        self.TNotebook1_t3 = tk.Frame(self.TNotebook1)
        self.TNotebook1.add(self.TNotebook1_t3, padding=3)
        self.TNotebook1.tab(2, text="Training", compound="left", underline="-1", )
        self.TNotebook1_t3.configure(background="#d9d9d9", highlightbackground="#d9d9d9", highlightcolor="black")

        self.label_testing = struct_label(self.TNotebook1_t1, 'Testing:', relx=0.034, rely=0.016, height=40, width=120,
                                          size=20, bold=True, justify='left')
        self.text_testing = struct_label(self.TNotebook1_t1, 'FALSE', relx=0.439, rely=0.016, height=40, width=120,
                                         size=18, bold=True, justify='left')
        self.label_trading = struct_label(self.TNotebook1_t1, 'Trading', relx=0.101, rely=0.08, height=35, width=75,
                                          size=16, justify='left')
        self.label_act_dact = struct_label(self.TNotebook1_t1, 'Inactive', relx=0.65, rely=0.08, height=35, width=65,
                                           size=10, justify='right')

        self.img_indicator_light = tk.Canvas(self.TNotebook1_t1)
        self.img_indicator_light.place(relx=0.9, rely=0.1, relheight=0.03, relwidth=0.060)
        self.img_indicator_light.configure(borderwidth=0, highlightthickness=0, background="#d9d9d9")
        self.ind_img_container = self.img_indicator_light.create_image(0, 0, anchor='nw')

        self.Frame1 = struct_frame(self.TNotebook1_t1, relx=0.034, rely=0.135, relheight=0.4, relwidth=0.946, borderwidth='2')

        self.rb_cont_arr = []
        self.text_amount_arr = []
        self.text_usd_arr = []
        for count, c in enumerate(bybit_gui_support.Symbols.all):
            y_pos = 10 + (5 * count) + (20 * count)
            rb = ttk.Radiobutton(self.Frame1)
            rb.place(x=10, y=y_pos, width=46, height=20)
            rb.configure(value=count, text=c, variable=bybit_gui_support.TRADING_CRYPTO, command=lambda: bybit_gui_support.change_trading_crypto())
            self.rb_cont_arr.append(rb)
            self.text_amount_arr.append(struct_label(self.Frame1, '0.0', x=51, y=y_pos, height=20, width=95, justify='right'))
            self.text_usd_arr.append(struct_label(self.Frame1, '($0 USD)', x=151, y=y_pos, height=20, width=100, justify='left'))

        self.label_trade_amount = struct_label(self.Frame1, 'Trade $', x=10, y=115, height=25, width=60, size=13, justify='right')

        self.entry_trade_amount = struct_entry(self.Frame1, x=70, y=120, height=20, width=70, size=11)

        self.label_usd_0 = struct_label(self.Frame1, 'USD', x=140, y=117, height=25, width=30, size=10)
        self.label_loss = struct_label(self.Frame1, 'Loss $', x=10, y=145, height=25, width=60, size=13, justify='right')

        self.entry_loss = struct_entry(self.Frame1, x=70, y=150, height=20, width=45, size=11)

        self.label_usd_1 = struct_label(self.Frame1, 'USD', x=115, y=147, height=25, width=30, size=10)
        self.label_period = struct_label(self.Frame1, 'Period', x=145, y=145, height=25, width=50, size=12)

        self.entry_loss_period = struct_entry(self.Frame1, x=196, y=150, height=20, width=30, size=11)
        # self.label_min = struct_label(self.Frame1, 'min', x=227, y=147, height=25, width=25)

        self.label_leverage = struct_label(self.Frame1, 'Leverage', x=10, y=175, height=23, width=75, size=12, justify='left')
        self.slider_leverage = ttk.Scale(self.Frame1, from_=1, to=100, orient='horizontal',
                                         command=lambda s: bybit_gui_support.LEVERAGE.set(int(float(s))))
        self.slider_leverage.place(x=10, y=205, height=20, width=170)

        self.entry_leverage = EntrySlider(slider=self.slider_leverage, textvariable=bybit_gui_support.LEVERAGE,
                                          master=self.Frame1, takefocus="", cursor="ibeam", font="-family {Segoe UI} -size 9")
        self.entry_leverage.place(x=185, y=200, height=24, width=60)
        self.entry_leverage.bind('<FocusOut>', lambda s: s.widget.set(EntrySlider.var_parce(self.entry_leverage.get())))

        # self.entry_leverage = struct_entry(self.Frame1, x=185, y=200, height=24, width=60)
        # self.entry_leverage.configure(textvariable=str(bybit_gui_support.LEVERAGE))
        # self.entry_leverage.bind('<FocusOut>', lambda s: self.slider_leverage.set(int(float(s.widget.get()))))

        self.Label_pred_header = struct_label(self.TNotebook1_t1, 'Prediction', relx=0.101, rely=0.534, height=35, width=100, size=16,
                                              justify='left')

        self.Frame2 = struct_frame(self.TNotebook1_t1, x=10, rely=0.59, relheight=0.115, relwidth=0.946, borderwidth="2")

        self.label_pred_total = struct_label(self.Frame2, 'Total:', relx=0.03, rely=0, height=22, width=70, size=12, justify='right')
        self.text_pred_total = struct_label(self.Frame2, '0', relx=0.321, rely=0.01, height=22, width=80, size=11, justify='left')
        self.label_pred_correct = struct_label(self.Frame2, 'Correct:', relx=0.03, rely=0.3, height=22, width=70, size=12, justify='right')
        self.text_pred_correct = struct_label(self.Frame2, '0', relx=0.321, rely=0.325, height=22, width=80, size=11, justify='left')
        self.label_pred_acc = struct_label(self.Frame2, 'Accuracy:', relx=0.03, rely=0.625, height=22, width=70, size=12, justify='right')
        self.text_pred_acc = struct_label(self.Frame2, '0%', relx=0.321, rely=0.63, height=22, width=80, size=11, justify='left')

        self.text_error_bar = struct_label(self.TNotebook1_t1, '', relx=0.1, rely=0.71, height=90, relwidth=0.8, size=10, bold=True,
                                           justify='center', )

        self.btn_update_trades = struct_button(self.TNotebook1_t1, 'Update Trade Values', x=140, rely=0.870, height=60, width=0,
                                               command=lambda: bybit_gui_support.update_trading_settings())
        self.btn_start_stop = struct_button(self.TNotebook1_t1, 'Loading', x=10, rely=0.870, height=60, width=260,
                                            command=lambda: bybit_gui_support.start_btn())
        self.btn_start_stop.state(['disabled'])

        self.label_fees = struct_label(self.TNotebook1_t1, '', x=10, rely=0.973, height=15, width=260,
                                       size=7, justify='center')
        """
        # ************************************
        # tab 2 OPTIONS
        # ************************************
        """
        self.label_opt_testing = struct_label(self.TNotebook1_t2, 'Testing', relx=0.034, rely=0.016, height=40, width=120, size=20, justify='left')

        self.btn_testing_on_off = struct_button(self.TNotebook1_t2, 'On', relx=0.507, rely=0.016, height=40, width=100,
                                                command=lambda: bybit_gui_support.testing_toggle())
        self.TFrame1 = struct_frame(self.TNotebook1_t2, relx=0.034, rely=0.112, relheight=0.128, relwidth=0.946, borderwidth='2')

        self.label_api_key_enter = struct_label(self.TFrame1, 'API Key', relx=0.036, rely=0.125, height=25, width=75, size=12, justify='right')
        self.entry_api_key = struct_entry(self.TFrame1, relx=0.339, rely=0.125, relheight=0.313, relwidth=0.607, size=11)

        self.label_api_secret_enter = struct_label(self.TFrame1, 'API Secret', relx=0.036, rely=0.563, height=25, width=75, size=12, justify='right')

        self.entry_api_secret = struct_entry(self.TFrame1, relx=0.339, rely=0.563, relheight=0.313, relwidth=0.607, size=11)

        self.btn_opt_save_settings = struct_button(self.TNotebook1_t2, 'Save Settings', relx=0.034, rely=0.929, height=30, width=100,
                                                   command=lambda: bybit_gui_support.save_settings())

        self.TFrame2 = struct_frame(self.TNotebook1_t2, relx=0.034, rely=0.272, relheight=0.345, relwidth=0.946, borderwidth='2')
        """
        # ************************************
        # tab 3 TRAINING
        # ************************************
        """
        self.label_opt_training = struct_label(self.TNotebook1_t3, 'Model Training', relx=0.034, rely=0.016, height=40, width=240, size=20,
                                               justify='left')

        self.FrameTraining1 = struct_frame(self.TNotebook1_t3, relx=0.034, rely=0.085, height=220, relwidth=0.946, borderwidth='2')

        label_height, field_height, field_width = 22, 20, 75

        self.label_target = struct_label(self.FrameTraining1, 'Target:', x=1, y=1, height=label_height, width=90, size=11, justify='right')
        self.combo_box_target = ttk.Combobox(self.FrameTraining1, textvariable=bybit_gui_support.TRAIN_SYMBOL,
                                             values=bybit_gui_support.Symbols.all, state='readonly',
                                             font="-family {Segoe UI} -size 9")
        self.combo_box_target.place(x=96, y=3, height=field_height, width=50)
        self.combo_box_target.current(0)

        self.label_future_p = struct_label(self.FrameTraining1, 'Future:', x=1, y=25, height=label_height, width=90, size=11, justify='right')

        self.entry_future_p = EntryInt(self.FrameTraining1, takefocus="", cursor="ibeam", font="-family {Segoe UI} -size 9")
        self.entry_future_p.place(x=96, y=27, height=field_height, width=field_width)

        self.label_data_size = struct_label(self.FrameTraining1, 'Data Size:', x=1, y=49, height=label_height, width=90, size=11, justify='right')

        self.entry_data_size = EntryInt(self.FrameTraining1, takefocus="", cursor="ibeam", font="-family {Segoe UI} -size 9")
        self.entry_data_size.place(x=96, y=51, height=field_height, width=field_width)

        self.label_data_period = struct_label(self.FrameTraining1, 'Data Period:', x=1, y=71, height=label_height, width=90, size=11, justify='right')

        self.combo_box_data_period = ttk.Combobox(self.FrameTraining1, textvariable=bybit_gui_support.DATA_PERIOD,
                                                  values=list(bybit_gui_support.DataPeriods.all.keys()),
                                                  state='readonly',font="-family {Segoe UI} -size 9")
        self.combo_box_data_period.place(x=96, y=73, height=field_height, width=field_width)
        self.combo_box_data_period.current(0)

        self.label_epoch = struct_label(self.FrameTraining1, 'Epoch(s):', x=1, y=93, height=label_height, width=90, size=11, justify='right')

        self.entry_epoch = EntryInt(self.FrameTraining1, takefocus="", cursor="ibeam", font="-family {Segoe UI} -size 9")
        self.entry_epoch.place(x=96, y=95, height=field_height, width=field_width)

        self.label_seq_len = struct_label(self.FrameTraining1, 'Seq Length:', x=1, y=117, height=label_height, width=90, size=11, justify='right')

        self.entry_seq_len = EntryInt(self.FrameTraining1, takefocus="", cursor="ibeam", font="-family {Segoe UI} -size 9")
        self.entry_seq_len.place(x=96, y=119, height=field_height, width=field_width)

        self.label_batch_size = struct_label(self.FrameTraining1, 'Batch Size:', x=1, y=141, height=label_height, width=90, size=11, justify='right')

        self.entry_batch_size = EntryInt(self.FrameTraining1, takefocus="", cursor="ibeam", font="-family {Segoe UI} -size 9")
        self.entry_batch_size.place(x=96, y=143, height=field_height, width=field_width)

        self.label_ma = struct_label(self.FrameTraining1, 'Moving Avg:', x=1, y=165, height=label_height, width=90, size=11, justify='right')

        self.entry_ma = struct_entry(self.FrameTraining1, x=96, y=167, height=field_height, width=100, size=9)

        self.label_ema = struct_label(self.FrameTraining1, 'EM Avg:', x=1, y=189, height=label_height, width=90, size=11, justify='right')
        self.entry_ema = struct_entry(self.FrameTraining1, x=96, y=191, height=field_height, width=100, size=9)

        self.training_fields = [self.combo_box_target, self.entry_future_p, self.entry_data_size, self.combo_box_data_period, self.entry_epoch,
                                self.entry_seq_len, self.entry_batch_size, self.entry_ma, self.entry_ema]

        self.label_opt_model = struct_label(self.TNotebook1_t3, 'Model', relx=0.101, y=270, height=26, width=65, size=16, justify='left')

        self.FrameTraining2 = struct_frame(self.TNotebook1_t3, relx=0.034, y=302, height=210, relwidth=0.946, borderwidth='2')

        self.model_layer_lst = []

        self.btn_sub_layer = struct_button(self.FrameTraining2, '-', x=147, y=2, height=24, width=24,
                                           command=lambda: bybit_gui_support.model_struct_row())
        self.label_ema = struct_label(self.FrameTraining2, 'Layer', x=176, y=1, height=25, width=45, size=13, justify='center')

        self.btn_add_layer = struct_button(self.FrameTraining2, '+', x=225, y=2, height=24, width=24,
                                           command=lambda: bybit_gui_support.model_struct_row(
                                               layer=bybit_gui_support.LayerOptions.DEFAULT_BLUEPRINT_LAYER))

        self.checkbutton_new_data = ttk.Checkbutton(self.TNotebook1_t3, text='New Data')
        self.checkbutton_new_data.place(relx=0.034, y=515, height=22, width=95)
        self.checkbutton_new_data.invoke()

        self.checkbutton_verbose = ttk.Checkbutton(self.TNotebook1_t3, text='Verbose')
        self.checkbutton_verbose.place(relx=0.034, y=535, height=22, width=85)
        self.checkbutton_verbose.state(['!alternate'])

        self.btn_start_train = struct_button(self.TNotebook1_t3, 'Start Training', relx=0.034, y=560, height=30, width=85,
                                             command=lambda: bybit_gui_support.train_model())

        self.label_train_status = struct_label(self.TNotebook1_t3, '', relx=0.370, y=540, height=22, width=170, size=8, justify='left')

        self.prog_bar_train = ttk.Progressbar(self.TNotebook1_t3, orient="horizontal", mode="determinate")
        self.prog_bar_train.place(relx=0.37, y=567, height=15, width=135)
        self.label_prog_percent = struct_label(self.TNotebook1_t3, '', relx=0.86, y=565, height=18, width=40, size=8, justify='left')

        """
        END OF TAB 3
        """
        self.label_model_arr = []
        self.btn_gen_model_arr = []
        self.btn_info_log_model_arr = []
        self.btn_del_model_arr = []
        self.text_model_name_arr = []
        for count, c in enumerate(bybit_gui_support.Symbols.all):
            y_pos = 10 + (5 * count) + (40 * count)
            self.label_model_arr.append(struct_label(self.TFrame2, f'{c} Model:', x=3, y=y_pos, height=30, width=105, size=15, justify='left'))
            self.btn_gen_model_arr.append(struct_button(self.TFrame2, 'Generate', x=108, y=y_pos + 5, height=26, width=58,
                                                        command=lambda sym=c: bybit_gui_support.gen_model_btn(sym)))
            self.btn_info_log_model_arr.append(struct_button(self.TFrame2, 'Info', x=168, y=y_pos + 5, height=26, width=36,
                                                             command=lambda sym=c: bybit_gui_support.get_model_info(sym, display_gui=True)))
            self.btn_info_log_model_arr.append(struct_button(self.TFrame2, 'Delete', x=206, y=y_pos + 5, height=26, width=45,
                                                             command=lambda sym=c: bybit_gui_support.delete_model(sym)))
            self.text_model_name_arr.append(struct_label(self.TFrame2, '...', x=8, y=y_pos + 30, height=15, width=240, size=7, justify='right'))

        self.btn_opt_cancel = struct_button(self.TNotebook1_t2, 'Cancel', relx=0.608, rely=0.929, height=30, width=100)
        self.btn_reset_stats = struct_button(self.TNotebook1_t2, 'Reset Stats', relx=0.034, rely=0.625, height=25, width=70,
                                             command=lambda: bybit_gui_support.reset_stats())

        self.info_frame = struct_frame(self.main_frame, x=295, rely=0.0, relheight=1.0, width=595, relief='flat', borderwidth="2")

        self.combo_box_plot_type = ttk.Combobox(self.info_frame, textvariable=bybit_gui_support.PLOT_TYPE)
        self.combo_box_plot_type.place(x=4, rely=0.25, height=23, width=80)
        self.combo_box_plot_type.configure(background="#d9d9d9")
        self.combo_box_plot_type['values'] = bybit_gui_support.PlotTypes.all
        self.combo_box_plot_type['state'] = 'readonly'
        self.combo_box_plot_type.bind('<<ComboboxSelected>>', bybit_gui_support.plot_combo_box)

        self.combo_box_bal_period = ttk.Combobox(self.info_frame, textvariable=bybit_gui_support.BAL_PERIOD)
        self.combo_box_bal_period['values'] = list(bybit_gui_support.BALANCE_PERIODS_DICT.keys())
        self.combo_box_bal_period['state'] = 'readonly'
        self.combo_box_bal_period.bind('<<ComboboxSelected>>', bybit_gui_support.bal_period_combo_box)
        self.combo_box_bal_period.configure(background="#d9d9d9")

        self.ma_btns = []
        bybit_gui_support.place_moving_avg_btns(self, bybit_gui_support.MOVING_AVG_DICT)
        self.label_break_line = tk.Label(self.info_frame)

        self.loading_plot_label = struct_label(self.info_frame, 'Loading...', x=4, rely=0.290, relheight=0.650, width=584, size=25, justify='center')

        self.frame_api_display = struct_frame(self.info_frame, x=4, rely=0.946, relheight=0.052, width=584, borderwidth='2', padx="1", pady="1")

        self.label_api_key = struct_label(self.frame_api_display, 'Api Key:', relx=0.003, rely=0, height=14, width=50, size=7, justify='right')
        self.label_api_secret = struct_label(self.frame_api_display, 'Api Secret:', relx=0.003, rely=0.45, height=14, width=50, size=7,
                                             justify='right')

        self.text_api_key = struct_label(self.frame_api_display, ' ', relx=0.091, rely=0, height=15, width=90, size=7)
        self.text_api_secret = struct_label(self.frame_api_display, ' ', relx=0.091, rely=0.45, height=15, width=170, size=7)
        self.time_to_next_up = struct_label(self.frame_api_display, 'Update in: 99 sec', x=508, y=0, height=15, width=70, size=7, justify='right')

        self.text_version = struct_label(self.frame_api_display, f'Version: {bybit_gui_support.VERSION}', x=508, y=12, height=15, width=70, size=7,
                                         justify='right')

        self.inner_frame_logo = struct_frame(self.info_frame, x=4, rely=0.0, relheight=0.077, width=584)
        photo_location = os.path.join(prog_location, "bin/res/main_logo.png")
        global _img0
        _img0 = tk.PhotoImage(file=photo_location)
        self.logo_name = tk.Label(self.inner_frame_logo)
        self.logo_name.place(relx=0, rely=0, height=50, width=550)
        self.logo_name.configure(activebackground="#f9f9f9", activeforeground="black", background="#d9d9d9", disabledforeground="#a3a3a3",
                                 foreground="#000000", highlightbackground="#d9d9d9", highlightcolor="black",
                                 image=_img0, text='')

        self.inner_frame_profit = struct_frame(self.info_frame, x=4, rely=0.092, relheight=0.154, width=584, borderwidth='2')

        self.label_total_profit = struct_label(self.inner_frame_profit, 'Total Profit:', relx=0.008, rely=0.1, height=40, width=159, size=20,
                                               bold=True, justify='right')
        self.text_total_profit = struct_label(self.inner_frame_profit, '$0', relx=0.286, rely=0.1, height=40, width=200, size=20, justify='left')
        self.label_testing_profit = struct_label(self.inner_frame_profit, '', relx=0.008, rely=0.5, height=30, width=159,
                                                 size=15, justify='left')
        self.text_testing_profit = struct_label(self.inner_frame_profit, '', relx=0.288, rely=0.5, height=30,
                                                width=199, size=15, justify='left')
        self.label_prediction = struct_label(self.inner_frame_profit, 'Prediction:', relx=0.676, rely=0.1, height=40, width=169, size=20, bold=True,
                                             justify='left')
        self.text_prediction = struct_label(self.inner_frame_profit, '-', relx=0.676, rely=0.5, height=35, width=169, size=16, justify='left')


class EntryInt(ttk.Entry):
    def __init__(self, master=None, **kwargs):
        self.var = tk.StringVar()
        tk.Entry.__init__(self, master, textvariable=self.var, **kwargs)
        self.old_value = ''
        self.var.trace('w', self.check)
        self.get, self.set = self.var.get, self.var.set

    def check(self, *args):
        if self.get().isdigit():
            # the current value is only digits; allow this
            self.old_value = self.get()
        elif len(self.get()) == 0:
            self.set('')
        elif self.get() == '':
            pass
        else:
            # there's non-digit characters in the input; reject this
            self.set(self.old_value)


class EntrySlider(ttk.Entry):
    def __init__(self, slider, textvariable, master=None, **kwargs):
        self.var = textvariable
        self.linked_slider = slider
        tk.Entry.__init__(self, master, textvariable=self.var, **kwargs)
        self.var.trace('w', self.check)
        self.get, self.set = self.var.get, self.var.set

    def check(self, *args):
        try:
            v = self.var_parce(self.var.get())
            if v <= 0:
                self.var.set(1)
            elif v > 100:
                self.var.set(100)
            self.linked_slider.set(int(float(self.var.get())))
        except ValueError:
            pass

    @staticmethod
    def var_parce(var):
        try:
            return int(float(var))
        except ValueError:
            return 1


def popup_bonus(**kwargs):
    win = tk.Toplevel()
    win.iconphoto(False, tk.PhotoImage(file='bin/res/icon.png'))
    win.wm_title(f'{kwargs.get("title")}:  {str(kwargs.get("location"))}')

    def make_label(text, column=None, columnspan=None):
        label = tk.Label(win, text=text, bg=None)
        if columnspan is not None:
            label.grid(row=row, columnspan=columnspan)
        else:
            label.grid(row=row, column=column)

    row = 1
    make_label(f'Input:', 0)
    make_label(f'Sequences: {kwargs.get("seq")}', 1)
    make_label(f'Features: {kwargs.get("features")}', 2)
    row += 1
    make_label('=' * 30, row, columnspan=3)
    row += 2
    make_label('Name', 0)
    make_label('Neurons', 1)
    make_label('# of Params', 2)
    row += 1
    make_label('-' * 50, columnspan=3)
    for layer in kwargs.get('layers'):
        row += 1
        for col, value in enumerate(layer):
            make_label(f'{value}', column=col)
    row += 1
    make_label('-' * 50, columnspan=3)
    row += 1
    make_label('Total Trainable Parameters:', columnspan=2)
    # l7 = ttk.Label(win, text=f'Total Trainable Parameters:')
    # l7.grid(row=row, column=0, columnspan=2)
    make_label(f'{kwargs.get("total_param")}', 2)
    row += 1
    b = ttk.Button(win, text="Close", command=win.destroy)
    b.grid(row=row, column=1)


if __name__ == '__main__':
    vp_start_gui()
