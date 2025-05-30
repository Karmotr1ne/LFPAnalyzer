import sys
import os
import neo
import pywt
import h5py
import numpy as np
import pandas as pd
import pyqtgraph as pg
import pyquery as pq
from neo.io import NixIO
from PyQt5 import QtWidgets,QtCore,QtGui
from PyQt5.QtCore import QSettings, QDir, Qt
from functools import partial
from scipy.sparse.linalg import spsolve
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, peak_widths

class LFPAnalyzer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('LFP Data Analyzer - by Songlin Yang (Karmotr1ne@GitHub)')
        self.resize(1200, 800)
        self.setWindowIcon(QtGui.QIcon('app_icon.png'))
        self.signals = {}
        self.loaded_paths = set()
        self.last_folder = None
        self.path_map = {} 
        self.source_map = {}
        self.current_key = None
        self.raw_signals = {}
        self.proc_key = None
        self.proc_signal = None
        self.spike_info = {}
        self.baseline_curve = []
        self._create_main_layout()
        self.raw_plot.addLegend()
        self.raw_plot.plotItem.legend.setVisible(False)

        # wavelet spinbox
        self.combo_wavelet.setCurrentText('cmor1.5-1.0')
        self.combo_wavelet.currentTextChanged.connect(self.on_wavelet_changed)

        #enable drag
        self.raw_plot.plotItem.vb.setMouseEnabled(x=True, y=True)
        self.proc_plot.plotItem.vb.setMouseEnabled(x=True, y=True)

        #signal interactivity
        self.shortcut_auto_range = QtWidgets.QShortcut(QtGui.QKeySequence("A"), self)
        self.shortcut_auto_range.activated.connect(self.reset_view)

        self.manual_spike_scatter = pg.ScatterPlotItem(size=8, brush=pg.mkBrush('#006400'))
        self.proc_plot.addItem(self.manual_spike_scatter)

        # Hover marker
        self.hover_marker = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(0, 200, 0, 150))
        self.hover_marker.setVisible(False)
        self.hover_marker.setZValue(1000)
        self.proc_plot.addItem(self.hover_marker)

        # Redo manual mark
        self.shortcut_undo_spike = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+Z"), self)
        self.shortcut_undo_spike.activated.connect(self.undo_manual_spike)
        self.manual_spike_history = []

        #Memory location and Load more data
        self.settings = QSettings('FileLocation', 'LFPAnalyzer')        
        self.file_tree.itemClicked.connect(self.on_tree_item_clicked)

        #batch memory
        self.last_preprocess_input_folder = None
        self.last_preprocess_output_folder = None
        self.last_extract_input_folder = None
        self.last_extract_output_folder = None

    #GUI
    def _create_main_layout(self):
        # Central widget and main splitter
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Left panel: File tree and operations
        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_panel.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)

        # File tree outside label 
        lbl_tree = QtWidgets.QLabel('Current Data Files')
        lbl_tree.setAlignment(QtCore.Qt.AlignLeft)
        left_layout.addWidget(lbl_tree)

        # File tree function
        self.file_tree = QtWidgets.QTreeWidget()
        self.file_tree.setHeaderHidden(True) #hidden
        self.file_tree.setIndentation(0)
        self.file_tree.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        left_layout.addWidget(self.file_tree)

        # enable right click and menu
        self.file_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.file_tree.customContextMenuRequested.connect(self.open_context_menu)

        # File operation buttons
        btn_layout = QtWidgets.QHBoxLayout()
        self.btn_add = QtWidgets.QPushButton('Add')
        self.btn_remove = QtWidgets.QPushButton('Remove')
        self.btn_save = QtWidgets.QPushButton('Save')
        btn_layout.addWidget(self.btn_add)
        btn_layout.addWidget(self.btn_remove)
        btn_layout.addWidget(self.btn_save)
        left_layout.addLayout(btn_layout)

        # Downsample size
        bin_layout = QtWidgets.QHBoxLayout()
        bin_layout.addWidget(QtWidgets.QLabel('Downsample:'))
        self.spin_down = QtWidgets.QSpinBox()
        self.spin_down.setRange(1, 100)
        self.spin_down.setValue(20)
        bin_layout.addWidget(self.spin_down)
        self.btn_apply = QtWidgets.QPushButton('Apply')
        bin_layout.addWidget(self.btn_apply)
        left_layout.addLayout(bin_layout)

        # ALS parameters
        self.base_lambda = 1e4  # λ
        self.base_p = 1e-4      # p

        als_layout = QtWidgets.QHBoxLayout()
        als_layout.addWidget(QtWidgets.QLabel('λ (10^x):'))

        self.spin_sigma_lambda = QtWidgets.QSpinBox()
        self.spin_sigma_lambda.setRange(-3, 5)
        self.spin_sigma_lambda.setValue(0)
        als_layout.addWidget(self.spin_sigma_lambda)

        als_layout.addWidget(QtWidgets.QLabel('p (10^x):'))
        self.spin_sigma_p = QtWidgets.QSpinBox()
        self.spin_sigma_p.setRange(-3, 3)
        self.spin_sigma_p.setValue(0)
        als_layout.addWidget(self.spin_sigma_p)

        left_layout.addLayout(als_layout)

        # Reduce baseline UI (with checkbox and method combo)
        baseline_layout = QtWidgets.QHBoxLayout()
        self.chk_use_time = QtWidgets.QCheckBox("Set baseline by time")
        self.chk_use_time.setChecked(False)
        self.chk_use_time.stateChanged.connect(self.toggle_time_selector)
        baseline_layout.addWidget(self.chk_use_time)
        baseline_layout.addStretch()  # Method right align

        baseline_method_layout = QtWidgets.QHBoxLayout()
        baseline_method_layout.addWidget(QtWidgets.QLabel("Method:"))
        self.combo_baseline_mode = QtWidgets.QComboBox()
        self.combo_baseline_mode.addItems(['mode', 'mean', 'median'])
        baseline_method_layout.addWidget(self.combo_baseline_mode)
        baseline_layout.addLayout(baseline_method_layout)

        left_layout.addLayout(baseline_layout)

        #wavelet controller
        wavelet_group = QtWidgets.QGroupBox("Wavelet Settings")
        wavelet_form = QtWidgets.QFormLayout()

        #Auto freq and wavelet
        freq_recommend_layout = QtWidgets.QHBoxLayout()
        self.checkbox_auto_wavelet = QtWidgets.QCheckBox("Auto mother wavelet")
        self.checkbox_auto_wavelet.setChecked(True)
        self.btn_recommend_freq = QtWidgets.QPushButton("Auto Frequency")
        self.btn_recommend_freq.clicked.connect(self.recommend_freq_and_wavelet)
        freq_recommend_layout.addWidget(self.checkbox_auto_wavelet)
        freq_recommend_layout.addStretch()
        freq_recommend_layout.addWidget(self.btn_recommend_freq)

        auto_layout = QtWidgets.QHBoxLayout()
        auto_layout.addWidget(self.checkbox_auto_wavelet)
        auto_layout.addSpacing(10) 
        auto_layout.addWidget(self.btn_recommend_freq)
        wavelet_form.addRow(auto_layout)

        #Wavelet selector dropdown
        self.combo_wavelet = QtWidgets.QComboBox()
        self.combo_wavelet.addItems(['morl','cmor1.5-1.0','mexh','gaus1'])

        #Frequency range
        self.freq_low = QtWidgets.QSpinBox()
        self.freq_low.setRange(1, 10000)
        self.freq_low.setValue(20)
        self.freq_high = QtWidgets.QSpinBox()
        self.freq_high.setRange(1, 10000)
        self.freq_high.setValue(250)

        freq_range_layout = QtWidgets.QHBoxLayout()
        dash_label = QtWidgets.QLabel("-")
        dash_label.setAlignment(Qt.AlignCenter)
        dash_label.setFixedWidth(15) 
        freq_range_layout.addWidget(self.freq_low)
        freq_range_layout.addWidget(dash_label)
        freq_range_layout.addWidget(self.freq_high)

        #Layout
        wavelet_form.addRow("Mother wavelet:", self.combo_wavelet)
        wavelet_form.addRow("Freq Range (Hz):", freq_range_layout)

        # Threshold
        self.wavelet_thresh = QtWidgets.QDoubleSpinBox()
        self.wavelet_thresh.setRange(0.5, 10)
        self.wavelet_thresh.setSingleStep(0.1)
        self.wavelet_thresh.setValue(3.0)
        wavelet_form.addRow("Threshold (SD):", self.wavelet_thresh)

        wavelet_group.setLayout(wavelet_form)
        left_layout.addWidget(wavelet_group)

        #Spike Clustering
        filter_group = QtWidgets.QGroupBox("Spike Filter Settings")
        filter_form = QtWidgets.QFormLayout()

        self.mergeSpikesCheckBox = QtWidgets.QCheckBox("Merge nearby spikes")
        self.mergeSpikesCheckBox.setChecked(True)
        filter_form.addRow(self.mergeSpikesCheckBox)

        self.minIntervalSpinBox = QtWidgets.QDoubleSpinBox()
        self.minIntervalSpinBox.setRange(1.0, 1000.0)
        self.minIntervalSpinBox.setSingleStep(1.0)
        self.minIntervalSpinBox.setValue(200.0)
        filter_form.addRow("Min Interval (ms):", self.minIntervalSpinBox)

        self.minAmplitudeSpinBox = QtWidgets.QDoubleSpinBox()
        self.minAmplitudeSpinBox.setRange(0.0, 1000.0)
        self.minAmplitudeSpinBox.setValue(0.05)
        self.minAmplitudeSpinBox.setSingleStep(0.01)
        filter_form.addRow("Min Amplitude (mV):", self.minAmplitudeSpinBox)

        self.relThresholdSpinBox = QtWidgets.QDoubleSpinBox()
        self.relThresholdSpinBox.setRange(0, 10)
        self.relThresholdSpinBox.setDecimals(0)
        self.relThresholdSpinBox.setSingleStep(1)
        self.relThresholdSpinBox.setValue(0)
        filter_form.addRow("Threshold (SD):", self.relThresholdSpinBox)

        filter_group.setLayout(filter_form)
        left_layout.addWidget(filter_group)

        # Spike analysis controls - GroupBox version
        spike_group = QtWidgets.QGroupBox("Spike Analysis Settings")
        spike_layout = QtWidgets.QVBoxLayout()

        # Manual Spike Checkbox + Show Raw + Clear Button
        manual_layout = QtWidgets.QHBoxLayout()
        self.chk_manual_spike = QtWidgets.QCheckBox("Manual Spike")
        self.chk_manual_spike.setChecked(False)
        self.chk_show_raw = QtWidgets.QCheckBox("Show Origin")
        self.chk_show_raw.setChecked(False)
        self.btn_clear_spikes = QtWidgets.QPushButton('Clear Spikes')
        manual_layout.addWidget(self.chk_manual_spike)
        manual_layout.addWidget(self.chk_show_raw)
        manual_layout.addWidget(self.btn_clear_spikes)
        spike_layout.addLayout(manual_layout)

        # Spike window settings
        self.chk_auto_ap_window = QtWidgets.QCheckBox("Auto AP Analysis Window")
        self.chk_auto_ap_window.setChecked(True)
        spike_layout.addWidget(self.chk_auto_ap_window)

        window_layout = QtWidgets.QHBoxLayout()
        window_layout.addWidget(QtWidgets.QLabel('Pre AP (ms):'))
        self.spin_window_pre = QtWidgets.QSpinBox()
        self.spin_window_pre.setRange(0, 500)
        self.spin_window_pre.setValue(10)
        window_layout.addWidget(self.spin_window_pre)

        window_layout.addWidget(QtWidgets.QLabel('Post AP (ms):'))
        self.spin_window_post = QtWidgets.QSpinBox()
        self.spin_window_post.setRange(0, 500)
        self.spin_window_post.setValue(20)
        window_layout.addWidget(self.spin_window_post)
        spike_layout.addLayout(window_layout)

        # Pack group
        spike_group.setLayout(spike_layout)
        left_layout.addWidget(spike_group)


        # Batch processing button
        batch_group = QtWidgets.QGroupBox("Batch Processing")
        batch_layout = QtWidgets.QVBoxLayout()

        self.btn_batch_preprocess = QtWidgets.QPushButton('Preprocess All')
        batch_layout.addWidget(self.btn_batch_preprocess)

        # processing
        self.batch_progress_bar = QtWidgets.QProgressBar()
        self.batch_progress_bar.setMinimum(0)
        self.batch_progress_bar.setMaximum(100)
        self.batch_progress_bar.setValue(0)
        self.batch_progress_bar.setVisible(False)  # hide
        self.batch_progress_bar.setTextVisible(True)
        batch_layout.addWidget(self.batch_progress_bar)

        batch_group.setLayout(batch_layout)
        left_layout.addWidget(batch_group)

        # Spacer
        left_layout.addStretch()

        splitter.addWidget(left_panel)

        # Right panel: Plots and controls
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        self._drag_start = None

        # Raw signal plot
        self.raw_plot = pg.PlotWidget(title='Raw Signal')
        self.raw_plot.setBackground('w')  
        right_layout.addWidget(self.raw_plot)

        # Processed signal plot
        self.proc_plot = pg.PlotWidget(title='Processed Signal')
        self.proc_plot.setBackground('w') 

        #enable drag
        self.bind_mouse_events(self.raw_plot)
        self.bind_mouse_events(self.proc_plot)

        #sync
        self.proc_selector_fill = pg.LinearRegionItem([0, 0], brush=(0, 100, 255, 50))
        self.proc_selector_fill.setZValue(10)
        self.proc_selector_fill.setMovable(False)
        self.proc_selector_fill.setVisible(False)
        self.proc_plot.addItem(self.proc_selector_fill)

        right_layout.addWidget(self.proc_plot)

        #draw selected area
        self.raw_plot.scene().sigMouseMoved.connect(self.on_mouse_drag_move)
        self.proc_plot.scene().sigMouseMoved.connect(self.on_mouse_drag_move)

        # Operation buttons below processed plot
        op_layout = QtWidgets.QHBoxLayout()
        self.btn_smooth = QtWidgets.QPushButton('ALS Detrend')
        self.btn_align = QtWidgets.QPushButton("Zero Baseline")
        self.btn_detect = QtWidgets.QPushButton('Spike Detect')
        self.btn_filter = QtWidgets.QPushButton('Filter Spikes')
        self.btn_ap_analysis = QtWidgets.QPushButton('AP Analysis')

        op_layout.addWidget(self.btn_smooth)
        op_layout.addWidget(self.btn_align)
        op_layout.addWidget(self.btn_detect)
        op_layout.addWidget(self.btn_filter)
        op_layout.addWidget(self.btn_ap_analysis)
        right_layout.addLayout(op_layout)
        splitter.addWidget(right_panel)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)

        # right stretch
        splitter.setStretchFactor(0, 0)  
        splitter.setStretchFactor(1, 1) 
        splitter.setSizes([250, 950])

        # Connect signals to slots (to be implemented)
        self.btn_add.clicked.connect(self.load_file)
        self.btn_remove.clicked.connect(self.remove_file)
        self.btn_save.clicked.connect(self.save)
        self.btn_apply.clicked.connect(self.apply_downsample)
        self.btn_batch_preprocess.clicked.connect(self.batch_preprocess)
        self.btn_smooth.clicked.connect(self.apply_smooth)
        self.btn_align.clicked.connect(self.apply_zero_baseline)
        self.btn_detect.clicked.connect(self.apply_detect)
        self.btn_filter.clicked.connect(self.apply_filter)
        self.btn_ap_analysis.clicked.connect(self.apanalysis)
        self.btn_clear_spikes.clicked.connect(self.clear_spikes)
        self.chk_manual_spike.stateChanged.connect(self.update_spike_mode_link)

# Methods
    #load file
    def load_single_file(self, path):
        """Load a .abf or .h5 file, including raw, processed signals, and spike events."""
        from copy import deepcopy
        import neo

        base_name = os.path.basename(path)
        ext = os.path.splitext(path)[1].lower()

        try:
            if ext == '.abf':
                reader = neo.io.AxonIO(filename=path)
                block = reader.read_block(lazy=False)
            elif ext == '.h5':
                reader = NixIO(filename=path, mode='ro')
                block = reader.read_block(lazy=False)
                if not hasattr(self, 'h5_readers'):
                    self.h5_readers = {}
                self.h5_readers[path] = reader
            else:
                QtWidgets.QMessageBox.warning(self, "Warning", f"Unsupported file type: {ext}")
                return
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load Error", f"Failed to load {path}\n{e}")
            return

        try:
            for i, seg in enumerate(block.segments):
                raw_sig = None
                proc_sig = None

                for sig in seg.analogsignals:
                    if sig.name == "raw":
                        raw_sig = deepcopy(sig)
                    elif sig.name == "processed":
                        proc_sig = deepcopy(sig)

                if proc_sig is None and seg.analogsignals:
                    proc_sig = deepcopy(seg.analogsignals[0])
                    print(f"[Warning] No 'processed' signal found, using the first available signal for {base_name} sweep {i}.")

                if proc_sig is None:
                    continue

                display_name = self._generate_unique_name(f"{base_name}_sweep{i}")

                self.signals[display_name] = proc_sig
                self.path_map[display_name] = path
                self.loaded_paths.add(path)

                if raw_sig is not None:
                    self.raw_signals[display_name] = raw_sig
                else:
                    self.raw_signals[display_name] = proc_sig  # fallback

                # Add item to file tree
                item = QtWidgets.QTreeWidgetItem([display_name])
                item.setData(0, QtCore.Qt.UserRole, display_name)
                self.file_tree.addTopLevelItem(item)

                # Set current view if empty
                if self.current_key is None:
                    self.current_key = display_name
                    self.proc_key = display_name
                    self.proc_signal = self.signals[display_name]
                    self.raw_signal = self.raw_signals.get(display_name)
                    self.file_tree.setCurrentItem(item)
                    self.on_tree_item_clicked(item, 0)

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load Error", f"Error during parsing {path}\n{e}")

    #load method
    def get_all_files(self, folder, suffixes=('.abf', '.h5')):
        """Get all files in folder matching given suffixes."""
        file_list = []
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(suffixes):
                    file_list.append(os.path.join(root, file))
        return file_list

    def is_nix_file(self, filepath):
        """Check if a .h5 file is a NixIO format."""
        import h5py
        try:
            with h5py.File(filepath, 'r') as f:
                if 'nix_version' in f.attrs or 'nix_version' in f.keys():
                    return True
        except Exception:
            pass
        return False

    def load_file(self):
        """Let user select and load multiple .abf or .h5 files."""
        files = self.select_files('Select File(s)', 'Signal Files (*.abf *.h5)')
        if not files:
            return

        new_files = [f for f in files if f not in self.loaded_paths]
        if not new_files:
            self.show_info('All selected files have already been loaded.', title="Already Loaded")
            return

        for path in new_files:
            self.load_single_file(path)

        # Focus first loaded file
        if new_files:
            first_path = new_files[0]
            base_name = os.path.splitext(os.path.basename(first_path))[0]
            for i in range(self.file_tree.topLevelItemCount()):
                item = self.file_tree.topLevelItem(i)
                key = item.data(0, QtCore.Qt.UserRole)
                if key.startswith(base_name):
                    self.file_tree.setCurrentItem(item)
                    self.on_tree_item_clicked()
                    break

    def find_original_raw_key(self, key):
        while key in self.source_map:
            key = self.source_map[key]
        return key.split("_")[0]  # move index

    #save file
    def closeEvent(self, event):
        """Prompt to save before exiting. Close all resources."""
        reply = QtWidgets.QMessageBox.question(
            self,
            'Exit Confirmation',
            "Do you want to save your work before exiting?",
            QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Discard | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Save
        )

        if reply == QtWidgets.QMessageBox.Save:
            self.save()
            self.cleanup_resources()
            event.accept()

        elif reply == QtWidgets.QMessageBox.Discard:
            # exit
            self.cleanup_resources()
            event.accept()

        else:
            # close
            event.ignore()

    def cleanup_resources(self):
        """Close all open file readers safely."""
        if hasattr(self, 'h5_readers'):
            for reader in self.h5_readers.values():
                try:
                    reader.close()
                except Exception:
                    pass

    def save(self):
        """Save selected signals to separate HDF5 files."""
        selected_items = self.file_tree.selectedItems()
        if not selected_items:
            self.show_warning("No signals selected in the file tree.")
            return

        out_dir = self.select_folder("Select Output Folder")
        if not out_dir:
            return

        errors = []

        for item in selected_items:
            key = item.data(0, QtCore.Qt.UserRole)
            signal = self.signals.get(key, None)
            if signal is None:
                errors.append(f"{key}: signal not found")
                continue

            out_path = os.path.join(out_dir, f"{key.replace('/', '_')}.h5")
            try:
                self._save_signal_to_h5(signal, out_path, key)
            except Exception as e:
                errors.append(f"{key}: {e}")

        if errors:
            self.show_warning("Some files failed to save:\n" + "\n".join(errors))
        else:
            self.show_info("All selected signals saved successfully.", title="Success")

    def _save_signal_to_h5(self, signal, out_path, key):
        from neo import Block, Segment
        from neo.io import NixIO

        if os.path.exists(out_path):
            os.remove(out_path)

        blk = Block(name=f"block_{key}")
        seg = Segment(name=f"seg_{key}")

        signal.name = f"sig_{key}"
        seg.analogsignals.append(signal)
        blk.segments.append(seg)

        # signal
        with NixIO(filename=out_path, mode='ow') as io:
            io.write_block(blk)

    #remove
    def remove_file(self):
        """Remove selected signals from the view and memory."""
        selected_items = self.file_tree.selectedItems()
        for item in selected_items:
            key = item.data(0, QtCore.Qt.UserRole)
            self.signals.pop(key, None)
            self.path_map.pop(key, None)
            self.raw_signals.pop(key, None)
            self.spike_info.pop(key, None)
            self.source_map.pop(key, None)

            index = self.file_tree.indexOfTopLevelItem(item)
            self.file_tree.takeTopLevelItem(index)

        # Clear plots
        self.clear_plot(self.raw_plot, selector=self.time_selector if hasattr(self, 'time_selector') else None)
        self.clear_plot(self.proc_plot, selector=self.proc_selector_fill if hasattr(self, 'proc_selector_fill') else None)

        self.current_key = None
        self.proc_key = None
        self.proc_signal = None

    #plot signal
    def _plot_signal(self, signal, plot, color='b', width=1, name=None, return_item=False):
        t = (signal.times - signal.t_start).rescale('s').magnitude.flatten()
        y = signal.magnitude.flatten()

        pen = pg.mkPen(color=color, width=width)
        plot.plot(t, y, pen=pen, name=name)

        items = plot.listDataItems()
        return items[-1] if items and return_item else None

    #plot patch
    def finalize_processing(self, proc_key, proc_signal, source_key):
        self.proc_key = proc_key
        self.proc_signal = proc_signal
        self.current_key = source_key

        self.clear_raw_plot()
        self.clear_proc_plot()

        if self.chk_show_raw.isChecked():
            # origin
            origin_base_key = self.find_original_raw_key(source_key)
            self.raw_signal = self.raw_signals.get(origin_base_key, None)
        else:
            # last
            prev_key = self.source_map.get(source_key, None)
            self.raw_signal = self.signals.get(prev_key, None)

        # raw plot
        if self.raw_signal is not None:
            self._plot_signal(self.raw_signal, self.raw_plot, color='b')

        # processed plot
        self._plot_signal(proc_signal, self.proc_plot, color='m')

        self.reset_proc_view()

        # click tree item
        for i in range(self.file_tree.topLevelItemCount()):
            item = self.file_tree.topLevelItem(i)
            key = item.data(0, QtCore.Qt.UserRole)
            if key == proc_key:
                self.file_tree.setCurrentItem(item)
                break

    def update_legend_visibility(self):
        """hide when select time for baseline"""
        show_legend = not self.chk_use_time.isChecked()
        if self.raw_plot.plotItem.legend is not None:
            self.raw_plot.plotItem.legend.setVisible(show_legend)
        if self.proc_plot.plotItem.legend is not None:
            self.proc_plot.plotItem.legend.setVisible(show_legend)
    
    def reset_view(self):
        """Reset zoom without disabling interaction"""
        for plot in [self.raw_plot, self.proc_plot]:
            if plot.listDataItems():
                plot.enableAutoRange(axis='xy')
                plot.plotItem.vb.setMouseEnabled(x=True, y=True)

    def reset_proc_view(self):
        if self.proc_plot.listDataItems():
            self.proc_plot.enableAutoRange(axis='xy')
            self.proc_plot.plotItem.vb.setMouseEnabled(x=not self.chk_use_time.isChecked(), y=not self.chk_use_time.isChecked())

    def reset_raw_view(self):
        if self.raw_plot.listDataItems():
            self.raw_plot.enableAutoRange(axis='xy')
            self.raw_plot.plotItem.vb.setMouseEnabled(x=not self.chk_use_time.isChecked(), y=not self.chk_use_time.isChecked())

    def ensure_selector(self, name, plot, color=(0, 100, 255, 100)):
        if not hasattr(self, name) or getattr(self, name) is None:
            selector = pg.LinearRegionItem([0, 0.01], brush=color)
            selector.setMovable(False)
            selector.setZValue(10)
            setattr(self, name, selector)
            plot.addItem(selector)
        return getattr(self, name)
    def update_selector_region(self, region):
        if hasattr(self, "time_selector"):
            self.time_selector.setRegion(region)
        if hasattr(self, "proc_selector_fill"):
            self.proc_selector_fill.setRegion(region)

    #unique name
    def _generate_unique_name(self, base_name):
        """Return a unique name not already in self.signals."""
        name = base_name
        counter = 1
        while name in self.signals:
            name = f"{base_name}_v{counter}"
            counter += 1
        return name

    #file tree
    def on_tree_item_clicked(self, item=None, column=0):
        """Update plots when clicking an item."""
        if item is None:
            item = self.file_tree.currentItem()
            if item is None:
                return

        key = item.data(0, QtCore.Qt.UserRole)

        if key not in self.signals:
            return

        self.current_key = key
        self.proc_key = key
        self.proc_signal = self.signals.get(key, None)
        self.raw_signal = self.raw_signals.get(key, None)

        self.clear_proc_plot()
        self.clear_raw_plot()

        # raw
        if self.raw_signal is not None:
            t = (self.raw_signal.times - self.raw_signal.t_start).rescale('s').magnitude.flatten()
            y = self.raw_signal.magnitude[:, 0].flatten()
            self.raw_plot.plot(t, y, pen=pg.mkPen('k'))

        # proc
        if self.proc_signal is not None:
            self.plot_detected_spikes()

    #tree right click open menu
    #menu
    def open_context_menu(self, position):
        menu = QtWidgets.QMenu()

        clear_reload_action = menu.addAction("Clear and Reload")
        overlay_action = menu.addAction("Overlay Selected Signals")

        action = menu.exec_(self.file_tree.viewport().mapToGlobal(position))

        if action == clear_reload_action:
            self.clear_and_reload()
        elif action == overlay_action:
            self.overlay_selected_signals()

    #clear&reload
    def clear_and_reload(self):
        self.file_tree.clear()
        self.signals.clear()
        self.loaded_paths.clear()
        self.current_key = None
        self.raw_signal = None
        self.proc_signal = None
        self.baseline_curve = []
        self.raw_plot.clear()
        self.proc_plot.clear()
    def clear_plot_with_selector(self, plot_name, selector_name):
        """clear selected plot and keep utensils"""
        plot = getattr(self, plot_name, None)
        selector = getattr(self, selector_name, None)

        if plot is not None:
            plot.clear()
            if selector is not None and selector not in plot.items():
                plot.addItem(selector)
    def clear_raw_plot(self):
        self.clear_plot_with_selector('raw_plot', 'time_selector')
    def clear_proc_plot(self):
        self.proc_plot.clear()

        if hasattr(self, "proc_selector_fill") and self.proc_selector_fill is not None:
            self.proc_plot.addItem(self.proc_selector_fill)

        if hasattr(self, "hover_marker") and self.hover_marker is not None:
            self.proc_plot.addItem(self.hover_marker)

        if hasattr(self, "manual_spike_scatter") and self.manual_spike_scatter is not None:
            self.proc_plot.addItem(self.manual_spike_scatter)
    def clear_spikes(self):
        """Clear all spikes for the current signal."""
        if self.proc_key not in self.spike_info:
            self.show_info("No spikes to clear.", title="Info")
            return

        info = self.spike_info[self.proc_key]
        info['auto_times'] = None
        info['auto_amps'] = None
        info['manual_times'] = []
        info['manual_amps'] = []

        self.update_manual_spike_scatter()
        self.plot_detected_spikes()

    #overlay
    def overlay_selected_signals(self):
        """Overlay multiple selected signals in raw_plot, with clickable legend toggle."""
        if self.baseline_curve is not None:
            self.raw_plot.removeItem(self.baseline_curve)
            self.baseline_curve = []

        selected_items = self.file_tree.selectedItems()
        if not selected_items:
            QtWidgets.QMessageBox.warning(self, "Warning", "No signals selected.")
            return

        self.raw_plot.clear()
        legend = self.raw_plot.plotItem.legend
        if legend is not None:
            legend.clear()

        colors = ['b', 'r', 'm', 'c', 'y', 'k']
        color_cycle = iter(colors * 10)

        self.overlay_curves = {}

        for item in selected_items:
            key = item.data(0, QtCore.Qt.UserRole)
            if key not in self.signals:
                continue

            signal = self.signals[key]
            curve = self._plot_signal(signal, self.raw_plot, color=next(color_cycle), name=key, return_item=True)
            self.overlay_curves[key] = curve

        # Set last selected as current
        last_key = selected_items[-1].data(0, QtCore.Qt.UserRole)
        if last_key in self.signals:
            self.current_key = last_key
            self.raw_signal = self.signals[last_key]
            self.proc_signal = None

        # Bind legend label click to toggle curve visibility
        if legend is not None:
            for sample in legend.items:
                _, label = sample
                text = label.text
                if text in self.overlay_curves:
                    curve = self.overlay_curves[text]
                    label.mousePressEvent = partial(toggle_curve_visibility, curve=curve)

        self.update_legend_visibility()

        # renew current_key
        last_key = selected_items[-1].data(0, QtCore.Qt.UserRole)
        if last_key in self.signals:
            self.current_key = last_key
            self.raw_signal = self.signals[last_key]
            self.proc_signal = None

        # click and hide
        for sample in legend.items:
            _, label = sample
            text = label.text
            if text not in self.overlay_curves:
                continue
            curve = self.overlay_curves[text]
            label.mousePressEvent = lambda event, c=curve: c.setVisible(not c.isVisible())

        self.update_legend_visibility()  # click box depend

    #Utility
    def create_processed_signal(self, original_key, processed_magnitude, suffix):
        from neo.core import AnalogSignal
        original_signal = self.signals[original_key]

        new_signal = AnalogSignal(
            processed_magnitude.reshape(-1, 1),
            units=original_signal.units,
            sampling_rate=original_signal.sampling_rate,
            t_start=original_signal.t_start,
            name=f"{original_signal.name}_{suffix}" if original_signal.name else None
        )

        new_key = self._generate_unique_name(f"{original_key}_{suffix}")
        self.signals[new_key] = new_signal
        self.source_map[new_key] = original_key  # orign

        item_new = QtWidgets.QTreeWidgetItem([new_key])
        item_new.setData(0, QtCore.Qt.UserRole, new_key)
        self.file_tree.addTopLevelItem(item_new)

        return new_key, new_signal
         
    #size reduce
    def apply_downsample(self):
        selected_items = self.file_tree.selectedItems()
        if not selected_items:
            QtWidgets.QMessageBox.warning(self, "Warning", "No signals selected for downsampling.")
            return

        factor = self.spin_down.value()

        first_key = None
        first_signal = None

        for item in selected_items:
            key = item.data(0, QtCore.Qt.UserRole)
            if key not in self.signals:
                continue

            original_signal = self.signals[key]
            new_magnitude = original_signal.magnitude[::factor]

            # sampling_rate remove factor
            original_signal.sampling_rate /= factor

            new_key, new_signal = self.create_processed_signal(key, new_magnitude, f"down{factor}x")

            if first_key is None:
                first_key, first_signal = new_key, new_signal

        if first_key:
            self.finalize_processing(first_key, first_signal, source_key=first_key)

    #smooth
    def apply_smooth(self):
        selected_items = self.file_tree.selectedItems()
        if not selected_items:
            QtWidgets.QMessageBox.warning(self, "Warning", "No signals selected.")
            return

        sigma_lambda = self.spin_sigma_lambda.value()
        sigma_p = self.spin_sigma_p.value()
        lam = self.base_lambda * (10 ** sigma_lambda)
        p = self.base_p * (10 ** sigma_p)

        first_key = None
        first_signal = None

        for item in selected_items:
            key = item.data(0, QtCore.Qt.UserRole)
            if key not in self.signals:
                continue

            signal = self.signals[key]
            baseline = als_baseline(signal.magnitude.flatten(), lam=lam, p=p)
            detrended = signal.magnitude.flatten() - baseline

            new_key, new_signal = self.create_processed_signal(key, detrended, "als")

            if first_key is None:
                first_key, first_signal = new_key, new_signal

        if first_key:
            self.finalize_processing(first_key, first_signal, source_key=first_key)

    def apply_zero_baseline(self):
        selected_items = self.file_tree.selectedItems()
        if not selected_items:
            QtWidgets.QMessageBox.warning(self, "Warning", "No signals selected.")
            return

        method = self.combo_baseline_mode.currentText()

        first_key, first_signal = None, None

        for item in selected_items:
            key = item.data(0, QtCore.Qt.UserRole)
            if key not in self.signals or "als" not in key:
                QtWidgets.QMessageBox.warning(self, "Warning", f"{key} is not an ALS-processed signal.")
                continue

            signal = self.signals[key]
            y = signal.magnitude.flatten()

            self.current_key = key
            sample_values = self.extract_data_from_time_range()

            if sample_values is None or len(sample_values) == 0:
                sample_values = y

            baseline_val = extract_baseline_value(sample_values, method=method)
            centered = y - baseline_val
            centered = np.clip(centered, 0, None)

            new_key, new_signal = self.create_processed_signal(key, centered, "zeroed")

            if first_key is None:
                first_key, first_signal = new_key, new_signal

        if first_key:
            self.finalize_processing(first_key, first_signal, source_key=first_key)

    #baseline time selector
    def toggle_time_selector(self, state):
        use_time = state == Qt.Checked

        if hasattr(self, "time_selector") and self.time_selector is not None:
            self.time_selector.setVisible(use_time)

        if hasattr(self, "proc_selector_fill") and self.proc_selector_fill is not None:
            self.proc_selector_fill.setVisible(use_time)

        self.raw_plot.plotItem.vb.setMouseEnabled(x=not use_time, y=not use_time)
        self.proc_plot.plotItem.vb.setMouseEnabled(x=not use_time, y=not use_time)

        if not use_time:
            self.raw_plot.plotItem.vb.enableAutoRange(axis='xy')
            self.proc_plot.plotItem.vb.enableAutoRange(axis='xy')

        self.update_legend_visibility()

    def extract_data_from_time_range(self):
        if self.current_key is None or "als" not in self.current_key:
            return None

        signal = self.signals[self.current_key]
        t = signal.times.rescale('s').magnitude.flatten()
        y = signal.magnitude.flatten()

        # from showing range
        if hasattr(self, "time_selector") and self.time_selector is not None:
            x0, x1 = self.time_selector.getRegion()
        else:
            x0, x1 = self.raw_plot.viewRange()[0]

        mask = (t >= x0) & (t <= x1)
        return y[mask] if np.any(mask) else y

    #mouse event
    def bind_mouse_events(self, plot):
        plot.mousePressEvent = lambda e: self.handle_mouse_event(e, plot)
        plot.mouseReleaseEvent = lambda e: self.handle_mouse_release(e, plot)

        if plot == self.proc_plot:
            plot.scene().sigMouseMoved.connect(self.proc_plot_mouse_moved)

    def proc_plot_mouse_moved(self, pos):
        if self.proc_plot.plotItem.vb.sceneBoundingRect().contains(pos):
            mouse_point = self.proc_plot.plotItem.vb.mapSceneToView(pos)
            x = mouse_point.x()
            y = mouse_point.y()

            if self.chk_manual_spike.isChecked():
                self.update_hover_marker(x, y)
            else:
                self.hover_marker.setVisible(False)

    def handle_mouse_event(self, event, plot):
        if self.chk_manual_spike.isChecked() and plot == self.proc_plot:
            self.handle_manual_spike_click(event)
        elif self.chk_use_time.isChecked() and event.button() == Qt.LeftButton:
            self.start_time_select(event, plot)
        else:
            super(pg.PlotWidget, plot).mousePressEvent(event)

    def handle_mouse_release(self, event, plot):
        if self.chk_use_time.isChecked() and event.button() == Qt.LeftButton:
            self.end_time_select(event, plot)
        else:
            super(pg.PlotWidget, plot).mouseReleaseEvent(event)

    def start_time_select(self, event, plot):
        if event.button() != Qt.LeftButton:
            return
        
        if not self.chk_use_time.isChecked():
            return

        mouse_point = plot.plotItem.vb.mapToView(event.pos())
        self._drag_start = mouse_point.x()

        region = [self._drag_start, self._drag_start + 0.01]

        self.time_selector = self.ensure_selector("time_selector", self.raw_plot)
        self.proc_selector_fill = self.ensure_selector("proc_selector_fill", self.proc_plot)
        self.update_selector_region(region)

        self.time_selector.setVisible(True)

        # sync proc_plot
        if hasattr(self, "proc_selector_fill"):
            self.proc_selector_fill.setRegion(region)
            self.proc_selector_fill.setVisible(True)
    
    def end_time_select(self, event, plot):
        if event.button() != Qt.LeftButton or self._drag_start is None:
            return
        
        if not self.chk_use_time.isChecked():
            return

        mouse_point = plot.plotItem.vb.mapToView(event.pos())
        x0 = self._drag_start
        x1 = mouse_point.x()

        if abs(x1 - x0) < 1e-3:
            x1 = x0 + 0.01

        region = [min(x0, x1), max(x0, x1)]

        self.time_selector = self.ensure_selector("time_selector", self.raw_plot)
        self.proc_selector_fill = self.ensure_selector("proc_selector_fill", self.proc_plot)
        self.update_selector_region(region)

        self._drag_start = None
        self.sync_xrange(plot)

        # sync proc_plot
        if hasattr(self, "proc_selector_fill"):
            self.proc_selector_fill.setRegion(region)
            self.proc_selector_fill.setVisible(True)

    def on_mouse_drag_move(self, pos):
        if self.proc_plot.plotItem.vb.sceneBoundingRect().contains(pos):
            mouse_point = self.proc_plot.plotItem.vb.mapSceneToView(pos)
            x = mouse_point.x()
            y = mouse_point.y()

            if self.chk_manual_spike.isChecked():
                self.update_hover_marker(x, y)
            else:
                self.hover_marker.setVisible(False)

        # time select
        if self._drag_start is not None and self.chk_use_time.isChecked():
            for plot in [self.raw_plot, self.proc_plot]:
                vb = plot.plotItem.vb
                if vb.sceneBoundingRect().contains(pos):
                    mouse_point = vb.mapToView(pos)
                    x = mouse_point.x()
                    x0 = self._drag_start
                    x1 = x
                    region = [min(x0, x1), max(x0, x1)]
                    self.time_selector = self.ensure_selector("time_selector", self.raw_plot)
                    self.proc_selector_fill = self.ensure_selector("proc_selector_fill", self.proc_plot)
                    self.update_selector_region(region)
                    break

    def sync_xrange(self, source_plot):
        """sync x axis"""
        x_min, x_max = source_plot.viewRange()[0]
        target_plot = self.proc_plot if source_plot == self.raw_plot else self.raw_plot
        target_plot.setXRange(x_min, x_max, padding=0)

    def undo_manual_spike(self):
        """Undo the last manually added spike."""
        info = self.get_spike_info()
        if info is None:
            return

        manual_times = info.get('manual_times', [])
        manual_amps = info.get('manual_amps', [])

        if not manual_times:
            return

        manual_times.pop()
        manual_amps.pop()
        self.update_manual_spike_scatter()

    #spike detect
    def get_spike_info(self, key=None, create=False):
        """Get spike_info dict for the given key (default current), optionally create if missing."""
        if key is None:
            key = self.proc_key

        if key is None:
            return None

        if create:
            self.spike_info.setdefault(key, {})
            info = self.spike_info[key]
            info.setdefault('auto_times', None)
            info.setdefault('auto_amps', None)
            info.setdefault('manual_times', [])
            info.setdefault('manual_amps', [])
            return info
        else:
            return self.spike_info.get(key, None)

    def update_auto_spikes(self, times, amps, key=None):
        """Update auto-detected spikes for the given key."""
        info = self.get_spike_info(key=key, create=True)
        info['auto_times'] = times
        info['auto_amps'] = amps

    def update_manual_spike_scatter(self):
        """Update manual spike scatter points."""
        info = self.get_spike_info()
        if info is None:
            self.manual_spike_scatter.clear()
            return

        manual_times = info.get('manual_times', [])
        manual_amps = info.get('manual_amps', [])

        spots = [{'pos': (x, y)} for x, y in zip(manual_times, manual_amps)]
        self.manual_spike_scatter.setData(spots)

    def update_spike_mode_link(self, state):
        if state == Qt.Checked:
            self.raw_plot.setXLink(self.proc_plot)
            self.raw_plot.plotItem.vb.setMouseEnabled(x=False, y=False)
            self.proc_plot.plotItem.vb.setMouseEnabled(x=True, y=False)
        else:
            self.raw_plot.setXLink(None)
            self.raw_plot.plotItem.vb.setMouseEnabled(x=True, y=True)
            self.proc_plot.plotItem.vb.setMouseEnabled(x=True, y=True)
        
    def estimate_freq_from_isi(self, y, fs):
        """
        Estimate frequency range based on inter-spike intervals (ISI).
        This function detects peaks, computes ISIs, and converts to frequency.
        """
        peaks, _ = find_peaks(y, prominence=np.std(y))
        if len(peaks) < 2:
            return 20, 250, "Not enough spikes. Defaulting to 20–250 Hz."

        isi_sec = np.diff(peaks) / fs  # ISI in seconds
        typical_period = np.median(isi_sec)  # Use median for robustness
        f_center = min(1.0 / typical_period, 250)  # Cap frequency to 250 Hz
        f_low = int(max(1, f_center / 2))
        f_high = int(min(fs / 2, f_center * 2))

        comment = f"Estimated center freq: {f_center:.1f} Hz, set range: {f_low}–{f_high} Hz"
        return f_low, f_high, comment

    def recommend_freq_and_wavelet(self):
        """
        Always estimate and set ISI-based freq band on click.
        If “Auto mother wavelet” is checked, also pick/adjust the wavelet.
        """
        y = self.proc_signal.magnitude.flatten()
        fs = float(self.proc_signal.sampling_rate.rescale('Hz').magnitude)

        # Step 1: Estimate ISI-based band
        f_low_est, f_high_est, f_comment = self.estimate_freq_from_isi(y, fs)
        self.freq_low.setValue(f_low_est)
        self.freq_high.setValue(f_high_est)

        comment = "[Step 1] ISI-based frequency range:\n" + f_comment

        # Step 2: Wavelet suggestion based purely on ISI band
        if self.checkbox_auto_wavelet.isChecked():
            if f_high_est <= 40:
                wavelet = 'mexh'
            elif f_high_est <= 80:
                wavelet = 'gaus1'
            else:
                wavelet = 'cmor1.5-1.0'

            comment += (
                f"\n\n[Step 2] Based on ISI-estimated freq range ({f_low_est}-{f_high_est} Hz), "
                f"selected wavelet: '{wavelet}'."
            )

            # Apply wavelet choice without triggering signal
            self.combo_wavelet.blockSignals(True)
            self.combo_wavelet.setCurrentText(wavelet)
            self.combo_wavelet.blockSignals(False)

        # Step 3: show summary dialog
        QtWidgets.QMessageBox.information(
            self,
            "Wavelet & Frequency Recommendation",
            comment
        )

    def on_wavelet_changed(self, wavelet_name):
        """
        If user selects a real wavelet, auto-update frequency spinboxes to safe range.
        """
        safe_ranges = {
            'gaus1': (30, 100),
            'mexh': (10, 80),
            'morl': (5, 80),
        }

        if wavelet_name in safe_ranges:
            lo, hi = safe_ranges[wavelet_name]
            self.freq_low.setValue(lo)
            self.freq_high.setValue(hi)

    def apply_detect(self):
        """
        Apply wavelet-based spike detection based on current GUI settings.
        Ensures that real wavelets (without center freq) are not mapped incorrectly.
        """
        if self.proc_key is None or self.proc_key not in self.signals:
            QtWidgets.QMessageBox.warning(self, "Warning", "No processed signal found. Run ALS or baseline removal first.")
            return

        signal = self.signals[self.proc_key]
        self.proc_signal = signal  # refresh

        y = signal.magnitude.flatten()
        fs = float(signal.sampling_rate.rescale('Hz').magnitude)

        wavelet_name = self.combo_wavelet.currentText()
        f_low = self.freq_low.value()
        f_high = self.freq_high.value()
        threshold_std = self.wavelet_thresh.value()

        # Get center frequency if available (optional), else fallback to default scaling safety
        try:
            wavelet_obj = pywt.ContinuousWavelet(wavelet_name)
            center_freq = wavelet_obj.center_frequency or 1.0  # safe default
        except Exception:
            center_freq = 1.0  # fallback for robustness

        max_allowed_freq = fs * center_freq
        if f_high > max_allowed_freq:
            f_high = int(max_allowed_freq)
            QtWidgets.QMessageBox.warning(
                self, "Frequency Clipped",
                f"High frequency clipped to {f_high} Hz to ensure valid wavelet scales."
            )

        # Validate frequency range against center_freq * fs
        max_allowed_freq = center_freq * fs
        if f_high > max_allowed_freq:
            f_high = int(max_allowed_freq)
            QtWidgets.QMessageBox.warning(
                self, "Frequency Clipped",
                f"High frequency clipped to {f_high} Hz to ensure valid wavelet scales."
            )

        # Call core detection
        try:
            spike_idx = wavelet_spike_detect(
                y, fs,
                freq_range=(f_low, f_high),
                threshold_std=threshold_std,
                wavelet_name=wavelet_name
            )

            if spike_idx is None or len(spike_idx) == 0:
                QtWidgets.QMessageBox.information(self, "No Spikes", "No spikes detected with current settings.")
                return

            auto_times = self.get_relative_time(signal, spike_idx, mode='index')
            auto_amps = y[spike_idx]

            if len(auto_times) == 0 or len(auto_amps) == 0:
                QtWidgets.QMessageBox.information(self, "No Spikes", "No spikes detected in the current settings.")
                return

            self.update_auto_spikes(auto_times, auto_amps)
            self.plot_detected_spikes()

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Detection Error", f"Spike detection failed:\n{e}")

    def handle_manual_spike_click(self, event):
        """Handle left click to add manual spike, right click to delete nearest spike."""
        if event.button() not in (Qt.LeftButton, Qt.RightButton):
            return

        if self.proc_signal is None or self.hover_spike is None:
            return

        click_x, click_y = self.hover_spike

        if event.button() == Qt.LeftButton:
            info = self.get_spike_info(create=True)
            info['manual_times'].append(click_x)
            info['manual_amps'].append(click_y)
            self.update_manual_spike_scatter()

        elif event.button() == Qt.RightButton:
            self.delete_nearest_spike(click_x, click_y)

    def apply_filter(self):
        """Apply amplitude and clustering filter to detected spikes for the current signal."""
        info = self.get_spike_info()
        if info is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "No spike candidates to filter.")
            return

        auto_times = info.get('auto_times', None)
        auto_amps = info.get('auto_amps', None)

        if auto_times is None or auto_amps is None or len(auto_times) == 0:
            QtWidgets.QMessageBox.warning(self, "Warning", "No detected spikes to filter.")
            return

        min_interval = self.minIntervalSpinBox.value() / 1000.0
        min_amp = self.minAmplitudeSpinBox.value()
        std_factor = self.relThresholdSpinBox.value()

        filtered_times, filtered_amps = cluster_spikes(
            auto_times, auto_amps,
            min_interval=min_interval,
            min_amplitude=min_amp,
            std_factor=std_factor
        )

        self.update_auto_spikes(filtered_times, filtered_amps)
        self.plot_detected_spikes()

    def plot_detected_spikes(self):
        """Plot processed signal with auto spikes; manual spikes handled separately."""
        if self.proc_signal is None:
            return

        self.clear_proc_plot()

        signal = self.proc_signal
        t = self.get_relative_time(signal, signal.times.rescale('s').magnitude, mode='time')
        y = signal.magnitude.flatten()

        self.proc_plot.plot(t, y, pen=pg.mkPen('b'))

        info = self.spike_info.get(self.proc_key, {})

        auto_times = info.get('auto_times', None)
        auto_amps = info.get('auto_amps', None)

        if auto_times is not None and len(auto_times) > 0:
            self.proc_plot.plot(
                auto_times, auto_amps,
                pen=None, symbol='o', symbolBrush='r', symbolSize=6, name="Auto Spikes"
            )

        # send to manual_spike_scatter
        self.update_manual_spike_scatter()

        self.reset_view()

    def get_relative_time(self, signal, indices_or_times, mode='index'):
        """Convert index or time array into relative time"""
        if mode == 'index':
            fs = float(signal.sampling_rate.rescale('Hz'))
            return np.array(indices_or_times) / fs
        elif mode == 'time':
            t0 = float(signal.t_start.rescale('s'))
            return np.array(indices_or_times) - t0
        else:
            raise ValueError("mode must be 'index' or 'time'")

    def update_hover_marker(self, x, y):
        """In full signal, find nearest local max around mouse, update hover marker."""
        if self.proc_signal is None:
            self.hover_marker.setVisible(False)
            self.hover_spike = None
            return

        t = self.get_relative_time(self.proc_signal, self.proc_signal.times.rescale('s').magnitude, mode='time')
        ydata = self.proc_signal.magnitude[:, 0]

        pixel_scale_x = (self.proc_plot.plotItem.vb.viewRange()[0][1] - self.proc_plot.plotItem.vb.viewRange()[0][0]) / self.proc_plot.width()
        pixel_scale_y = (self.proc_plot.plotItem.vb.viewRange()[1][1] - self.proc_plot.plotItem.vb.viewRange()[1][0]) / self.proc_plot.height()

        tol_x = 20 * pixel_scale_x
        tol_y = 20 * pixel_scale_y

        mask = (np.abs(t - x) <= tol_x) & (np.abs(ydata - y) <= tol_y)

        if not np.any(mask):
            self.hover_marker.setVisible(False)
            self.hover_spike = None
            return

        candidate_idx = np.where(mask)[0]
        best_idx = candidate_idx[np.argmax(ydata[candidate_idx])]

        best_x = t[best_idx]
        best_y = ydata[best_idx]

        self.hover_marker.setData([{'pos': (best_x, best_y)}])
        self.hover_marker.setVisible(True)
        self.hover_spike = (best_x, best_y)

    def update_manual_spike_scatter(self):
        """Update manual spike scatter points."""
        if self.proc_key not in self.spike_info:
            self.manual_spike_scatter.clear()
            return

        info = self.spike_info[self.proc_key]
        manual_times = info.get('manual_times', [])
        manual_amps = info.get('manual_amps', [])

        spots = [{'pos': (x, y)} for x, y in zip(manual_times, manual_amps)]
        self.manual_spike_scatter.setData(spots)

    def delete_nearest_spike(self, click_x, click_y):
        """Delete the nearest spike (auto or manual) under cursor if close enough."""
        info = self.get_spike_info()
        if info is None:
            return

        auto_times = info.get('auto_times', [])
        auto_amps = info.get('auto_amps', [])
        manual_times = info.get('manual_times', [])
        manual_amps = info.get('manual_amps', [])

        all_times = np.concatenate([auto_times, manual_times])
        all_amps = np.concatenate([auto_amps, manual_amps])

        if len(all_times) == 0:
            return

        distances = np.sqrt((all_times - click_x) ** 2 + (all_amps - click_y) ** 2)
        min_idx = np.argmin(distances)

        # max distance
        max_distance = 0.5  
        if distances[min_idx] > max_distance:
            return

        # auto or manual
        if min_idx < len(auto_times):
            info['auto_times'] = np.delete(auto_times, min_idx)
            info['auto_amps'] = np.delete(auto_amps, min_idx)
        else:
            idx = min_idx - len(auto_times)
            manual_times.pop(idx)
            manual_amps.pop(idx)

        self.plot_detected_spikes()

    def update_show_raw(self):
        """Show or hide the original raw signal associated with the current key."""
        self.raw_plot.clear()

        if not self.chk_show_raw.isChecked() or self.current_key is None:
            return

        # get raw
        base_key = self.current_key
        while base_key in self.source_map:
            base_key = self.source_map[base_key]

        # remove version: "_v1" "_v2"
        base_key = base_key.split('_v')[0]

        raw_signal = self.raw_signals.get(base_key)

        if raw_signal is not None:
            t = (raw_signal.times - raw_signal.t_start).rescale('s').magnitude.flatten()
            y = raw_signal.magnitude.flatten()
            self.raw_plot.plot(t, y, pen=pg.mkPen('k'))
            self.reset_raw_view()
            return

        QtWidgets.QMessageBox.warning(
            self, "Missing Raw Data",
            f"Original raw signal for '{self.current_key}' not found."
        )

    def update_raw_plot(self):
        """Redraw the raw signal and spike points."""
        self.raw_plot.clear()

        if self.raw_signal is None:
            return

        t = (self.raw_signal.times - self.raw_signal.t_start).rescale('s').magnitude.flatten()
        y = self.raw_signal.magnitude.flatten()

        self.raw_plot.plot(t, y, pen=pg.mkPen('k'))

        if self.chk_show_spikes.isChecked() and self.current_key:
            self.plot_spike_times(self.raw_plot, self.current_key)

        self.reset_raw_view()

    def update_proc_plot(self):
        """Redraw the processed signal and spike points."""
        self.proc_plot.clear()

        if self.proc_signal is None:
            return

        t = (self.proc_signal.times - self.proc_signal.t_start).rescale('s').magnitude.flatten()
        y = self.proc_signal.magnitude[:, 0]

        self.proc_plot.plot(t, y, pen=pg.mkPen('b'))

        if self.chk_show_spikes.isChecked() and self.current_key:
            self.plot_spike_times(self.proc_plot, self.current_key)

        self.reset_proc_view()

    #AP
    def apanalysis(self):
        """Analyze spikes and export one row per spike to a single CSV file with save dialog."""
        from scipy.signal import find_peaks, peak_widths

        selected_items = self.file_tree.selectedItems()
        if not selected_items:
            QtWidgets.QMessageBox.warning(self, "Warning", "No signals selected for AP analysis.")
            return

        all_spike_rows = []
        missing_prominence_count = 0
        total_spike_count = 0

        for item in selected_items:
            key = item.data(0, QtCore.Qt.UserRole)
            signal = self.signals.get(key, None)
            if signal is None:
                continue

            if key not in self.spike_info or self.spike_info[key].get('auto_times') is None:
                self.proc_key = key
                self.proc_signal = signal
                self.apply_detect()
                self.apply_filter()

            signal_y = signal.magnitude.flatten()
            fs = float(signal.sampling_rate.rescale('Hz'))
            dt = 1.0 / fs
            t_rel = (signal.times - signal.t_start).rescale('s').magnitude.flatten()
            t_abs = signal.times.rescale('s').magnitude.flatten()

            sweep_offset = 0
            if '_sweep' in key:
                try:
                    sweep_index = int(key.split('_sweep')[-1])
                    duration = t_abs[-1] - t_abs[0]
                    sweep_offset = sweep_index * duration
                except:
                    pass
            t_abs += sweep_offset

            info = self.get_spike_info(key=key, create=True)
            spike_times = []
            if info.get('manual_times'):
                spike_times.extend(info['manual_times'])
            if info.get('auto_times') is not None:
                spike_times.extend(info['auto_times'])

            if len(spike_times) == 0:
                continue

            spike_times = np.sort(np.array(spike_times))

            rise_times, decay_times = [], []
            for spk_time in spike_times[:10]:
                idx_center = np.argmin(np.abs(t_rel - spk_time))
                idx_start = max(0, idx_center - int(0.005 / dt))
                idx_end = min(len(signal_y), idx_center + int(0.005 / dt))
                ep_y = signal_y[idx_start:idx_end]
                if len(ep_y) == 0:
                    continue
                dy = np.gradient(ep_y)
                peak_idx = np.argmax(ep_y)
                if peak_idx > 1:
                    rise_idx = np.argmax(dy[:peak_idx])
                    rise_time = (peak_idx - rise_idx) * dt * 1000
                    rise_times.append(rise_time)
                if peak_idx < len(dy) - 2:
                    decay_idx = np.argmin(dy[peak_idx:])
                    decay_time = decay_idx * dt * 1000
                    decay_times.append(decay_time)

            window_pre_ms = min(np.median(rise_times) * 1.5, 50) if rise_times else 10
            window_post_ms = min(np.median(decay_times) * 1.5, 50) if decay_times else 20
            baseline = extract_baseline_value(signal_y, method='mode') if 'zeroed' in key.lower() else 0.0
            sweep_id = key.split('_sweep')[-1] if '_sweep' in key else '0'

            for i, spk_time in enumerate(spike_times):
                total_spike_count += 1
                idx_center = np.argmin(np.abs(t_rel - spk_time))
                idx_start = max(0, idx_center - int(window_pre_ms / 1000 / dt))
                idx_end = min(len(signal_y), idx_center + int(window_post_ms / 1000 / dt))
                ep_y = signal_y[idx_start:idx_end]
                if len(ep_y) == 0:
                    continue

                peaks, properties = find_peaks(ep_y, prominence=0)
                if len(peaks) == 0:
                    peak_idx = np.argmax(ep_y)
                    prominence = np.nan
                    missing_prominence_count += 1
                else:
                    best_peak = np.argmax(ep_y[peaks])
                    peak_idx = peaks[best_peak]
                    try:
                        prominence = properties['prominences'][best_peak]
                    except:
                        prominence = np.nan
                        missing_prominence_count += 1

                amp = ep_y[peak_idx] + baseline
                width = peak_widths(ep_y, [peak_idx], rel_height=0.5)[0][0] * dt * 1000

                half_amp = ep_y[peak_idx] / 2.0
                rise_candidates = np.where(ep_y[:peak_idx] <= half_amp)[0]
                rise_time = (peak_idx - rise_candidates[-1]) * dt * 1000 if len(rise_candidates) > 0 else np.nan
                decay_candidates = np.where(ep_y[peak_idx:] <= half_amp)[0]
                decay_time = decay_candidates[0] * dt * 1000 if len(decay_candidates) > 0 else np.nan
                isi = (spike_times[i] - spike_times[i - 1]) * 1000 if i > 0 else np.nan

                spike_data = {
                    'Source': key,
                    'Sweep': f'sweep{sweep_id}',
                    'Spike Index': i + 1,
                    'Spike Relative Time (ms)': t_rel[idx_center] * 1000,
                    'Spike Absolute Time (ms)': t_abs[idx_center] * 1000,
                    'Width (ms)': width,
                    'Amplitude (mV)': amp,
                    'Prominence (mV)': prominence,
                    'Rise Time (ms)': rise_time,
                    'Decay Time (ms)': decay_time,
                    'ISI (ms)': isi
                }
                all_spike_rows.append(spike_data)

        if not all_spike_rows:
            QtWidgets.QMessageBox.information(self, "Export Done", "AP Analysis finished.\nNo spikes were detected.")
            return

        # name=key
        first_key = selected_items[0].data(0, QtCore.Qt.UserRole)
        base_name = first_key.split('_sweep')[0]
        suggested_name = f"{base_name}_apanalysis.csv"

        # path
        default_dir = os.path.dirname(self.path_map.get(first_key, os.getcwd()))

        first_key = selected_items[0].data(0, QtCore.Qt.UserRole)
        base_name = first_key.split('_sweep')[0]
        suggested_name = f"{base_name}_apanalysis.csv"

        real_path = self.path_map.get(first_key)
        default_dir = os.path.dirname(real_path) if real_path and os.path.exists(real_path) else os.getcwd()

        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save AP Analysis CSV",
            os.path.join(default_dir, suggested_name), 
            "CSV Files (*.csv)"
        )

        if not file_path:
            return  # cancel

        self.last_folder = os.path.dirname(file_path)
        df = pd.DataFrame(all_spike_rows)
        df.to_csv(file_path, index=False)

        message = f"AP Analysis completed!\n\nTotal spikes analyzed: {total_spike_count}"
        if missing_prominence_count > 0:
            message += f"\nSpikes missing prominence: {missing_prominence_count}"
        QtWidgets.QMessageBox.information(self, "Export Done", message)

    #batch
    def batch_preprocess(self):
        """Batch preprocess all sweeps and save as fully standard NIX format."""
        import quantities as pq
        from neo import Block, Segment, AnalogSignal, Event
        import neo

        input_folder = self.select_folder("Select Input Folder for Batch Preprocessing")
        if not input_folder:
            return

        output_folder = self.select_folder("Select Output Folder to Save Processed Data")
        if not output_folder:
            return

        self.last_preprocess_input_folder = input_folder
        self.last_preprocess_output_folder = output_folder

        file_list = self.get_all_files(input_folder, suffixes=('.abf', '.h5'))
        if not file_list:
            self.show_warning("No .abf or .h5 files found.")
            return

        self.create_progress_dialog("Batch Preprocessing...", len(file_list))

        success, failed = [], []

        for idx, filepath in enumerate(file_list):
            try:
                rel_path = os.path.relpath(filepath, input_folder)
                save_path = os.path.join(output_folder, rel_path).replace('.abf', '.h5')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                if os.path.exists(save_path):
                    reply = QtWidgets.QMessageBox.question(
                        self,
                        "Overwrite Confirmation",
                        f"File already exists:\n{save_path}\n\nDo you want to overwrite it?",
                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                        QtWidgets.QMessageBox.No
                    )
                    if reply == QtWidgets.QMessageBox.No:
                        continue

                ext = os.path.splitext(filepath)[1].lower()
                if ext == '.abf':
                    reader = neo.io.AxonIO(filepath)
                    block = reader.read_block()
                elif ext == '.h5':
                    if self.is_nix_file(filepath):
                        reader = NixIO(filepath, mode='ro')
                        block = reader.read_block()
                    else:
                        raise ValueError(f"Non-NIX h5 file skipped: {filepath}")
                else:
                    raise ValueError(f"Unsupported file extension: {filepath}")

                if block is None or not block.segments:
                    raise ValueError(f"No valid segments found in {filepath}")

                new_block = Block(name=f"block_{os.path.basename(filepath)}")

                for i, segment in enumerate(block.segments):
                    if not segment.analogsignals:
                        continue

                    signal = segment.analogsignals[0]
                    y = signal.magnitude.flatten()
                    t = (signal.times - signal.t_start).rescale('s').magnitude.flatten()
                    sampling_rate = float(signal.sampling_rate.rescale('Hz'))

                    raw_signal = AnalogSignal(
                        y.reshape(-1, 1),
                        units=signal.units,
                        sampling_rate=sampling_rate * pq.Hz,
                        t_start=0 * pq.s,
                        name="raw"
                    )

                    factor = self.spin_down.value()
                    if factor <= 0:
                        factor = 1

                    y_ds = y[::factor]
                    t_ds = t[::factor]

                    lam = self.base_lambda * (10 ** self.spin_sigma_lambda.value())
                    p = self.base_p * (10 ** self.spin_sigma_p.value())
                    baseline = als_baseline(y_ds, lam=lam, p=p)
                    detrended = y_ds - baseline

                    downsampled_sampling_rate = sampling_rate / factor

                    processed_signal = AnalogSignal(
                        detrended.reshape(-1, 1),
                        units=signal.units,
                        sampling_rate=downsampled_sampling_rate * pq.Hz,
                        t_start=0 * pq.s,
                        name="processed"
                    )

                    new_segment = Segment(name=f"sweep{i}")
                    new_segment.analogsignals.append(raw_signal)
                    new_segment.analogsignals.append(processed_signal)

                    new_block.segments.append(new_segment)

                with NixIO(filename=save_path, mode='ow') as writer:
                    writer.write_block(new_block)

                success.append(filepath)

            except Exception as e:
                print(f"Error processing {filepath}: {e}")
                failed.append(filepath)

            self.update_progress(idx + 1)

        self.batch_progress_bar.setVisible(False)

        summary = f"Batch Preprocessing Finished.\n\nSuccess: {len(success)} files\nFailed: {len(failed)} files"
        if failed:
            summary += "\n\nFailed files:\n" + "\n".join(os.path.basename(f) for f in failed)
        self.show_info(summary, title="Batch Preprocessing Done")

    def preprocess_segment(self, segment):
        """Preprocess a segment: downsampling and ALS baseline correction."""
        if segment is None or not segment.analogsignals:
            raise ValueError("Segment is empty or missing analog signals.")

        signal = segment.analogsignals[0]
        y = signal.magnitude.flatten()
        t = (signal.times - signal.t_start).rescale('s').magnitude.flatten()

        # Downsampling
        factor = self.spin_down.value()
        if factor <= 0:
            factor = 1

        y_ds = y[::factor]
        t_ds = t[::factor]

        # ALS
        lam = self.base_lambda * (10 ** self.spin_sigma_lambda.value())
        p = self.base_p * (10 ** self.spin_sigma_p.value())

        baseline = als_baseline(y_ds, lam=lam, p=p)
        detrended = y_ds - baseline

        # spike detect = None
        peaks_info = None

        return detrended, peaks_info, t, y

    def create_progress_dialog(self, title, total_files):
        """new fixed bar"""
        self.total_batch_files = total_files
        self.batch_progress_bar.setMaximum(total_files)
        self.batch_progress_bar.setValue(0)
        self.batch_progress_bar.setVisible(True)
        self.batch_progress_bar.setFormat(f"0 / {total_files}")

    def update_progress(self, current):
        """renew"""
        if hasattr(self, 'batch_progress_bar'):
            self.batch_progress_bar.setValue(current)
            self.batch_progress_bar.setFormat(f"{current} / {self.total_batch_files}")
            if current >= self.total_batch_files:
                self.batch_progress_bar.setVisible(False)

    #pop notice
    def run_batch_processing(self, file_list, per_file_func, title="Processing..."):
        """Unified batch processing framework."""
        self.create_progress_dialog(title, len(file_list))
        success, failed = [], []

        for idx, filepath in enumerate(file_list):
            try:
                per_file_func(filepath)
                success.append(filepath)
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
                failed.append(filepath)
            self.update_progress(idx+1)

        self.batch_progress_bar.setVisible(False)
        return success, failed

    def show_warning(self, message):
        QtWidgets.QMessageBox.warning(self, "Warning", message)

    def show_info(self, message, title="Info"):
        QtWidgets.QMessageBox.information(self, title, message)

    def select_files(self, title, file_filter):
        last_dir = self.settings.value('lastDir', QDir.homePath())
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(self, title, last_dir, file_filter)
        if files:
            self.settings.setValue('lastDir', os.path.dirname(files[0]))
        return files

    def select_folder(self, title):
        last_dir = self.settings.value('lastDir', QDir.homePath())
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, title, last_dir)
        if folder:
            self.settings.setValue('lastDir', folder)
        return folder

    def clear_plot(self, plot, selector=None):
        plot.clear()
        if selector and selector not in plot.items():
            plot.addItem(selector)


# ALS
from scipy.sparse import spdiags, diags
from scipy.sparse.linalg import spsolve
import warnings

def als_baseline(y, lam=1e4, p=1e-4, niter=10):
    """
    Asymmetric Least Squares baseline correction.
    y: input signal
    lam: smoothness parameter (lambda)
    p: asymmetry parameter
    niter: number of iterations
    """
    L = len(y)

    if L < 10:  # protect small
        warnings.warn("Signal too short for ALS baseline, returning original signal.")
        return y.copy()

    D = diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    w = np.ones(L)

    for _ in range(niter):
        W = spdiags(w, 0, L, L)
        Z = W + lam * D @ D.T
        try:
            z = spsolve(Z, w * y)
        except Exception:
            warnings.warn("ALS solve failed, returning original signal.")
            return y.copy()
        w = p * (y > z) + (1 - p) * (y < z)

    return z

#get baseline height
def extract_baseline_value(y, method='mode', sample_size=10000):
    """get baseline value from signal"""
    y = y.flatten()
    sample = y[:sample_size] if len(y) > sample_size else y

    if method == 'mean':
        return np.mean(sample)
    elif method == 'median':
        return np.median(sample)
    elif method == 'mode':
        hist, bin_edges = np.histogram(sample, bins=100)
        max_bin = np.argmax(hist)
        return (bin_edges[max_bin] + bin_edges[max_bin + 1]) / 2
    else:
        raise ValueError("Invalid method")

#spike detect
def wavelet_spike_detect(signal, fs, freq_range=(80, 250), threshold_std=3, wavelet_name='morl'):
    wavelet = pywt.ContinuousWavelet(wavelet_name)

    try:

        low_f, high_f = freq_range
        if low_f >= high_f or low_f <= 0:
            return np.array([], dtype=int)

        central_freq = pywt.central_frequency(wavelet)
        if central_freq <= 0:
            raise ValueError("Invalid central frequency")

        min_scale = central_freq * fs / high_f
        max_scale = central_freq * fs / low_f

        if min_scale <= 0 or max_scale <= 0:
            raise ValueError("Invalid scale range")

        scale_start = int(np.floor(min_scale))
        scale_end = int(np.ceil(max_scale)) + 1
        if scale_end <= scale_start:
            raise ValueError("Empty scale range")

        scales = np.arange(scale_start, scale_end)

    except Exception:
        # fallback：fix mexh / gaus1
        scales = np.arange(1, 128)

    coeffs, freqs = pywt.cwt(signal, scales, wavelet, sampling_period=1/fs)
    band_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    if not np.any(band_mask):
        return np.array([], dtype=int)

    band_energy = np.abs(coeffs[band_mask, :]).mean(axis=0)
    
    window_size = int(fs * 0.5)
    if window_size < 10:
        window_size = 10

    local_mean = uniform_filter1d(band_energy, size=window_size)
    residual = band_energy - local_mean
    local_std = np.std(residual)

    spike_idx = np.where(residual > threshold_std * local_std)[0]
    return spike_idx

#cluster
def cluster_spikes(spike_times, spike_amps, min_interval=0.03, min_amplitude=0.0, std_factor=0.0):
    spike_times = np.array(spike_times)
    spike_amps = np.array(spike_amps)

    sorted_idx = np.argsort(spike_times)
    times_sorted = spike_times[sorted_idx]
    amps_sorted = spike_amps[sorted_idx]

    filtered_times = []
    filtered_amps = []

    i = 0
    while i < len(times_sorted):
        group_start = i
        while i + 1 < len(times_sorted) and times_sorted[i + 1] - times_sorted[i] < min_interval:
            i += 1
        group_end = i + 1

        group_times = times_sorted[group_start:group_end]
        group_amps = amps_sorted[group_start:group_end]

        # Step 1: SD filter first
        if std_factor > 0 and len(group_amps) > 1:
            mean_amp = np.mean(group_amps)
            std = np.std(group_amps)
            threshold = mean_amp + std_factor * std
            keep_mask = group_amps >= threshold
        else:
            keep_mask = np.ones_like(group_amps, dtype=bool)

        # Step 2: min amplitude
        keep_mask &= group_amps >= min_amplitude

        if np.any(keep_mask):
            keep_idx = np.argmax(group_amps * keep_mask)
            filtered_times.append(group_times[keep_idx])
            filtered_amps.append(group_amps[keep_idx])

        i = group_end

    return np.array(filtered_times), np.array(filtered_amps)

#overlay visibility
def toggle_curve_visibility(event, curve):
    curve.setVisible(not curve.isVisible())

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = LFPAnalyzer()
    win.show()
    sys.exit(app.exec_())