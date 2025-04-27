import sys
import os
import neo
import pywt
import h5py
import numpy as np
import pyqtgraph as pg
from neo.io import NixIO
from PyQt5 import QtWidgets,QtCore,QtGui
from PyQt5.QtCore import QSettings, QDir, Qt
from functools import partial
from scipy import sparse
from scipy.sparse.linalg import spsolve

class LFPAnalyzer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('LFP Data Analyzer')
        self.resize(1200, 800)
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
        self.combo_wavelet.setCurrentText('cmor')

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

        # mother wavelet
        self.combo_wavelet = QtWidgets.QComboBox()
        self.combo_wavelet.addItems(['morl', 'cmor', 'mexh', 'gaus1'])
        wavelet_form.addRow("Mother wavelet:", self.combo_wavelet)

        # Frequency
        freq_range_layout = QtWidgets.QHBoxLayout()
        freq_range_layout.setContentsMargins(0, 0, 0, 0)
        self.freq_low = QtWidgets.QSpinBox()
        self.freq_low.setRange(1, 1000)
        self.freq_low.setValue(80)
        self.freq_high = QtWidgets.QSpinBox()
        self.freq_high.setRange(1, 2000)
        self.freq_high.setValue(250)
        freq_range_layout.addWidget(self.freq_low)
        freq_range_layout.addWidget(QtWidgets.QLabel(" - "))
        freq_range_layout.addWidget(self.freq_high)

        freq_widget = QtWidgets.QWidget()
        freq_widget.setLayout(freq_range_layout)
        wavelet_form.addRow("Freq Range (Hz):", freq_widget)

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
        filter_form.addRow("Threshold (SE):", self.relThresholdSpinBox)

        filter_group.setLayout(filter_form)
        left_layout.addWidget(filter_group)

        # Manual Spike + Show Raw on same line
        manual_layout = QtWidgets.QHBoxLayout()
        self.chk_manual_spike = QtWidgets.QCheckBox("Manual Spike")
        self.chk_manual_spike.setChecked(False)
        self.chk_show_raw = QtWidgets.QCheckBox("Show Origin")
        self.chk_show_raw.setChecked(False)
        self.btn_save_proc = QtWidgets.QPushButton('Save Spike')
        manual_layout.addWidget(self.chk_manual_spike)
        manual_layout.addWidget(self.chk_show_raw)
        manual_layout.addWidget(self.btn_save_proc)

        left_layout.addLayout(manual_layout)
        
        self.chk_show_raw.stateChanged.connect(self.update_show_raw)
        self.btn_save_proc.clicked.connect(self.save_proc_and_spike)

        # Spike analysis window
        window_layout = QtWidgets.QHBoxLayout()
        window_layout.addWidget(QtWidgets.QLabel('Pre AP(ms):'))
        self.spin_window_pre = QtWidgets.QSpinBox()
        self.spin_window_pre.setRange(0, 500)
        self.spin_window_pre.setValue(10)
        window_layout.addWidget(self.spin_window_pre)

        window_layout.addWidget(QtWidgets.QLabel('Post AP(ms):'))
        self.spin_window_post = QtWidgets.QSpinBox()
        self.spin_window_post.setRange(0, 500)
        self.spin_window_post.setValue(20)
        window_layout.addWidget(self.spin_window_post)
        left_layout.addLayout(window_layout)

        # Batch processing button
        self.btn_batch = QtWidgets.QPushButton('Batch Process')
        left_layout.addWidget(self.btn_batch)

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
        self.btn_batch.clicked.connect(self.batch_process)
        self.btn_smooth.clicked.connect(self.apply_smooth)
        self.btn_align.clicked.connect(self.apply_zero_baseline)
        self.btn_detect.clicked.connect(self.apply_detect)
        self.btn_filter.clicked.connect(self.apply_filter)
        self.btn_ap_analysis.clicked.connect(self.apanalysis)
        self.chk_manual_spike.stateChanged.connect(self.update_spike_mode_link)
    

# Methods
    #load file
    def load_single_file(self, path):
        """Load a single .abf or .h5 file into signals, raw_signals, and file tree."""
        from copy import deepcopy
        import neo

        base_name = os.path.basename(path)
        ext = os.path.splitext(path)[1].lower()

        try:
            if ext == '.abf':
                reader = neo.io.AxonIO(filename=path)
                block = reader.read_block(lazy=False)
                reader = None
            elif ext == '.h5':
                reader = neo.io.NixIO(filename=path, mode='ro')
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
                if not seg.analogsignals:
                    continue

                signal = seg.analogsignals[0]

                raw_copy = deepcopy(signal)
                proc_copy = deepcopy(signal)

                display_name = self._generate_unique_name(f"{base_name}_sweep{i+1}")

                self.raw_signals[display_name] = raw_copy
                self.signals[display_name] = proc_copy
                self.path_map[display_name] = path
                self.loaded_paths.add(path)

                item = QtWidgets.QTreeWidgetItem([display_name])
                item.setData(0, QtCore.Qt.UserRole, display_name)
                self.file_tree.addTopLevelItem(item)

                # get spike
                if ext == '.h5':
                    self.load_spikes_from_h5(path, display_name)

                # load and plot
                if self.current_key is None:
                    self.current_key = display_name
                    self.proc_key = display_name
                    self.proc_signal = proc_copy
                    self.raw_signal = raw_copy
                    self.file_tree.setCurrentItem(item)
                    self.on_tree_item_clicked(item, 0)

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load Error", f"Error during parsing {path}\n{e}")

    #load method
    def load_spikes_from_h5(self, path, key):
        """Load saved spike info for a signal, only if available."""
        import h5py
        try:
            with h5py.File(path, 'r') as f:
                if 'spikes' not in f:
                    return 

                grp_spikes = f['spikes']

                has_auto = 'auto' in grp_spikes and 'times' in grp_spikes['auto'] and 'amplitudes' in grp_spikes['auto']
                has_manual = 'manual' in grp_spikes and 'times' in grp_spikes['manual'] and 'amplitudes' in grp_spikes['manual']

                if not (has_auto or has_manual):
                    return 

                # if with spikes, new
                self.spike_info[key] = {}

                if has_auto:
                    self.spike_info[key]['auto_times'] = grp_spikes['auto']['times'][:]
                    self.spike_info[key]['auto_amps'] = grp_spikes['auto']['amplitudes'][:]

                if has_manual:
                    self.spike_info[key]['manual_times'] = grp_spikes['manual']['times'][:].tolist()
                    self.spike_info[key]['manual_amps'] = grp_spikes['manual']['amplitudes'][:].tolist()

        except Exception as e:
            print(f"Warning: Failed to load spikes from {path}: {e}")

    def load_file(self):
        """Let user select multiple .abf or .h5 files for loading."""
        last_dir = self.settings.value('lastDir', QDir.homePath())

        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, 'Select File(s)', last_dir,
            'Signal Files (*.abf *.h5)'
        )
        if not files:
            return

        self.settings.setValue('lastDir', os.path.dirname(files[0]))

        new_files = [f for f in files if f not in self.loaded_paths]
        if not new_files:
            QtWidgets.QMessageBox.information(
                self, 'Already loaded', 'All selected files have already been loaded.')
            return

        for f in new_files:
            self.load_single_file(f)

        # plot first one
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
        """Save selected signals as separate HDF5 files with auto-naming."""
        selected_items = self.file_tree.selectedItems()
        if not selected_items:
            QtWidgets.QMessageBox.warning(self, "Warning", "No signals selected in the file tree.")
            return

        last_dir = self.settings.value('lastSaveDir', QDir.homePath())
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Output Folder", last_dir)
        if not out_dir:
            return

        self.settings.setValue('lastSaveDir', out_dir)
        errors = []

        for item in selected_items:
            key = item.data(0, QtCore.Qt.UserRole)
            signal = self.signals.get(key, None)
            if signal is None:
                errors.append(f"{key}: signal not found")
                continue

            # name
            base_name = key.replace('/', '_')
            out_path = os.path.join(out_dir, f"{base_name}.h5")

            try:
                self._save_signal_to_h5(signal, out_path, key)
            except Exception as e:
                errors.append(f"{base_name}.h5: {e}")

        if errors:
            QtWidgets.QMessageBox.warning(self, "Partial Save", "Some files failed to save:\n" + "\n".join(errors))
        else:
            QtWidgets.QMessageBox.information(self, "Success", "All selected signals saved successfully.")

    def _save_signal_to_h5(self, signal, out_path, key):
        # clear
        if os.path.exists(out_path):
            os.remove(out_path)

        blk = neo.Block(name=f"block_{key}")
        seg = neo.Segment(name=f"seg_{key}")
        signal.name = f"sig_{key}"
        seg.analogsignals.append(signal)
        blk.segments.append(seg)
        with NixIO(filename=out_path, mode='ow') as io:
            io.write_block(blk)

        info = self.spike_info.get(key, None)
        if info is None:
            return

        auto_times  = info.get('auto_times',  None)
        auto_amps   = info.get('auto_amps',   None)
        manual_times = info.get('manual_times', [])
        manual_amps  = info.get('manual_amps',  [])

        # 如果既没自动也没手动，就不创建 spikes 组
        has_auto   = (auto_times is not None and getattr(auto_times, 'size', len(auto_times)) > 0)
        has_manual = (len(manual_times) > 0)

        if not (has_auto or has_manual):
            return

        with h5py.File(out_path, 'a') as f:
            if 'spikes' in f:
                del f['spikes']
            grp_spikes = f.create_group('spikes')

            if has_auto:
                g_auto = grp_spikes.create_group('auto')
                g_auto.create_dataset('times',      data=auto_times)
                g_auto.create_dataset('amplitudes', data=auto_amps)

            if has_manual:
                g_man = grp_spikes.create_group('manual')
                g_man.create_dataset('times',      data=manual_times)
                g_man.create_dataset('amplitudes', data=manual_amps)
   
    #remove
    def remove_file(self):
        """Remove selected signal(s) from tree, memory, and path map."""
        selected_items = self.file_tree.selectedItems()
        for item in selected_items:
            key = item.data(0, QtCore.Qt.UserRole)
            if key in self.signals:
                del self.signals[key]

            if key in self.path_map:
                path = self.path_map.pop(key)
                self.loaded_paths.discard(path)

            index = self.file_tree.indexOfTopLevelItem(item)
            self.file_tree.takeTopLevelItem(index)

        # clear plot
        self.raw_plot.clear()
        self.proc_plot.clear()
        self.raw_signal = None
        self.proc_signal = None
        self.current_key = None
        self.proc_key = None
       
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
        for plot in [self.raw_plot, self.proc_plot]:
            if plot.listDataItems():
                plot.enableAutoRange(axis='xy')
                plot.plotItem.vb.setMouseEnabled(x=not self.chk_use_time.isChecked(), y=not self.chk_use_time.isChecked())

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
    #add new
    def _add_signal_to_tree(self, display_name, signal):
        self.signals[display_name] = signal
        item = QtWidgets.QTreeWidgetItem([display_name])
        item.setData(0, QtCore.Qt.UserRole, display_name)
        self.file_tree.addTopLevelItem(item)
    def save_proc_and_spike(self):
        """Save current processed signal and spike into tree view (internal save)."""
        if self.proc_signal is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "No processed signal to save.")
            return

        from copy import deepcopy

        new_signal = deepcopy(self.proc_signal)
        new_key = self._generate_unique_name(f"{self.proc_key}_saved")

        self.signals[new_key] = new_signal

        # 登记source_map！
        self.source_map[new_key] = self.proc_key

        if self.proc_key in self.spike_info:
            self.spike_info[new_key] = deepcopy(self.spike_info[self.proc_key])

        if self.proc_key in self.raw_signals:
            self.raw_signals[new_key] = deepcopy(self.raw_signals[self.proc_key])

        if self.proc_key in self.path_map:
            self.path_map[new_key] = self.path_map[self.proc_key]

        item = QtWidgets.QTreeWidgetItem([new_key])
        item.setData(0, QtCore.Qt.UserRole, new_key)
        self.file_tree.addTopLevelItem(item)

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

    def handle_manual_spike_click(self, event):
        """Handle left/right click for manual spike add/remove."""
        if event.button() not in (Qt.LeftButton, Qt.RightButton):
            return

        if self.proc_signal is None or self.hover_spike is None:
            return

        click_x, click_y = self.hover_spike

        info = self.get_spike_info(create=True)

        if event.button() == Qt.LeftButton:
            # manual spike
            info['manual_times'].append(click_x)
            info['manual_amps'].append(click_y)
            self.update_manual_spike_scatter()

        elif event.button() == Qt.RightButton:
            # delete spike (auto or manual)
            self.delete_nearest_spike(click_x, click_y)

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
        """When manual spike marking is enabled, link raw/proc x-axis. Otherwise unlink."""
        if state == Qt.Checked:
            self.proc_plot.setXLink(self.raw_plot)  
        else:
            self.proc_plot.setXLink(None)            

    def apply_detect(self):
        if self.proc_key is None or self.proc_key not in self.signals:
            QtWidgets.QMessageBox.warning(self, "Warning", "No processed signal found. Please run ALS or Zero Baseline first.")
            return

        signal = self.signals[self.proc_key]
        self.proc_signal = signal  # renew proc_signal

        y = signal.magnitude.flatten()
        fs = float(signal.sampling_rate.rescale('Hz').magnitude)

        wavelet_name = self.combo_wavelet.currentText()
        low_f = self.freq_low.value()
        high_f = self.freq_high.value()
        threshold_std = self.wavelet_thresh.value()

        try:
            spike_idx = wavelet_spike_detect(
                y, fs,
                freq_range=(low_f, high_f),
                threshold_std=threshold_std,
                wavelet_name=wavelet_name
            )

            if spike_idx is None or len(spike_idx) == 0:
                QtWidgets.QMessageBox.information(self, "No Spikes", "No spikes detected in the current settings.")
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
            # add spike
            info = self.get_spike_info(create=True)
            info['manual_times'].append(click_x)
            info['manual_amps'].append(click_y)
            self.update_manual_spike_scatter()

        elif event.button() == Qt.RightButton:
            # delete spike
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
        se_factor = self.relThresholdSpinBox.value()

        filtered_times, filtered_amps = cluster_spikes(
            auto_times, auto_amps,
            min_interval=min_interval,
            min_amplitude=min_amp,
            se_factor=se_factor
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
        ydata = self.proc_signal.magnitude.flatten()

        # locate
        pixel_scale_x = (self.proc_plot.plotItem.vb.viewRange()[0][1] - self.proc_plot.plotItem.vb.viewRange()[0][0]) / self.proc_plot.width()
        pixel_scale_y = (self.proc_plot.plotItem.vb.viewRange()[1][1] - self.proc_plot.plotItem.vb.viewRange()[1][0]) / self.proc_plot.height()

        # scale
        tol_x = 20 * pixel_scale_x
        tol_y = 20 * pixel_scale_y

        # get data point
        mask = (np.abs(t - x) <= tol_x) & (np.abs(ydata - y) <= tol_y)

        if not np.any(mask):
            self.hover_marker.setVisible(False)
            self.hover_spike = None
            return

        # get max
        candidate_idx = np.where(mask)[0]
        best_idx = candidate_idx[np.argmax(ydata[candidate_idx])]

        best_x = t[best_idx]
        best_y = ydata[best_idx]

        # refresh hover marker
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
        max_distance = 0.15  
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
        """Show or hide original saved raw signal."""
        self.raw_plot.clear()

        if self.chk_show_raw.isChecked() and self.current_key is not None:
            # find
            raw_signal = self.raw_signals.get(self.current_key)

            if raw_signal is None:
                # move suffix
                base_key = self.current_key.split("_saved")[0]
                raw_signal = self.raw_signals.get(base_key)

            if raw_signal is None:
                return

            t = (raw_signal.times - raw_signal.t_start).rescale('s').magnitude.flatten()
            y = raw_signal.magnitude[:, 0].flatten()

            self.raw_plot.plot(t, y, pen=pg.mkPen('k'))

    #AP
    def apanalysis(self):
        """Analyze and export spike features with corrected units and stable rise/decay time extraction, and save both relative and absolute spike times."""

        import pandas as pd
        from scipy.signal import find_peaks, peak_widths

        selected_items = self.file_tree.selectedItems()
        if not selected_items:
            QtWidgets.QMessageBox.warning(self, "Warning", "No signals selected for AP analysis.")
            return

        if self.last_folder:
            folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder to Save CSV Files", self.last_folder)
        else:
            folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder to Save CSV Files")
        if not folder:
            return

        self.last_folder = folder

        window_pre_ms = self.spin_window_pre.value()
        window_post_ms = self.spin_window_post.value()

        results_by_file = {}

        for item in selected_items:
            key = item.data(0, QtCore.Qt.UserRole)
            signal = self.signals.get(key, None)
            if signal is None:
                continue

            signal_y = signal.magnitude.flatten()
            fs = float(signal.sampling_rate.rescale('Hz'))
            dt = 1.0 / fs
            t_rel = (signal.times - signal.t_start).rescale('s').magnitude.flatten()
            t_abs = signal.times.rescale('s').magnitude.flatten()

            info = self.get_spike_info(key=key)
            if info is None:
                continue

            spike_times = []
            if info.get('manual_times'):
                spike_times.extend(info['manual_times'])
            if info.get('auto_times') is not None:
                spike_times.extend(info['auto_times'])

            if len(spike_times) == 0:
                continue

            spike_times = np.sort(np.array(spike_times))

            if 'zeroed' in key.lower():
                baseline = extract_baseline_value(signal_y, method='mode')
            else:
                baseline = 0.0

            features = {
                'Spike Relative Time (ms)': [],
                'Spike Absolute Time (ms)': [],
                'Width (ms)': [],
                'Amplitude (mV)': [],
                'Prominence (mV)': [],
                'Rise Time (ms)': [],
                'Decay Time (ms)': [],
                'ISI (ms)': []
            }

            last_spike_time_abs = None

            for spk_time in spike_times:
                idx_center = np.argmin(np.abs(t_rel - spk_time))

                idx_start = max(0, idx_center - int(window_pre_ms/1000/dt))
                idx_end = min(len(signal_y), idx_center + int(window_post_ms/1000/dt))

                ep_y = signal_y[idx_start:idx_end]
                ep_t = t_rel[idx_start:idx_end] - t_rel[idx_center]

                if len(ep_y) == 0:
                    continue

                peaks, properties = find_peaks(ep_y, prominence=0)
                if len(peaks) == 0:
                    continue

                peak_idx = peaks[np.argmax(ep_y[peaks])]
                amp = ep_y[peak_idx] + baseline
                prominence = properties['prominences'][np.argmax(ep_y[peaks])]
                width = peak_widths(ep_y, [peak_idx], rel_height=0.5)[0][0] * dt * 1000

                half_amp = ep_y[peak_idx] / 2.0

                left_part = ep_y[:peak_idx]
                rise_candidates = np.where(left_part <= half_amp)[0]
                if len(rise_candidates) > 0:
                    rise_time = (peak_idx - rise_candidates[-1]) * dt * 1000
                else:
                    rise_time = np.nan

                right_part = ep_y[peak_idx:]
                decay_candidates = np.where(right_part <= half_amp)[0]
                if len(decay_candidates) > 0:
                    decay_time = decay_candidates[0] * dt * 1000
                else:
                    decay_time = np.nan

                features['Spike Relative Time (ms)'].append(t_rel[idx_center] * 1000)
                features['Spike Absolute Time (ms)'].append(t_abs[idx_center] * 1000)
                features['Width (ms)'].append(width)
                features['Amplitude (mV)'].append(amp)
                features['Prominence (mV)'].append(prominence)
                features['Rise Time (ms)'].append(rise_time)
                features['Decay Time (ms)'].append(decay_time)

                if last_spike_time_abs is None:
                    features['ISI (ms)'].append(np.nan)
                else:
                    features['ISI (ms)'].append((t_abs[idx_center] - last_spike_time_abs) * 1000)

                last_spike_time_abs = t_abs[idx_center]

            if not features['Spike Relative Time (ms)']:
                continue

            base_name = key.split('_sweep')[0]

            if base_name not in results_by_file:
                results_by_file[base_name] = []

            results_by_file[base_name].append(features)

        for base_name, features_list in results_by_file.items():
            combined = {k: [] for k in features_list[0].keys()}

            for feat in features_list:
                for k, v in feat.items():
                    combined[k].extend(v)

            df = pd.DataFrame(combined).transpose()
            df.columns = [f"Spike{i+1}" for i in range(df.shape[1])]

            save_path = os.path.join(folder, f"{base_name}_apanalysis.csv")
            df.to_csv(save_path, header=True)

        QtWidgets.QMessageBox.information(self, "Export Done", "AP Analysis completed!")

    def batch_process(self):
        QtWidgets.QMessageBox.information(self, "Batch", "Batch processing not implemented yet.")


# ALS
def als_baseline(y, lam=1e4, p=1e-4, niter=10):
    """
    Asymmetric Least Squares baseline correction.
    y: input signal
    lam: smoothness parameter (lambda)
    p: asymmetry parameter
    niter: number of iterations
    """
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    w = np.ones(L)

    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * y)
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
    if wavelet_name == "cmor":
        wavelet_name = "cmor1.5-1.0"

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

    mean_val = np.mean(band_energy)
    std_val = np.std(band_energy)
    spike_idx = np.where(band_energy > mean_val + threshold_std * std_val)[0]
    return spike_idx

#cluster
def cluster_spikes(spike_times, spike_amps, min_interval=0.03, min_amplitude=0.0, se_factor=0.0):
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

        # Step 1: SE filter first
        if se_factor > 0 and len(group_amps) > 1:
            mean_amp = np.mean(group_amps)
            if len(group_amps) > 1:
                se = np.std(group_amps) / np.sqrt(len(group_amps))
                threshold = mean_amp + se_factor * se
            else:
                threshold = mean_amp
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