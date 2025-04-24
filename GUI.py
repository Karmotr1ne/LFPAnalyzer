import sys
import os
import neo
import pywt
import numpy as np
import pyqtgraph as pg
from neo.io import NixIO
from PyQt5 import QtWidgets,QtCore
from PyQt5.QtCore import QSettings, QDir, Qt
from scipy import sparse
from scipy.sparse.linalg import spsolve

class LFPAnalyzer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('LFP Data Analyzer')
        self.resize(1200, 800)
        self.signals = {}
        self.loaded_paths = set()
        self.path_map = {}  # display_name -> path
        self.current_key = None
        self.raw_signal = None
        self.proc_key = None
        self.proc_signal = None
        self.baseline_curve = []
        self._create_main_layout()
        self.raw_plot.addLegend()
        self.raw_plot.plotItem.legend.setVisible(False)
        self.combo_wavelet.setCurrentText('cmor')

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

        self.combo_baseline_mode = QtWidgets.QComboBox()
        self.combo_baseline_mode.addItems(['mode', 'mean', 'median'])
        baseline_layout.addWidget(QtWidgets.QLabel("Method:"))
        baseline_layout.addWidget(self.combo_baseline_mode)

        left_layout.addLayout(baseline_layout) 

        # Wavelet mother selection
        wavelet_layout = QtWidgets.QHBoxLayout()
        wavelet_layout.addWidget(QtWidgets.QLabel("Mother wavelet:"))

        self.combo_wavelet = QtWidgets.QComboBox()
        self.combo_wavelet.addItems(['morl', 'cmor', 'mexh', 'gaus1'])
        wavelet_layout.addWidget(self.combo_wavelet)

        left_layout.addLayout(wavelet_layout)

        #Spike Clustering
        self.mergeSpikesCheckBox = QtWidgets.QCheckBox("Merge nearby spikes")
        self.mergeSpikesCheckBox.setChecked(True)
        left_layout.addWidget(self.mergeSpikesCheckBox)

        # time interval
        intervalLayout = QtWidgets.QHBoxLayout()
        intervalLayout.addWidget(QtWidgets.QLabel("Min Interval (ms):"))
        self.minIntervalSpinBox = QtWidgets.QDoubleSpinBox()
        self.minIntervalSpinBox.setRange(1.0, 1000.0)
        self.minIntervalSpinBox.setSingleStep(1.0)
        self.minIntervalSpinBox.setValue(200.0)
        intervalLayout.addWidget(self.minIntervalSpinBox)
        left_layout.addLayout(intervalLayout)
        
        # relative threshold
        relLayout = QtWidgets.QHBoxLayout()
        relLayout.addWidget(QtWidgets.QLabel("Threshold (SE):"))
        self.relThresholdSpinBox = QtWidgets.QDoubleSpinBox()
        self.relThresholdSpinBox.setRange(0, 10)
        self.relThresholdSpinBox.setDecimals(0)
        self.relThresholdSpinBox.setSingleStep(1)
        self.relThresholdSpinBox.setValue(0)
        relLayout.addWidget(self.relThresholdSpinBox)
        left_layout.addLayout(relLayout)

        # mini amptitude
        ampLayout = QtWidgets.QHBoxLayout()
        ampLayout.addWidget(QtWidgets.QLabel("Min Amp:"))
        self.minAmplitudeSpinBox = QtWidgets.QDoubleSpinBox()
        self.minAmplitudeSpinBox.setRange(0.0, 1000.0)
        self.minAmplitudeSpinBox.setValue(0.05)
        self.minAmplitudeSpinBox.setSingleStep(0.01)
        ampLayout.addWidget(self.minAmplitudeSpinBox)
        left_layout.addLayout(ampLayout)

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
        self.raw_plot.mousePressEvent = lambda e: self.start_time_select(e, self.raw_plot)
        self.raw_plot.mouseReleaseEvent = lambda e: self.end_time_select(e, self.raw_plot)
        right_layout.addWidget(self.raw_plot)

        # Processed signal plot
        self.proc_plot = pg.PlotWidget(title='Processed Signal')
        self.proc_plot.setBackground('w') 
        #sync
        self.proc_selector_fill = pg.LinearRegionItem([0, 0], brush=(0, 100, 255, 50))
        self.proc_selector_fill.setZValue(10)
        self.proc_selector_fill.setMovable(False)
        self.proc_selector_fill.setVisible(False)
        self.proc_plot.addItem(self.proc_selector_fill)
        #mouse selection
        self.proc_plot.mousePressEvent = lambda e: self.start_time_select(e, self.proc_plot)
        self.proc_plot.mouseReleaseEvent = lambda e: self.end_time_select(e, self.proc_plot) 
        right_layout.addWidget(self.proc_plot)

        #draw selected area
        self.raw_plot.scene().sigMouseMoved.connect(self.on_mouse_drag_move)
        self.proc_plot.scene().sigMouseMoved.connect(self.on_mouse_drag_move)


        # Operation buttons below processed plot
        op_layout = QtWidgets.QHBoxLayout()
        self.btn_smooth = QtWidgets.QPushButton('ALS Detrend')
        self.btn_align = QtWidgets.QPushButton("Zero Baseline")
        self.btn_detect = QtWidgets.QPushButton('Spike Detect')
        op_layout.addWidget(self.btn_smooth)
        op_layout.addWidget(self.btn_align)
        op_layout.addWidget(self.btn_detect)

        right_layout.addLayout(op_layout)
        splitter.addWidget(right_panel)

        # Connect signals to slots (to be implemented)
        self.btn_add.clicked.connect(self.load_file)
        self.btn_remove.clicked.connect(self.remove_file)
        self.btn_save.clicked.connect(self.save)
        self.btn_apply.clicked.connect(self.apply_downsample)
        self.btn_batch.clicked.connect(self.batch_process)
        self.btn_smooth.clicked.connect(self.apply_smooth)
        self.btn_align.clicked.connect(self.apply_zero_baseline)
        self.btn_detect.clicked.connect(self.apply_detect)

# Methods
    #load file
    def load_single_file(self, path):
        """Load a single ABF or HDF5 file into the UI and signal dict."""
        try:
            ext = os.path.splitext(path)[1].lower()
            if ext == ".abf":
                reader = neo.io.AxonIO(filename=path)
            elif ext == ".h5":
                reader = neo.io.NixIO(filename=path, mode='ro')
            else:
                QtWidgets.QMessageBox.warning(
                    self, "Unsupported Format", f"Unsupported file type:\n{path}"
                )
                return

            block = reader.read_block(lazy=False)
            base_name = os.path.splitext(os.path.basename(path))[0]

            for i, seg in enumerate(block.segments):
                if not seg.analogsignals:
                    continue  # avoid blank segment
                signal = seg.analogsignals[0]

                display_name = self._generate_unique_name(f"{base_name}_sweep{i+1}")
                self.signals[display_name] = signal
                self.path_map[display_name] = path
                self.loaded_paths.add(path)

                item = QtWidgets.QTreeWidgetItem([display_name])
                item.setData(0, QtCore.Qt.UserRole, display_name)
                self.file_tree.addTopLevelItem(item)

                # first as current key
                if self.current_key is None:
                    self.current_key = display_name
                    self.raw_signal = signal
                    self.clear_raw_plot()
                    self._plot_signal(signal, self.raw_plot, color='b')

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error", f"Failed to load:\n{os.path.basename(path)}\n\n{e}")
    
    #load method
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

    #save file
    def save(self):
        """Save selected signals as separate HDF5 files with auto-naming."""
        selected_items = self.file_tree.selectedItems()
        if not selected_items:
            QtWidgets.QMessageBox.warning(self, "Warning", "No signals selected in the file tree.")
            return

        # Select output folder
        last_dir = self.settings.value('lastSaveDir', QDir.homePath())
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(
             self, "Select Output Folder", last_dir
        )
        if not out_dir:
            return
        
        self.settings.setValue('lastSaveDir', out_dir)

        errors = []

        for item in selected_items:
            key = item.data(0, QtCore.Qt.UserRole)
            if key not in self.signals:
                errors.append(str(key))
                continue

            signal = self.signals[key]

            # Auto-generated name
            base_name = os.path.splitext(str(key))[0]
            default_name = f"{base_name}.h5"
            out_path = os.path.join(out_dir, default_name)

            # Save individual signal to its own HDF5
            try:
                io = NixIO(filename=out_path, mode='ow')
                blk = neo.Block()
                seg = neo.Segment()
                seg.analogsignals.append(signal)
                blk.segments.append(seg)
                io.write_block(blk)
                io.close()
            except Exception as e:
                errors.append(f"{default_name}: {str(e)}")

        # Final report
        if errors:
            QtWidgets.QMessageBox.warning(
                self, "Partial Save",
                f"Some files failed to save:\n" + "\n".join(errors))
        else:
            QtWidgets.QMessageBox.information(self, "Success", "All selected signals saved successfully.")

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

        # units
        if hasattr(signal, "units") and signal.units is not None:
            plot.setLabel('left', f'Amplitude ({signal.units})')

        items = plot.listDataItems()
        return items[-1] if items else None
    
    #plot patch
    def finalize_processing(self, proc_key, proc_signal, source_key):
        """renew outcome to proc_plot, origin to raw_plot"""
        self.proc_key = proc_key
        self.proc_signal = proc_signal
        self.current_key = source_key
        self.raw_signal = self.signals.get(source_key)

        # plot
        self.clear_raw_plot()
        self.clear_proc_plot()

        if self.raw_signal is not None:
            self._plot_signal(self.raw_signal, self.raw_plot, color='b')

        t = self.get_relative_time(proc_signal, proc_signal.times.rescale('s').magnitude, mode='time')
        y = proc_signal.magnitude.flatten()
        self.proc_plot.plot(t, y, pen=pg.mkPen('m'))

        # file_tree select
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
    def on_tree_item_clicked(self, *_):
        selected = self.file_tree.selectedItems()
        if not selected:
            return

        top_item = selected[0]
        display_name = top_item.data(0, QtCore.Qt.UserRole)
        if display_name not in self.signals:
            return

        signal = self.signals[display_name]
        self.current_key = display_name
        self.raw_signal = signal

        # renew raw
        self.clear_raw_plot()
        self._plot_signal(signal, self.raw_plot, color='b')

        # maintain proc
        self.raw_plot.plotItem.vb.enableAutoRange(axis='xy')

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
        self.clear_plot_with_selector('proc_plot', 'proc_selector_fill')
    
    #overlay
    def overlay_selected_signals(self):
        if self.baseline_curve is not None:
            self.raw_plot.removeItem(self.baseline_curve)
            self.baseline_curve = []
            
        selected_items = self.file_tree.selectedItems()
        if not selected_items:
            QtWidgets.QMessageBox.warning(self, "Warning", "No signals selected.")
            return

        self.raw_plot.clear()
        legend = self.raw_plot.plotItem.legend
        colors = ['b', 'r', 'm', 'c', 'y', 'k']
        color_cycle = iter(colors * 10)

        self.overlay_curves = {}

        for item in selected_items:
            key = item.data(0, QtCore.Qt.UserRole)
            if key not in self.signals:
                continue
            signal = self.signals[key]
            t = signal.times.rescale('s').magnitude.flatten()
            y = signal.magnitude.flatten()
            pen = pg.mkPen(color=next(color_cycle), width=1)
            curve = self.raw_plot.plot(t, y, pen=pen, name=key)
            self.overlay_curves[key] = curve

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
    #size reduce
    def apply_downsample(self):
        selected_items = self.file_tree.selectedItems()
        if not selected_items:
            QtWidgets.QMessageBox.warning(self, "Warning", "No signals selected for downsampling.")
            return

        factor = self.spin_down.value()
        from neo.core import AnalogSignal
        import quantities as pq

        first_key = None
        first_signal = None

        for item in selected_items:
            key = item.data(0, QtCore.Qt.UserRole)
            if key not in self.signals:
                continue

            orig = self.signals[key]
            try:
                new_signal = AnalogSignal(
                    orig.magnitude[::factor],
                    units=orig.units,
                    sampling_rate=orig.sampling_rate / factor,
                    t_start=orig.t_start,
                    name=f"{orig.name}_down{factor}x" if orig.name else None
                )
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Error", f"{key} downsample failed:\n{e}")
                continue

            display_name = self._generate_unique_name(f"{key}_down{factor}x")
            self.signals[display_name] = new_signal

            item_new = QtWidgets.QTreeWidgetItem([display_name])
            item_new.setData(0, QtCore.Qt.UserRole, display_name)
            self.file_tree.addTopLevelItem(item_new)

            if first_key is None:
                first_key = display_name
                first_signal = new_signal

        if first_key:
            self.finalize_processing(first_key, first_signal, source_key=first_key.rsplit("_down", 1)[0])

    
    #smooth
    def apply_smooth(self):
        from neo.core import AnalogSignal
        import quantities as pq

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

            raw_signal = self.signals[key]
            y = raw_signal.magnitude.flatten()

            try:
                baseline = als_baseline(y, lam=lam, p=p)
                detrended = y - baseline

                smoothed_signal = AnalogSignal(
                    detrended.reshape(-1, 1),
                    units=raw_signal.units,
                    sampling_rate=raw_signal.sampling_rate,
                    t_start=raw_signal.t_start,
                    name=f"{raw_signal.name}_als" if raw_signal.name else None
                )

                display_name = self._generate_unique_name(f"{key}_als")
                self.signals[display_name] = smoothed_signal

                item_new = QtWidgets.QTreeWidgetItem([display_name])
                item_new.setData(0, QtCore.Qt.UserRole, display_name)
                self.file_tree.addTopLevelItem(item_new)

                if first_key is None:
                    first_key = display_name
                    first_signal = smoothed_signal

            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "ALS Error", f"{key} failed:\n{e}")

        if first_key:
            self.finalize_processing(first_key, first_signal, source_key=first_key.rsplit("_als", 1)[0])

    #remove baseline
    def apply_zero_baseline(self):
        from neo.core import AnalogSignal

        selected_items = self.file_tree.selectedItems()
        if not selected_items:
            QtWidgets.QMessageBox.warning(self, "Warning", "No signals selected.")
            return

        method = self.combo_baseline_mode.currentText()
        first_key = None
        first_signal = None

        for item in selected_items:
            key = item.data(0, QtCore.Qt.UserRole)
            if key not in self.signals or "als" not in key:
                QtWidgets.QMessageBox.warning(self, "Warning", f"{key} is not an ALS-processed signal.")
                continue

            signal = self.signals[key]
            y = signal.magnitude.flatten()

            self.current_key = key  # get for extract_data_from_time_range
            sample_values = self.extract_data_from_time_range()
            if sample_values is None or len(sample_values) == 0:
                sample_values = y

            baseline_val = extract_baseline_value(sample_values, method=method)
            centered = y - baseline_val
            centered = np.clip(centered, 0, None)

            aligned_signal = AnalogSignal(
                centered.reshape(-1, 1),
                units=signal.units,
                sampling_rate=signal.sampling_rate,
                t_start=signal.t_start,
                name=f"{signal.name}_zeroed" if signal.name else None
            )

            display_name = self._generate_unique_name(f"{key}_zeroed")
            self.signals[display_name] = aligned_signal

            item_new = QtWidgets.QTreeWidgetItem([display_name])
            item_new.setData(0, QtCore.Qt.UserRole, display_name)
            self.file_tree.addTopLevelItem(item_new)

            if first_key is None:
                first_key = display_name
                first_signal = aligned_signal

        if first_key:
            self.finalize_processing(first_key, first_signal, source_key=first_key.rsplit("_zeroed", 1)[0])

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
        if self._drag_start is None or not self.chk_use_time.isChecked():
            return

        for plot in [self.raw_plot, self.proc_plot]:
            vb = plot.plotItem.vb
            if vb.sceneBoundingRect().contains(pos):
                mouse_point = vb.mapToView(pos)
                x = mouse_point.x()
                x0 = self._drag_start
                x1 = x
                region = [min(x0, x1), max(x0, x1)]

                # selector exist
                self.time_selector = self.ensure_selector("time_selector", self.raw_plot)
                self.proc_selector_fill = self.ensure_selector("proc_selector_fill", self.proc_plot)

                self.update_selector_region(region)
                break

    def sync_xrange(self, source_plot):
        """sync x axis"""
        x_min, x_max = source_plot.viewRange()[0]
        target_plot = self.proc_plot if source_plot == self.raw_plot else self.raw_plot
        target_plot.setXRange(x_min, x_max, padding=0)

    #spike detect
    def apply_detect(self):
        if self.proc_key is None or self.proc_key not in self.signals:
            QtWidgets.QMessageBox.warning(self, "Warning", "No processed signal found. Please run ALS or Zero Baseline first.")
            return

        signal = self.signals[self.proc_key]
        self.proc_signal = signal

        y = signal.magnitude.flatten()
        fs = float(signal.sampling_rate.rescale('Hz').magnitude)

        wavelet_name = self.combo_wavelet.currentText()
        spike_idx = wavelet_spike_detect(y, fs, wavelet_name=wavelet_name)
        spike_amps = y[spike_idx]
        spike_times = self.get_relative_time(signal, spike_idx, mode='index')

        if self.mergeSpikesCheckBox.isChecked():
            min_interval = self.minIntervalSpinBox.value() / 1000.0
            min_amp = self.minAmplitudeSpinBox.value()
            se_factor = self.relThresholdSpinBox.value()

            spike_times, spike_amps = cluster_spikes(
                spike_times, spike_amps,
                min_interval=min_interval,
                min_amplitude=min_amp,
                se_factor=se_factor
            )

        self.spike_times = spike_times
        self.spike_amps = spike_amps
        self.plot_detected_spikes()

    def plot_detected_spikes(self):
        if self.proc_signal is None or self.spike_times is None:
            return

        self.clear_proc_plot()

        signal = self.proc_signal
        t = self.get_relative_time(signal, signal.times.rescale('s').magnitude, mode='time')
        y = signal.magnitude.flatten()

        self.proc_plot.plot(t, y, pen=pg.mkPen('b'))
        self.proc_plot.plot(self.spike_times, self.spike_amps, pen=None,
                            symbol='o', symbolBrush='r', symbolSize=6, name="Spikes")

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
    scales = np.arange(1, 128)
    coeffs, freqs = pywt.cwt(signal, scales, wavelet_name, sampling_period=1/fs)
    band_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
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
            std_amp = np.std(group_amps)
            threshold = mean_amp + se_factor * std_amp
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

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = LFPAnalyzer()
    win.show()
    sys.exit(app.exec_())