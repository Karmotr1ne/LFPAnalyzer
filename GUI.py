import sys
import os
import neo
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
        self.proc_signal = None
        self.baseline_curve = []
        self._create_main_layout()

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
        self.chk_use_time = QtWidgets.QCheckBox("Select baseline range")
        self.chk_use_time.setChecked(False)
        self.chk_use_time.stateChanged.connect(self.toggle_time_selector)

        self.combo_baseline_mode = QtWidgets.QComboBox()
        self.combo_baseline_mode.addItems(['mode', 'mean', 'median'])

        # def layout
        baseline_layout = QtWidgets.QHBoxLayout()

        self.chk_use_time = QtWidgets.QCheckBox("Select baseline range")
        self.chk_use_time.setChecked(False)
        self.chk_use_time.stateChanged.connect(self.toggle_time_selector)
        baseline_layout.addWidget(self.chk_use_time)

        self.combo_baseline_mode = QtWidgets.QComboBox()
        self.combo_baseline_mode.addItems(['mode', 'mean', 'median'])
        baseline_layout.addWidget(QtWidgets.QLabel("Method:"))
        baseline_layout.addWidget(self.combo_baseline_mode)

        left_layout.addLayout(baseline_layout) 

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
            signal = block.segments[0].analogsignals[0]

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error", f"Failed to load:\n{os.path.basename(path)}\n\n{e}")
            return

        #load name
        base_name = os.path.splitext(os.path.basename(path))[0]
        display_name = self._generate_unique_name(base_name)

        self.signals[display_name] = signal
        self.path_map[display_name] = path
        self.loaded_paths.add(path)

        item = QtWidgets.QTreeWidgetItem([display_name])
        item.setData(0, QtCore.Qt.UserRole, display_name)
        self.file_tree.addTopLevelItem(item)

        # If first file, set as current and plot
        if self.current_key is None:
            self.current_key = display_name
            self.raw_signal = signal
            self.clear_raw_plot()
            self._plot_signal(signal, self.raw_plot, color='b')

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
       
    #plot signal
    def _plot_signal(self, signal, plot, color='b', width=1, name=None, return_item=False):

        t = signal.times.rescale('s').magnitude.flatten()
        y = signal.magnitude.flatten()

        pen = pg.mkPen(color=color, width=width)

        # plot
        plot.plot(t, y, pen=pen, name=name)

        if return_item:
            return plot.listDataItems()[-1]

        # add proc_selector_fill to proc_plot
        if plot == self.proc_plot:
            if hasattr(self, "proc_selector_fill") and self.proc_selector_fill is not None:
                if self.proc_selector_fill not in self.proc_plot.items():
                    self.proc_plot.addItem(self.proc_selector_fill, ignoreBounds=True)

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

        self.current_key = display_name
        self.raw_signal = self.signals[display_name]
        self.proc_signal = None

        self.clear_raw_plot()
        self._plot_signal(self.raw_signal, self.raw_plot, color='b')
        self.clear_proc_plot()

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
        legend = self.raw_plot.addLegend(offset=(10, 10))
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

        # last = current key
        last_key = selected_items[-1].data(0, QtCore.Qt.UserRole)
        if last_key in self.signals:
            self.current_key = last_key
            self.raw_signal = self.signals[last_key]
            self.proc_signal = None

        # legend toggle
        for sample in legend.items:
            _, label = sample  # (itemSample, labelItem)
            text = label.text

            if text not in self.overlay_curves:
                continue

            curve = self.overlay_curves[text]

            def make_toggle_handler(curve_ref=curve, label_ref=label):
                def toggle(event):
                    vis = not curve_ref.isVisible()
                    curve_ref.setVisible(vis)
                return toggle

            label.mousePressEvent = make_toggle_handler()

    #Utility         
    #size reduce
    def apply_downsample(self):
        if self.raw_signal is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "No raw signal to downsample.")
            return

        factor = self.spin_down.value()
        try:
            from neo.core import AnalogSignal
            import quantities as pq

            orig = self.raw_signal
            new_signal = AnalogSignal(
                orig.magnitude[::factor],
                units=orig.units,
                sampling_rate=orig.sampling_rate / factor,
                t_start=orig.t_start,
                name=f"{orig.name}_down{factor}x" if orig.name else None
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Downsampling failed:\n{e}")
            return

        # display name save
        base_display_name = f"{self.current_key}_down{factor}x"
        display_name = self._generate_unique_name(base_display_name)

        self.signals[display_name] = new_signal
        self.proc_signal = new_signal

        item = QtWidgets.QTreeWidgetItem([display_name])
        item.setData(0, QtCore.Qt.UserRole, display_name)
        self.file_tree.addTopLevelItem(item)
        self.file_tree.setCurrentItem(item)

        # plot
        self.proc_plot.clear()
        self._plot_signal(new_signal, self.proc_plot, color='r')
    
    #smooth
    def apply_smooth(self):
        for curve in self.baseline_curve:
            self.raw_plot.removeItem(curve)
        self.baseline_curve.clear()
        selected_items = self.file_tree.selectedItems()
        if not selected_items:
            QtWidgets.QMessageBox.warning(self, "Warning", "No signals selected.")
            return

        from neo.core import AnalogSignal
        import quantities as pq

        # clear process
        self.clear_proc_plot()

        # clear old baseline
        if self.baseline_curve:
            self.raw_plot.removeItem(self.baseline_curve)
            self.baseline_curve = []

        # get ALS parameters
        sigma_lambda = self.spin_sigma_lambda.value()
        sigma_p = self.spin_sigma_p.value()
        lam = self.base_lambda * (10 ** sigma_lambda)
        p = self.base_p * (10 ** sigma_p)

        self.baseline_curve = []  # multiple baseline

        for item in selected_items:
            key = item.data(0, QtCore.Qt.UserRole)
            if key not in self.signals:
                continue

            raw_signal = self.signals[key]
            y = raw_signal.magnitude.flatten()

            try:
                baseline = als_baseline(y, lam=lam, p=p)
                detrended = y - baseline

                # detect baseline
                method = self.combo_baseline_mode.currentText()

                # extract based on time selector (or fallback to detrended)
                sample_values = self.extract_data_from_time_range()
                if sample_values is None or len(sample_values) == 0:
                    sample_values = detrended

                # remove baseline
                baseline_val = extract_baseline_value(sample_values, method=method)
                detrended = detrended - baseline_val

                # new AnalogSignal
                smoothed_signal = AnalogSignal(
                    detrended.reshape(-1, 1),
                    units=raw_signal.units,
                    sampling_rate=raw_signal.sampling_rate,
                    t_start=raw_signal.t_start,
                    name=f"{raw_signal.name}_als" if raw_signal.name else None
                )

                # baseline as signal
                baseline_signal = AnalogSignal(
                    baseline.reshape(-1, 1),
                    units=raw_signal.units,
                    sampling_rate=raw_signal.sampling_rate,
                    t_start=raw_signal.t_start,
                    name=f"{raw_signal.name}_baseline" if raw_signal.name else None
                )

                # add to tree
                display_name = self._generate_unique_name(f"{key}_als")
                self.signals[display_name] = smoothed_signal
                new_item = QtWidgets.QTreeWidgetItem([display_name])
                new_item.setData(0, QtCore.Qt.UserRole, display_name)
                self.file_tree.addTopLevelItem(new_item)

                # show and keep raw_plot
                self.current_key = display_name
                self.proc_signal = smoothed_signal
                self.file_tree.setCurrentItem(new_item)

                self.clear_proc_plot()
                self._plot_signal(smoothed_signal, self.proc_plot, color='g', width=1, name=display_name)

                curve = self._plot_signal(baseline_signal, self.raw_plot, color='orange', width=2, name='Baseline', return_item=True)

                self.baseline_curve.append(curve)


            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "ALS Error", f"{key} failed:\n{e}")
                continue
    
    #remove baseline
    def apply_zero_baseline(self):
        from neo.core import AnalogSignal

        if self.current_key is None or self.current_key not in self.signals:
            QtWidgets.QMessageBox.warning(self, "Warning", "No signal selected.")
            return

        if "als" not in self.current_key:
            QtWidgets.QMessageBox.warning(self, "Invalid Source", "Please select an ALS-processed signal (name contains 'als').")
            return

        signal = self.signals[self.current_key]
        y = signal.magnitude.flatten()
        method = self.combo_baseline_mode.currentText()

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

        display_name = self._generate_unique_name(f"{self.current_key}_zeroed")
        self.signals[display_name] = aligned_signal
        self.proc_signal = aligned_signal

        new_item = QtWidgets.QTreeWidgetItem([display_name])
        new_item.setData(0, QtCore.Qt.UserRole, display_name)
        self.file_tree.addTopLevelItem(new_item)
        self.file_tree.setCurrentItem(new_item)

        self.current_key = display_name

        self.clear_proc_plot()
        self._plot_signal(aligned_signal, self.proc_plot, color='m', width=1, name=display_name)

    #baseline time selector
    def toggle_time_selector(self, state):
        use_time = state == Qt.Checked

        if hasattr(self, "time_selector") and self.time_selector is not None:
            self.time_selector.setVisible(use_time)

        if hasattr(self, "proc_selector_fill") and self.proc_selector_fill is not None:
            self.proc_selector_fill.setVisible(use_time)

        if self.raw_plot.plotItem.legend is not None:
            self.raw_plot.plotItem.legend.setVisible(not use_time)
        if self.proc_plot.plotItem.legend is not None:
            self.proc_plot.plotItem.legend.setVisible(not use_time)

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

        mouse_point = plot.plotItem.vb.mapToView(event.pos())
        self._drag_start = mouse_point.x()

        region = [self._drag_start, self._drag_start + 0.01]

        if not hasattr(self, "time_selector") or self.time_selector is None:
            self.time_selector = pg.LinearRegionItem(
                region, brush=(0, 100, 255, 100)
            )
            self.time_selector.setMovable(False)
            self.time_selector.setZValue(10)
            self.raw_plot.addItem(self.time_selector)
        else:
            self.time_selector.setRegion(region)

        self.time_selector.setVisible(True)

        # sync proc_plot
        if hasattr(self, "proc_selector_fill"):
            self.proc_selector_fill.setRegion(region)
            self.proc_selector_fill.setVisible(True)
    
    def end_time_select(self, event, plot):
        if event.button() != Qt.LeftButton or self._drag_start is None:
            return

        mouse_point = plot.plotItem.vb.mapToView(event.pos())
        x0 = self._drag_start
        x1 = mouse_point.x()

        if abs(x1 - x0) < 1e-3:
            x1 = x0 + 0.01

        region = [min(x0, x1), max(x0, x1)]

        if not hasattr(self, "time_selector") or self.time_selector is None:
            self.time_selector = pg.LinearRegionItem(
                region, brush=(0, 100, 255, 100)
            )
            self.time_selector.setMovable(False)
            self.time_selector.setZValue(10)
            self.time_selector.setPen(pg.mkPen(color='b', width=2))
            self.raw_plot.addItem(self.time_selector)
        else:
            self.time_selector.setRegion(region)

        self._drag_start = None
        self.sync_xrange(plot)

        # sync proc_plot
        if hasattr(self, "proc_selector_fill"):
            self.proc_selector_fill.setRegion(region)
            self.proc_selector_fill.setVisible(True)

    def on_mouse_drag_move(self, pos):
        if self._drag_start is None:
            return

        for plot in [self.raw_plot, self.proc_plot]:
            vb = plot.plotItem.vb
            if vb.sceneBoundingRect().contains(pos):
                mouse_point = vb.mapToView(pos)
                x = mouse_point.x()
                x0 = self._drag_start
                x1 = x
                region = [min(x0, x1), max(x0, x1)]

                if hasattr(self, "time_selector") and self.time_selector is not None:
                    self.time_selector.setRegion(region)

                # sync proc_plot
                if hasattr(self, "proc_selector_fill"):
                    self.proc_selector_fill.setRegion(region)
                    self.proc_selector_fill.setVisible(True)
                break

    def sync_xrange(self, source_plot):
        """sync x axis"""
        x_min, x_max = source_plot.viewRange()[0]
        target_plot = self.proc_plot if source_plot == self.raw_plot else self.raw_plot
        target_plot.setXRange(x_min, x_max, padding=0)

    def apply_detect(self):
        """Detect spikes on processed signal."""
        pass

    def batch_process(self):
        """Execute pipeline on all files in tree."""
        pass

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

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = LFPAnalyzer()
    win.show()
    sys.exit(app.exec_())