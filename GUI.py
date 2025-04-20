import sys
import os
import neo
import numpy as np
import pyqtgraph as pg
from neo.io import NixIO
from PyQt5 import QtWidgets, QtCore
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
        self._create_main_layout()

        #Memory location and Load more data
        self.settings = QSettings('FileLocation', 'LFPAnalyzer')        
        self.file_tree.itemClicked.connect(self.on_tree_item_clicked)

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
        als_layout.addWidget(QtWidgets.QLabel('λ (10^σ):'))

        self.spin_sigma_lambda = QtWidgets.QSpinBox()
        self.spin_sigma_lambda.setRange(-3, 5)  # range
        self.spin_sigma_lambda.setValue(0)
        als_layout.addWidget(self.spin_sigma_lambda)

        als_layout.addWidget(QtWidgets.QLabel('p (10^σ):'))
        self.spin_sigma_p = QtWidgets.QSpinBox()
        self.spin_sigma_p.setRange(-3, 3)
        self.spin_sigma_p.setValue(0)
        als_layout.addWidget(self.spin_sigma_p)

        left_layout.addLayout(als_layout)

        # Batch processing button
        self.btn_batch = QtWidgets.QPushButton('Batch Process')
        left_layout.addWidget(self.btn_batch)

        # Spacer
        left_layout.addStretch()

        splitter.addWidget(left_panel)

        # Right panel: Plots and controls
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)

        # Raw signal plot
        self.raw_plot = pg.PlotWidget(title='Raw Signal')
        self.raw_plot.setBackground('w')  
        right_layout.addWidget(self.raw_plot)

        # Processed signal plot
        self.proc_plot = pg.PlotWidget(title='Processed Signal')
        self.proc_plot.setBackground('w')  
        right_layout.addWidget(self.proc_plot)

        # Operation buttons below processed plot
        op_layout = QtWidgets.QHBoxLayout()
        self.btn_smooth = QtWidgets.QPushButton('ALS Detrend')
        self.btn_detect = QtWidgets.QPushButton('Spike Detect')
        op_layout.addWidget(self.btn_smooth)
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
            self.raw_plot.clear()
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
    def _plot_signal(self, signal, plot_widget, color='b', width=1, name=None):
        """
        Plot a signal on the given plot_widget with optional color, width and legend name.
        """
        if signal is None:
            return
        t = signal.times.rescale('s').magnitude.flatten()
        y = signal.magnitude.flatten()
        pen = pg.mkPen(color=color, width=width)
        if name:
            plot_widget.plot(t, y, pen=pen, name=name)
        else:
            plot_widget.plot(t, y, pen=pen)

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
        """Enable selection of multiple signals, draw top selected."""
        selected = self.file_tree.selectedItems()
        if not selected:
            return

        # show top
        top_item = selected[0]
        display_name = top_item.data(0, QtCore.Qt.UserRole)
        if display_name not in self.signals:
            return

        # refresh
        self.current_key = display_name
        self.raw_signal = self.signals[display_name]
        self.proc_signal = None

        # redraw raw_plot
        self.raw_plot.clear()
        self._plot_signal(self.raw_signal, self.raw_plot, color='b')

        # clear processe_plot
        self.proc_plot.clear()
    
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
    #right click open menu
    #clear&reload
    def clear_and_reload(self):
        self.file_tree.clear()
        self.signals.clear()
        self.loaded_paths.clear()
        self.current_key = None
        self.raw_signal = None
        self.proc_signal = None
        self.raw_plot.clear()
        self.proc_plot.clear()
    #overlay
    def overlay_selected_signals(self):
        selected_items = self.file_tree.selectedItems()
        if not selected_items:
            QtWidgets.QMessageBox.warning(self, "Warning", "No signals selected.")
            return

        self.raw_plot.clear()

        legend = self.raw_plot.addLegend(offset=(10, 10))
        colors = ['b', 'r', 'm', 'c', 'y', 'k']
        color_cycle = iter(colors * 10)

        self.overlay_curves = {}  # save legend -> plot item

        for item in selected_items:
            key = item.data(0, QtCore.Qt.UserRole)
            if key not in self.signals:
                continue
            signal = self.signals[key]
            color = next(color_cycle)
            t = signal.times.rescale('s').magnitude.flatten()
            y = signal.magnitude.flatten()
            pen = pg.mkPen(color=color, width=1)
            plot_item = self.raw_plot.plot(t, y, pen=pen, name=key)

            # save key to plot item
            self.overlay_curves[key] = plot_item

        # set last as signal
        last_key = selected_items[-1].data(0, QtCore.Qt.UserRole)
        if last_key in self.signals:
            self.current_key = last_key
            self.raw_signal = self.signals[last_key]
            self.proc_signal = None

        # click legend hide/show
        for sample in legend.items:
            label = sample[1]
            key = label.text

            # keep key leagel
            if key not in self.overlay_curves:
                continue

            curve = self.overlay_curves[key]

            def toggle(curve=curve):
                curve.setVisible(not curve.isVisible())

            # connect to click event
            label.mousePressEvent = lambda ev, curve=curve: toggle(curve)
            
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

    def batch_process(self):
        """Execute pipeline on all files in tree."""
        pass

    #smooth
    def apply_smooth(self):
        if self.raw_signal is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "No raw signal to smooth.")
            return

        try:
            from neo.core import AnalogSignal
            import quantities as pq

            y = self.raw_signal.magnitude.flatten()

            #cal lam and p
            sigma_lambda = self.spin_sigma_lambda.value()
            sigma_p = self.spin_sigma_p.value()

            lam = self.base_lambda * (10 ** sigma_lambda)
            p = self.base_p * (10 ** sigma_p)

            #als using lam and p
            baseline = als_baseline(y, lam=lam, p=p)
            detrended = y - baseline

            smoothed_signal = AnalogSignal(
                detrended.reshape(-1, 1),
                units=self.raw_signal.units,
                sampling_rate=self.raw_signal.sampling_rate,
                t_start=self.raw_signal.t_start,
                name=f"{self.raw_signal.name}_als" if self.raw_signal.name else None
            )

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"ALS failed:\n{e}")
            return

        # insert signal dict and tree
        base_name = f"{self.current_key}_als"
        display_name = self._generate_unique_name(base_name)
        self.signals[display_name] = smoothed_signal
        self.proc_signal = smoothed_signal

        item = QtWidgets.QTreeWidgetItem([display_name])
        item.setData(0, QtCore.Qt.UserRole, display_name)
        self.file_tree.addTopLevelItem(item)
        self.file_tree.setCurrentItem(item)

        self.proc_plot.clear()
        self._plot_signal(smoothed_signal, self.proc_plot, color='g')

    def apply_detect(self):
        """Detect spikes on processed signal."""
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
  
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = LFPAnalyzer()
    win.show()
    sys.exit(app.exec_())