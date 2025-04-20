import sys
import glob
import os
import neo
from neo.io import NixIO
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import QSettings, QDir, Qt
import pyqtgraph as pg

class LFPAnalyzer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('LFP Data Analyzer')
        self.resize(1200, 800)
        self.signals = {}
        self.current_file = None
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

        # File operation buttons
        btn_layout = QtWidgets.QHBoxLayout()
        self.btn_add = QtWidgets.QPushButton('Add')
        self.btn_remove = QtWidgets.QPushButton('Remove')
        self.btn_save = QtWidgets.QPushButton('Save')
        btn_layout.addWidget(self.btn_add)
        btn_layout.addWidget(self.btn_remove)
        btn_layout.addWidget(self.btn_save)
        left_layout.addLayout(btn_layout)

        # Bin x-axis (Downsample)
        bin_layout = QtWidgets.QHBoxLayout()
        bin_layout.addWidget(QtWidgets.QLabel('Downsample:'))
        self.spin_down = QtWidgets.QSpinBox()
        self.spin_down.setRange(1, 100)
        self.spin_down.setValue(20)
        bin_layout.addWidget(self.spin_down)
        self.btn_apply = QtWidgets.QPushButton('Apply')
        bin_layout.addWidget(self.btn_apply)
        left_layout.addLayout(bin_layout)

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

    #unique name
    def _generate_unique_name(self, base_name):
        """Return a unique name not already in self.signals."""
        name = base_name
        counter = 1
        while name in self.signals:
            name = f"{base_name}_v{counter}"
            counter += 1
        return name

    #load path
    def load_single_abf(self, path):
        """Helper to load one ABF file at path and update UI."""
        try:
            reader = neo.io.AxonIO(filename=path)
            block = reader.read_block(lazy=False)
            seg = block.segments[0]
            signal = seg.analogsignals[0]
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error", f"Failed to load {os.path.basename(path)}:\n{e}")
            return
        # unique name
        base_name = os.path.basename(path)
        display_name = self._generate_unique_name(base_name)
        # store
        self.signals[display_name] = signal
        # add to tree, save path to UserRole
        item = QtWidgets.QTreeWidgetItem([display_name])
        item.setData(0, QtCore.Qt.UserRole, display_name)
        self.file_tree.addTopLevelItem(item)
        # if first loaded or no current, set as current
        if self.current_file is None:
            self.current_file = path
            self.raw_signal = signal
            # plot raw
            self.raw_plot.clear()
            times = signal.times.rescale('s').magnitude.flatten()
            values = signal.magnitude.flatten()
            self.raw_plot.plot(times, values, pen='b')

    #load file
    def load_file(self):
        """Load ABF files or folder, allowing both append"""
        last_dir = self.settings.value('lastDir', QDir.homePath())

        # try folder
        dirpath = QtWidgets.QFileDialog.getExistingDirectory(
            self, 'Select ABF Directory', 
            last_dir, QtWidgets.QFileDialog.ShowDirsOnly
        )
        if dirpath:
            self.settings.setValue('lastDir', dirpath)
            # get all .abf
            all_abfs = sorted(glob.glob(os.path.join(dirpath, '*.abf')))
            if not all_abfs:
                QtWidgets.QMessageBox.warning(
                    self, 'Warning', 'No ABF files found in that directory.')
                return

            # get new files
            new_abfs = [f for f in all_abfs if f not in self.signals]

            if not new_abfs:
                # reload?
                reply = QtWidgets.QMessageBox.question(
                    self,
                    'No new files',
                    'All the ABF has been loaded。\nClear and reload？',
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
                )
                if reply == QtWidgets.QMessageBox.No:
                    return
                # Reload: clear old trace
                self.file_tree.clear()
                self.signals.clear()
                self.current_file = None
                self.raw_signal = None
                self.proc_signal = None
                self.raw_plot.clear()
                self.proc_plot.clear()
                new_abfs = all_abfs

            # load_single_abf (tree and signals)
            for f in new_abfs:
                self.load_single_abf(f)
            return

        # if not folder, sigle file
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, 'Select ABF File(s)', last_dir, 'ABF Files (*.abf)'
        )
        if not files:
            return
        self.settings.setValue('lastDir', os.path.dirname(files[0]))

        for f in files:
            if f in self.signals:
                continue
            self.load_single_abf(f)

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
        self.current_file = display_name
        self.raw_signal = self.signals[display_name]
        self.proc_signal = None

        # redraw raw_plot
        self.raw_plot.clear()
        self._plot_raw(self.raw_signal)

        # clear processe_plot
        self.proc_plot.clear()

    #plot signal
    def _plot_raw(self, signal):
        t = signal.times.rescale('s').magnitude.flatten()
        y = signal.magnitude.flatten()
        self.raw_plot.plot(t, y, pen=pg.mkPen('b', width=1))

    def remove_file(self):
        """Remove selected file from tree and clear plots if needed."""
        pass

    def batch_process(self):
        """Execute pipeline on all files in tree."""
        pass

    def apply_smooth(self):
        """Apply ALS detrend to current raw signal."""
        pass

    def apply_detect(self):
        """Detect spikes on processed signal."""
        pass

    #size reduce
    def apply_downsample(self):
        """click Apply run downsampl, draw and list in tree"""
        if self.raw_signal is None:
            QtWidgets.QMessageBox.warning(self, "Warning", "No raw signal to downsample.")
            return

        factor = self.spin_down.value()
        try:
            dsig = self.raw_signal[::factor]
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Downsampling failed:\n{e}")
            return

        # generate and apply unique display_name
        factor = self.spin_down.value()
        base = os.path.basename(self.current_file)
        base_wo_ext = os.path.splitext(base)[0]
        base_display_name = f"{base_wo_ext}_down{factor}x"
        display_name = self._generate_unique_name(base_display_name)

        #add to tree and selected
        self.signals[display_name] = dsig
        self.proc_signal = dsig

        item = QtWidgets.QTreeWidgetItem([display_name])
        item.setData(0, QtCore.Qt.UserRole, display_name)
        self.file_tree.addTopLevelItem(item)
        self.file_tree.setCurrentItem(item)

        #draw processed signal
        self.proc_plot.clear()
        t = dsig.times.rescale('s').magnitude.flatten()
        y = dsig.magnitude.flatten()
        self.proc_plot.plot(t, y, pen=pg.mkPen('r', width=1))

    #save in hdf5
    def save(self):
        """Save selected signals as separate HDF5 files with auto-naming."""
        selected_items = self.file_tree.selectedItems()
        if not selected_items:
            QtWidgets.QMessageBox.warning(self, "Warning", "No signals selected in the file tree.")
            return

        # Select output folder
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Output Folder", QDir.homePath()
        )
        if not out_dir:
            return

        errors = []

        for item in selected_items:
            key = item.data(0, QtCore.Qt.UserRole)
            if key not in self.signals:
                errors.append(str(key))
                continue

            signal = self.signals[key]

            # Auto-generated name
            base_name_wo_ext = os.path.splitext(str(key))[0]  # key = tree
            default_name = f"{base_name_wo_ext}.h5"
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
        
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = LFPAnalyzer()
    win.show()
    sys.exit(app.exec_())