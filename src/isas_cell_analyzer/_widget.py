from qtpy.QtWidgets import (
    QLabel,
    QGridLayout,
    QVBoxLayout,
    QPushButton,
    QWidget,
    QScrollArea,
    QLineEdit,
    QComboBox,
    QSizePolicy,
)
import napari

class CellAnalyzer(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, viewer: napari.viewer.Viewer):
        super().__init__()
        self.viewer = viewer

        ### QObjects
        # objects that can be updated are attributes of the class
        # for ease of access

        # Labels
        title = QLabel("ISAS Cell Analyzer")
        label_start_id = QLabel("Start analysis at:")
        label_included = QLabel("Included:")
        label_excluded = QLabel("Excluded:")
        label_remaining = QLabel("Remaining:")
        label_mean = QLabel("Mean size:")
        label_std = QLabel("Mean std:")
        label_conversion = QLabel("1 pixel equals:")
        self.label_amount_included = QLabel("1234")
        self.label_amount_excluded = QLabel("4231")
        self.label_amount_remaining = QLabel("9999")
        self.label_mean_included = QLabel("123.456")
        self.label_std_included = QLabel("11.111")

        label_mean.setToolTip("Only accounting for cells which have been inclded")
        label_std.setToolTip("Only accounting for cells which have been inclded")

        # Buttons
        btn_start_analysis = QPushButton("Start analysis")
        btn_export = QPushButton("Export")
        btn_import = QPushButton("Import")
        self.btn_include = QPushButton("Include")
        self.btn_exclude = QPushButton("Exclude")

        btn_export.setToolTip("Export tooltip")
        btn_import.setToolTip("Import tooltip")

        self.btn_include.setEnabled(False)
        self.btn_exclude.setEnabled(False)

        # LineEdits
        self.lineedit_start_id = QLineEdit()
        self.lineedit_conversion_rate = QLineEdit()

        # Comboboxes
        self.combobox_conversion_unit = QComboBox()

        self.combobox_conversion_unit.addItems(["mm", "µm", "nm"])

        # Horizontal lines
        spacer = QWidget()
        spacer.setFixedHeight(4)
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        line1 = QWidget()
        line1.setFixedHeight(4)
        line1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        line1.setStyleSheet("background-color: #c0c0c0")

        line2 = QWidget()
        line2.setFixedHeight(4)
        line2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        line2.setStyleSheet("background-color: #c0c0c0")

        ### GUI
        content = QWidget()
        content.setLayout(QGridLayout())
        content.layout().addWidget(title, 0, 0, 1, -1)

        content.layout().addWidget(spacer, 1, 0, 1, -1)

        content.layout().addWidget(btn_import, 2, 0, 1, 1)
        content.layout().addWidget(btn_export, 2, 2, 1, 1)

        content.layout().addWidget(btn_start_analysis, 3, 0, 1, 1)
        content.layout().addWidget(label_start_id, 3, 1, 1, 1)
        content.layout().addWidget(self.lineedit_start_id, 3, 2, 1, 1)
        
        content.layout().addWidget(line1, 4, 0, 1, -1)

        content.layout().addWidget(self.btn_exclude, 5, 0, 1, 1)
        content.layout().addWidget(self.btn_include, 5, 2, 1, 1)
        
        content.layout().addWidget(label_included, 6, 0, 1, 1)
        content.layout().addWidget(self.label_amount_included, 6, 2, 1, 1)

        content.layout().addWidget(label_mean, 7, 0, 1, 1)
        content.layout().addWidget(self.label_mean_included, 7, 2, 1, 1)

        content.layout().addWidget(label_std, 8, 0, 1, 1)
        content.layout().addWidget(self.label_std_included, 8, 2, 1, 1)

        content.layout().addWidget(label_excluded, 9, 0, 1, 1)
        content.layout().addWidget(self.label_amount_excluded, 9, 2, 1, 1)
        
        content.layout().addWidget(label_remaining, 10, 0, 1, 1)
        content.layout().addWidget(self.label_amount_remaining, 10, 2, 1, 1)
        
        content.layout().addWidget(line2, 11, 0, 1, -1)

        content.layout().addWidget(label_conversion, 12, 0, 1, 1)
        content.layout().addWidget(self.lineedit_conversion_rate, 12, 1, 1, 1)
        content.layout().addWidget(self.combobox_conversion_unit, 12, 2, 1, 1)

        scroll_area = QScrollArea()
        scroll_area.setWidget(content)
        scroll_area.setWidgetResizable(True)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(scroll_area)
