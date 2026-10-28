"""Contextual controls for the image's editing-tool toolbar."""

from PySide6 import QtCore, QtGui, QtWidgets


class ToolLayout(QtWidgets.QLayout):
    """Wrap tools so opening the sidebar never hides the active tool."""

    def __init__(self, parent):
        super().__init__(parent)
        self.items = []
        self.setContentsMargins(6, 6, 6, 6)
        self.setSpacing(6)

    def addItem(self, item):
        self.items.append(item)

    def count(self):
        return len(self.items)

    def itemAt(self, index):
        return self.items[index] if 0 <= index < self.count() else None

    def takeAt(self, index):
        return self.items.pop(index) if 0 <= index < self.count() else None

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self.arrange(QtCore.QRect(0, 0, width, 0), measure=True)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QtCore.QSize()
        for item in self.items:
            size = size.expandedTo(item.minimumSize())
        return size + QtCore.QSize(12, 12)

    def setGeometry(self, rectangle):
        super().setGeometry(rectangle)
        self.arrange(rectangle)

    def arrange(self, rectangle, measure=False):
        bounds = rectangle.adjusted(6, 6, -6, -6)
        x, y, height = bounds.x(), bounds.y(), 0
        for item in self.items:
            size = item.sizeHint()
            if height and x + size.width() > bounds.right() + 1:
                x, y, height = bounds.x(), y + height + self.spacing(), 0
            if not measure:
                item.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), size))
            x += size.width() + self.spacing()
            height = max(height, size.height())
        return y + height - rectangle.y() + 6


class ToolStrip(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        ToolLayout(self)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Minimum)

    def addAction(self, action):
        button = QtWidgets.QToolButton(self)
        button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly)
        button.setDefaultAction(action)
        self.layout().addWidget(button)

    def addWidget(self, widget):
        self.layout().addWidget(widget)


class EditingControls:
    def build_controls(self):
        self.mode = 'select'
        self.toolbar = ToolStrip(self.window)
        self.toolbar.setObjectName('annotation-tools')
        group = QtGui.QActionGroup(self)
        group.setExclusive(True)
        self.actions = {}
        for mode, text in [('select', 'Select'), ('move', 'Edit selection'), ('precise', 'Precise editing'),
                           ('proportional', 'Proportional editing'), ('add', 'Add new'), ('simplify', 'Simplify contours')]:
            action = QtGui.QAction(text, self)
            self.toolbar.addAction(action)
            action.setCheckable(True)
            action.setActionGroup(group)
            action.triggered.connect(lambda checked, mode=mode: self.set_tool_mode(mode))
            self.actions[mode] = action
        self.actions['select'].setChecked(True)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        self.selection_count = QtWidgets.QLabel('0 selected')
        layout.addWidget(self.selection_count)
        row = QtWidgets.QHBoxLayout()
        self.manual, self.program = QtWidgets.QCheckBox('Pre-existing'), QtWidgets.QCheckBox('This run')
        for box in (self.manual, self.program):
            box.setChecked(True)
            box.toggled.connect(self.visibility)
            row.addWidget(box)
        layout.addLayout(row)
        self.warning = QtWidgets.QLabel()
        self.warning.setWordWrap(True)
        layout.addWidget(self.warning)
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(['Annotation', 'Volume / distance'])
        self.tree.setMaximumHeight(210)
        self.tree.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tree.itemSelectionChanged.connect(self.tree_selection)
        self.tree.itemClicked.connect(lambda *_: self.reveal_selected_annotation())
        layout.addWidget(self.tree)
        row = QtWidgets.QHBoxLayout()
        delete = QtWidgets.QPushButton('Delete selected objects')
        delete.clicked.connect(self.delete_objects)
        row.addWidget(delete)
        self.undo_button = QtWidgets.QPushButton('Undo')
        self.undo_button.clicked.connect(lambda: self.document.undo.undo() if self.document else None)
        row.addWidget(self.undo_button)
        redo = QtWidgets.QPushButton('Redo')
        redo.clicked.connect(lambda: self.document.undo.redo() if self.document else None)
        row.addWidget(redo)
        layout.addLayout(row)

        def panel(title):
            widget = QtWidgets.QGroupBox(title)
            form = QtWidgets.QFormLayout(widget)
            layout.addWidget(widget)
            return widget, form

        self.move_controls, form = panel('Move selected objects')
        self.offsets = []
        for axis in 'XYZ':
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(-1e6, 1e6)
            spin.setDecimals(4)
            self.offsets.append(spin)
            form.addRow(f'Δ{axis} (µm)', spin)
        apply = QtWidgets.QPushButton('Apply translation')
        apply.clicked.connect(lambda: self.perform(lambda: self.document.translate(self.selections, [s.value() for s in self.offsets])))
        form.addRow(apply)

        self.point_controls, form = panel('Contour points')
        self.edit = QtWidgets.QCheckBox('Show points on this layer')
        self.edit.setChecked(True)
        self.edit.toggled.connect(self.draw_handles)
        form.addRow(self.edit)
        self.vertex = QtWidgets.QSpinBox(self)
        self.vertex.setMinimum(1)
        self.vertex.hide()
        self.vertex.valueChanged.connect(self.coordinates)
        self.xyz = []
        for axis in 'XYZ':
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(-1e6, 1e6)
            spin.setDecimals(4)
            self.xyz.append(spin)
            form.addRow(f'{axis} (µm)', spin)
        apply = QtWidgets.QPushButton('Apply point position')
        apply.clicked.connect(lambda: self.apply_point_position([s.value() for s in self.xyz]))
        form.addRow(apply)
        remove = QtWidgets.QPushButton('Delete selected points')
        remove.clicked.connect(self.delete_point)
        form.addRow(remove)
        self.proportional_controls, form = panel('Proportional editing · 3D neighborhood')
        self.radius = QtWidgets.QDoubleSpinBox()
        self.radius.setRange(.001, 10000)
        self.radius.setDecimals(3)
        self.radius.setValue(2)
        self.radius.setSingleStep(.2)
        self.radius.valueChanged.connect(self.draw_handles)
        form.addRow('Influence radius (µm)', self.radius)
        self.falloff = QtWidgets.QComboBox()
        self.falloff.addItems(['Smooth', 'Linear'])
        form.addRow('Falloff', self.falloff)

        self.add_controls, form = panel('Add new')
        self.tool = QtWidgets.QComboBox()
        self.tool.addItems(['Layered cell contour', 'Contour point', 'Signal'])
        self.tool.currentIndexChanged.connect(self.update_controls)
        form.addRow('Create', self.tool)
        self.signal_channel = QtWidgets.QComboBox()
        form.addRow(self.signal_channel)
        self.draft_info = QtWidgets.QLabel('No layers drawn')
        form.addRow(self.draft_info)
        self.close_layer_button = QtWidgets.QPushButton('Close layer (Enter)')
        self.close_layer_button.clicked.connect(self.finish_layer)
        form.addRow(self.close_layer_button)
        self.finish_cell_button = QtWidgets.QPushButton('Create cell from layers')
        self.finish_cell_button.clicked.connect(self.finish_cell)
        form.addRow(self.finish_cell_button)
        self.cancel_draft_button = QtWidgets.QPushButton('Discard drawing (Esc)')
        self.cancel_draft_button.clicked.connect(self.cancel_draft)
        form.addRow(self.cancel_draft_button)

        self.simplify_controls, form = panel('Simplify selected contours')
        self.reduction = QtWidgets.QDoubleSpinBox()
        self.reduction.setRange(1, 90)
        self.reduction.setValue(40)
        self.reduction.setSuffix('%')
        form.addRow('Remove up to', self.reduction)
        self.volume_tolerance = QtWidgets.QDoubleSpinBox()
        self.volume_tolerance.setRange(.1, 50)
        self.volume_tolerance.setValue(5)
        self.volume_tolerance.setSuffix('%')
        form.addRow('Max volume change', self.volume_tolerance)
        simplify = QtWidgets.QPushButton('Simplify · undoable')
        simplify.clicked.connect(self.simplify_selected)
        form.addRow(simplify)
        self.simplification_report = QtWidgets.QPlainTextEdit()
        self.simplification_report.setReadOnly(True)
        self.simplification_report.setMaximumHeight(180)
        self.simplification_report.setAccessibleName('Simplification results')
        self.simplification_report.hide()
        form.addRow(self.simplification_report)
        self.export = QtWidgets.QPushButton('Export edited Imaris…')
        self.export.clicked.connect(self.export_scene)
        layout.addWidget(self.export)
        self.info = QtWidgets.QLabel()
        self.info.setWordWrap(True)
        self.info.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
        layout.addWidget(self.info)

    def precise_active(self):
        return (self.mode in ('precise', 'proportional') or self.mode == 'move' and self.single_contour())

    def single_contour(self):
        return len(self.selections) == 1 and self.selection and self.selection[0] in ('surface', 'contour')

    def set_tool_mode(self, mode):
        self.mode = mode
        self.actions[mode].setChecked(True)
        if hasattr(self.window, 'annotation_dock'):
            self.window.annotation_dock.setVisible(mode != 'select')
        self.update_controls()
        self.draw_handles()

    def update_controls(self, *_):
        single = self.single_contour()
        self.selection_count.setText(f'{len(self.selections)} objects · {len(self.point_selection)} points')
        self.actions['move'].setEnabled(bool(self.selections))
        self.actions['precise'].setEnabled(bool(single))
        self.actions['proportional'].setEnabled(bool(single and self.selection[0] == 'surface'))
        self.actions['simplify'].setEnabled(any(k in ('surface', 'contour') for k, _ in self.selections))
        self.move_controls.setVisible(self.mode == 'move' and not single)
        self.point_controls.setVisible(bool(self.precise_active() and single))
        self.proportional_controls.setVisible(self.mode == 'proportional')
        self.add_controls.setVisible(self.mode == 'add')
        drawing = self.tool.currentIndex() == 0
        self.signal_channel.setVisible(self.tool.currentIndex() == 2)
        for widget in (self.draft_info, self.close_layer_button, self.finish_cell_button, self.cancel_draft_button):
            widget.setVisible(drawing)
        self.simplify_controls.setVisible(self.mode == 'simplify')
        messages = {
            'select': 'Click a visible object; Shift adds to selection. Drag a box to select an area. Click a selected contour dot to select that point. Middle-drag or Space+drag pans; Ctrl+wheel zooms.',
            'move': 'Drag selected objects to move them together. Their assigned signals follow their cells. For one contour, use its individual points.',
            'precise': 'Drag a white dot to move it. Shift+click adds points. Delete removes selected points; Delete selected objects removes the cell and its signals.',
            'proportional': 'Drag a dot to deform nearby mesh points in a 3D sphere. Influence radius is in µm; falloff tapers movement to zero at the radius.',
            'add': 'Signal: click to place A. Contour point: click a selected cell border. Layered cell: click a polygon, close the layer, change slice and draw more layers, then Create cell. Double-click also closes a layer.',
            'simplify': 'Reduce contour points while preserving a closed mesh and limiting volume change. Volumes and distances update after simplification.'}
        self.info.setText(messages[self.mode])
