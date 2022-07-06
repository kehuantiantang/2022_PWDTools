# coding=utf-8
import os.path
import sys
import util.utils as cv2
from PyQt5.QtCore import Qt, QRectF, pyqtSignal, QT_VERSION_STR
from PyQt5.QtGui import QImage, QPixmap, QIntValidator
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QFileDialog, QWidget, QToolButton, QVBoxLayout, QHBoxLayout, \
    QLabel, QSlider, QLineEdit
import os.path as osp
from PyQt5.QtWidgets import QApplication

from vis_gui.json_polygon import JsonLoader

jl = JsonLoader()

class QtImageViewer(QGraphicsView):
    """ PyQt image viewer widget for a QPixmap in a QGraphicsView scene with mouse zooming and panning.
    Displays a QImage or QPixmap (QImage is internally converted to a QPixmap).
    To display any other image format, you must first convert it to a QImage or QPixmap.
    Some useful image format conversion utilities:
        qimage2ndarray: NumPy ndarray <==> QImage    (https://github.com/hmeine/qimage2ndarray)
        ImageQt: PIL Image <==> QImage  (https://github.com/python-pillow/Pillow/blob/master/PIL/ImageQt.py)
    Mouse interaction:
        Left mouse button drag: Pan image.
        Right mouse button drag: Zoom box.
        Right mouse button doubleclick: Zoom to show entire image.
    """

    # Mouse button signals emit image scene (x, y) coordinates.
    # !!! For image (row, column) matrix indexing, row = y and column = x.
    leftMouseButtonPressed = pyqtSignal(float, float)
    rightMouseButtonPressed = pyqtSignal(float, float)
    leftMouseButtonReleased = pyqtSignal(float, float)
    rightMouseButtonReleased = pyqtSignal(float, float)
    leftMouseButtonDoubleClicked = pyqtSignal(float, float)
    rightMouseButtonDoubleClicked = pyqtSignal(float, float)


    def get_all_image(self, path):
        self.image_paths = []
        for root, _, files in os.walk(path):
            for file in sorted(files):
                if file.lower().split('.')[-1] in ['jpg', 'png', 'jpeg']:
                    self.image_paths.append((file.split('.')[0], osp.join(root, file)))


    def get_image(self, index):
        name, path = self.image_paths[index]
        assert osp.exists(path)
        return name, path

    def __init__(self, file_path, save_method, txt_save_path, canZoom = False, canPan = False):
        self.save = save_method
        self.get_all_image(file_path)

        self.index, self.name, self.txt_save_path = 0, '', txt_save_path
        # Store a local handle to the scene's current image pixmap.
        self._pixmapHandle = None
        # Image aspect ratio mode.
        # !!! ONLY applies to full image. Aspect ratio is always ignored when zooming.
        #   Qt.IgnoreAspectRatio: Scale image to fit viewport.
        #   Qt.KeepAspectRatio: Scale image to fit inside viewport, preserving aspect ratio.
        #   Qt.KeepAspectRatioByExpanding: Scale image to fill the viewport, preserving aspect ratio.
        self.aspectRatioMode = Qt.KeepAspectRatio

        path = self.init()
        self.set_attributes(None)

        self.scene = QGraphicsScene()
        self.zoomStack = []

        # Image is displayed as a QPixmap in a QGraphicsScene attached to this QGraphicsView.
        QGraphicsView.__init__(self)
        self.loadImageFromFile(path)

        self.setScene(self.scene)


        # Scroll bar behaviour.
        #   Qt.ScrollBarAlwaysOff: Never shows a scroll bar.
        #   Qt.ScrollBarAlwaysOn: Always shows a scroll bar.
        #   Qt.ScrollBarAsNeeded: Shows a scroll bar only when zoomed.
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Stack of QRectF zoom boxes in scene coordinates.


        # Flags for enabling/disabling mouse interaction.
        self.canZoom = canZoom
        self.canPan = canPan

        self.leftMouseButtonDoubleClicked.connect(self.handleDoubleLeftClick)
        self.rightMouseButtonDoubleClicked.connect(self.handleDoubleRightClick)



    def init(self):
        name, path = self.get_image(self.index)
        self.name = name
        return path

    def hasImage(self):
        """ Returns whether or not the scene contains an image pixmap.
        """
        return self._pixmapHandle is not None

    def clearImage(self):
        """ Removes the current image pixmap from the scene if it exists.
        """
        if self.hasImage():
            self.scene.removeItem(self._pixmapHandle)
            self._pixmapHandle = None

    def pixmap(self):
        """ Returns the scene's current image pixmap as a QPixmap, or else None if no image exists.
        :rtype: QPixmap | None
        """
        if self.hasImage():
            return self._pixmapHandle.pixmap()
        return None

    def image(self):
        """ Returns the scene's current image pixmap as a QImage, or else None if no image exists.
        :rtype: QImage | None
        """
        if self.hasImage():
            return self._pixmapHandle.pixmap().toImage()
        return None

    def setImage(self, image):
        """ Set the scene's current image pixmap to the input QImage or QPixmap.
        Raises a RuntimeError if the input image has type other than QImage or QPixmap.
        :type image: QImage | QPixmap
        """
        if type(image) is QPixmap:
            pixmap = image
        elif type(image) is QImage:
            pixmap = QPixmap.fromImage(image)
        else:
            raise RuntimeError("ImageViewer.setImage: Argument must be a QImage or QPixmap.")
        if self.hasImage():
            self._pixmapHandle.setPixmap(pixmap)
        else:
            self._pixmapHandle = self.scene.addPixmap(pixmap)
        self.setSceneRect(QRectF(pixmap.rect()))  # Set scene size to image size.
        self.updateViewer()

    def loadImageFromFile(self, fileName=""):
        """ Load an image from file.
        Without any arguments, loadImageFromFile() will popup a file dialog to choose the image file.
        With a fileName argument, loadImageFromFile(fileName) will attempt to load the specified image file directly.
        """
        if len(fileName) == 0:
            if QT_VERSION_STR[0] == '4':
                fileName = QFileDialog.getOpenFileName(self, "Open image file.")
            elif QT_VERSION_STR[0] == '5':
                fileName, dummy = QFileDialog.getOpenFileName(self, "Open image file.")
        if len(fileName) and os.path.isfile(fileName):
            img = cv2.imread(fileName)

            # boxes, polygon image
            img = jl.draw_bboxes(img, self.attributes)
            # img = jl.draw_polygons(img, self.attributes)
            h, w, c = img.shape

            img = img[..., ::-1].copy()
            image = QImage(img.data, h, w,  3*h, QImage.Format_RGB888)
            self.setImage(image)

    def updateViewer(self):
        """ Show current zoom (if showing entire image, apply current aspect ratio mode).
        """
        if not self.hasImage():
            return
        if len(self.zoomStack) and self.sceneRect().contains(self.zoomStack[-1]):
            self.fitInView(self.zoomStack[-1], Qt.IgnoreAspectRatio)  # Show zoomed rect (ignore aspect ratio).
        else:
            self.zoomStack = []  # Clear the zoom stack (in case we got here because of an invalid zoom).
            self.fitInView(self.sceneRect(), self.aspectRatioMode)  # Show entire image (use current aspect ratio mode).

    def resizeEvent(self, event):
        """ Maintain current zoom on resize.
        """
        self.updateViewer()

    def mouseDoubleClickEvent(self, event):
        """ Show entire image.
        """
        scenePos = self.mapToScene(event.pos())
        if event.button() == Qt.LeftButton:
            self.leftMouseButtonDoubleClicked.emit(scenePos.x(), scenePos.y())
        elif event.button() == Qt.RightButton:
            if self.canZoom:
                self.zoomStack = []  # Clear zoom stack.
                self.updateViewer()
            self.rightMouseButtonDoubleClicked.emit(scenePos.x(), scenePos.y())
        QGraphicsView.mouseDoubleClickEvent(self, event)


    def handleDoubleLeftClick(self, x, y):
        index = max(0, self.index -1)
        name, filepath = self.get_image(index)
        self.index, self.name = index, name
        self.save((self.name, self.index), (name, index))
        self.set_attributes(None)
        self.loadImageFromFile(filepath)
        # current, next status


    def handleDoubleRightClick(self, x, y):
        index = min(len(self.image_paths) - 1, self.index +1)
        name, filepath = self.get_image(index)
        self.index, self.name = index, name
        self.save((self.name, self.index), (name, index))
        self.set_attributes(None)
        self.loadImageFromFile(filepath)
        # current, next status

    def image_change(self):
        """
        change visible image according to index
        """
        name, filepath = self.get_image(self.index)
        self.name = name
        self.save((self.name, self.index), (self.name, self.index))
        self.set_attributes(None)
        self.loadImageFromFile(filepath)

    def set_current_index(self, index):
        self.index = min(max(0, index), len(self.image_paths) - 1)

    def get_current_index(self):
        return self.index

    def set_attributes(self, attributes):
        if attributes is None:
            p = osp.join(self.txt_save_path, self.name + '.json')
            attributes = None
            if osp.exists(p):
                context = jl.load_json(p)
                attributes = jl.get_objects(context)
        self.attributes = attributes

class Window(QWidget):
    def __init__(self, file_path, txt_save_path, labels):
        super(Window, self).__init__()
        self.labels = labels

        self.txt_save_path = txt_save_path
        self.viewer = QtImageViewer(file_path, self.do_action, txt_save_path)

        self.current_index = None
        self.init_bottom()

        self.qLabel = QLabel(self.viewer.name)
        # self.qLabel.setBold(True)
        self.qLabel.setStyleSheet("font: 20pt;")

        VBlayout = QVBoxLayout(self)
        VBlayout.addWidget(self.qLabel)
        VBlayout.addWidget(self.viewer)


        BottomHBLayer = QHBoxLayout()
        BottomHBLayer.setAlignment(Qt.AlignRight)
        BottomHBLayer.addWidget(self.slide)
        BottomHBLayer.addWidget(self.input_index)
        BottomHBLayer.setStretchFactor(self.slide, 9)
        BottomHBLayer.setStretchFactor(self.input_index, 1)

        VBlayout.addLayout(BottomHBLayer)

        # init-
        self.do_action((self.viewer.name, self.viewer.index), (self.viewer.name, self.viewer.index))

    def push_save_button(self):
        self.do_action((self.viewer.name, self.viewer.index), (self.viewer.name, self.viewer.index))

    def do_action(self, current, next):
        '''
        做一些操作，比如保存文件，或者是读取下一个目录
        :param current:
        :param next:
        :return:
        '''
        print('Do action', current, next)
        n_name, n_index = next
        # next status

        self.qLabel.setText(n_name)
        self.viewer.set_attributes(self.load_json(n_name))

    def init_bottom(self):
        self.slide = QSlider(Qt.Horizontal, self)
        self.input_index = QLineEdit(self)
        self.input_index.setText('0')
        self.input_index.setValidator(QIntValidator())
        self.input_index.editingFinished.connect(self.inputChange)

        # self.save.clicked.connect(self.push_save_button)
        self.slide.setMinimum(0)
        self.slide.setMaximum(len(self.viewer.image_paths))
        self.slide.setSingleStep(1)
        self.slide.setTickPosition(QSlider.TicksBelow)
        #设置刻度间距
        self.slide.setTickInterval(max(10, len(self.viewer.image_paths) // 10))
        self.slide.valueChanged.connect(self.slideChange)

    def init_attributes(self):
        return self.load_json(self.viewer.name)

    def load_json(self, name):
        p = osp.join(self.txt_save_path, name + '.json')
        attributes = None
        if osp.exists(p):
            context = jl.load_json(p)
            attributes = jl.get_objects(context)
        return attributes

    def inputChange(self):
        index = int(self.input_index.text())
        if index != self.current_index:
            self.current_index = index
            self.viewer.set_current_index(index)
            self.viewer.image_change()
            # self.input_index.setText(str(index))
            self.slide.setValue(self.viewer.index)

    def slideChange(self):
        index = self.slide.value()
        if index != self.current_index:
            self.current_index = index
            self.viewer.set_current_index(index)
            self.viewer.image_change()
            self.input_index.setText(str(index))
            # self.slide.setValue(self.viewer.index)

if __name__ == '__main__':


    # Create the application.
    app = QApplication(sys.argv)

    # Create image viewer and load an image file to display.
    viewer = Window(file_path = r"H:\quality_assessment", txt_save_path = r"H:\quality_assessment", labels=  ['flower', 'fruit', 'leaf', 'stem or runner', 'crown'])

    viewer.setFixedSize(int(512*1.8), int(512*1.8))
    # Show viewer and run application.
    viewer.show()
    sys.exit(app.exec_())

    # contexts = load_json('/Users/sober/Downloads/Project/4ai_dataset/01_01_00000021.json')
    # print(contexts)
    # for object in contexts.segmentations:
    #     class_id = object.data.ID
