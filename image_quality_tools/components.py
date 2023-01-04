# coding=utf-8
from PyQt5 import QtGui
from PyQt5.QtCore import Qt, pyqtSignal, QRectF
from PyQt5.QtGui import QIntValidator, QPixmap, QImage
from PyQt5.QtWidgets import QSlider, QLabel, QWidget, QLineEdit, QGraphicsView, QGraphicsScene, QProgressDialog, \
    QApplication
import os.path as osp
import numpy as np
import os
import sys
import cv2
from json_polygon import JsonLoader


class QSlideReadOnly(QSlider):

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent) -> None:
        # event = super().mouseMoveEvent(ev)
        pass

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent) -> None:
        # event = super().mouseReleaseEvent(ev)
        pass

    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        # event = super().mousePressEvent(ev)
        pass

    def keyPressEvent(self, ev: QtGui.QKeyEvent) -> None:
        # super().keyPressEvent(ev)
        if ev.key() == Qt.Key_Up or ev.key() == Qt.Key_Down:
            pass
        # elif ev.key() == Qt.Key_Left or ev.key() == Qt.Key_Right:
        #     pass
        else:
            super().keyPressEvent(ev)
        pass

class UpdateWidget(QWidget):
    def __init__(self, **kwargs):
        super(UpdateWidget, self).__init__()
        self.update_widget(**kwargs)

    def update_widget(self, **kwargs):
        pass

class TitleLabel(QLabel, UpdateWidget):

    def update_widget(self, **kwargs):
        name = kwargs.get('name')
        assert name is not None, 'Title input cannot be None'
        self.setStyleSheet("font: 16pt;")
        self.setText(name)

class UpdateLineEdit(QLineEdit, UpdateWidget):

    def update_widget(self, **kwargs):
        index, total = kwargs.get('index', None), kwargs.get('total', None)
        assert index is not None and total is not None, 'lineEdict None'
        self.setText('%s/%s'%(max(1, index+1), total))
        self.setReadOnly(True)
        self.setValidator(QIntValidator())


class CheckLabel(QLabel, UpdateWidget):
    # KEEP = 0
    # REJECT = 1
    # NOT_CHECK = 2

    NOT_CHECK = -1
    STATUS_0 = 0
    STATUS_1 = 1
    STATUS_2 = 2
    STATUS_3 = 3
    STATUS_4 = 4


    # def __init__(self, **kwargs):
    #     super(CheckLabel, self).__init__()
    #     self.update_widget(**kwargs)


    def set_selected(self, status) -> None:
        # print(status)
        # if status == CheckLabel.KEEP:
        #     self.setStyleSheet("font: 20pt;")
        #     self.setText("<font color='green'>Keep</font>")
        # elif status == CheckLabel.REJECT:
        #     self.setStyleSheet("font: 20pt;")
        #     self.setText("<font color='red'>Reject</font>")
        # elif status == CheckLabel.NOT_CHECK:
        #     self.setStyleSheet("font: 20pt;")
        #     self.setText("<font color='gray'>Not check</font>")
        # else:
        #     assert ValueError()



        if status == CheckLabel.NOT_CHECK:
            self.setStyleSheet("font: 16pt;")
            self.setText("<font color='gray'>Not check</font>")
        elif status in [CheckLabel.STATUS_0, CheckLabel.STATUS_1, CheckLabel.STATUS_2, CheckLabel.STATUS_3,
                        CheckLabel.STATUS_4]:
            self.setStyleSheet("font: 16pt;")
            self.setText("<font color='green'>Level:%s</font>"%status)
        else:
            assert ValueError('Other style')

    def update_widget(self, **kwargs):
        name, assessment = kwargs.get('name', None), kwargs.get('assessment', None)
        assert name is not None and assessment is not None, 'CheckLabel cannot be None'
        if name in assessment.keys():
            self.set_selected(assessment[name])
        else:
            self.set_selected(CheckLabel.NOT_CHECK)


class ImageUtil(object):
    def __init__(self, img_path, json_path, target_path, filter_bbox = 0.7):
        os.makedirs(target_path, exist_ok=True)
        self.filter_bbox = filter_bbox
        assert osp.split(img_path)[-1].split('.')[0] == osp.split(json_path)[-1].split('.')[0], 'Image file and json file ' \
                                                                                        'must match, but %s:%s'%(
            osp.split(img_path)[-1].split('.')[0], osp.split(json_path)[-1].split('.')[0])

        name = osp.split(img_path)[-1].split('.')[0]
        self.jl = JsonLoader()
        # read json annotation file, if not exist in target path, read from source
        target_json_path = osp.join(target_path, name + '.json')
        self.obj_dicts = self.__load_objs(json_path if not osp.exists(target_json_path) else osp.exists(target_json_path))
        self.img_path = img_path

    def __draw_mask(self, img_h, img_w):
        self.obj_dicts = self.jl.resize_annotation(self.obj_dicts, (img_h, img_w))

        # height, width = self.obj_dicts['height'], self.obj_dicts['width']
        mask = np.zeros((img_h, img_w, 3), dtype=np.uint8)


        np.random.seed(0)
        # TODO json 这里图片大小和输入的不一样
        for score, value, difficult in zip(self.obj_dicts['scores'], self.obj_dicts['bboxes'], self.obj_dicts['difficults']):
            # if score > self.filter_bbox and difficult:
            # TODO difficult
            if score > self.filter_bbox:
                xmin, ymin, xmax, ymax = map(round, value)
                mask[ymin:ymax, xmin:xmax, :] = [np.random.randint(0, 256) for _ in range(0, 3)]
        return mask




    def get_QimgWithMask(self):
        # img = cv2.imread(self.img_path, cv2.IMREAD_UNCHANGED)
        img = cv2.imdecode(np.fromfile(self.img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        # # boxes, polygon image
        # h, w, c = img.shape
        img = img[..., ::-1].copy().astype(np.uint8)

        # img = self.jl.draw_bboxes(img, self.obj_dicts)
        h, w = 800, 800
        img = cv2.resize(img, (h, w)).astype(np.uint8)
        self.obj_dicts = self.jl.resize_annotation(self.obj_dicts, (h, w))
        img = self.jl.draw_bboxes(img, self.obj_dicts)
        # mask = self.__draw_mask(h, w)

        # imgAddMask = cv2.addWeighted(img,0.7, mask, 0.3, 30)

        image = QImage(img.data, h, w,  3*h, QImage.Format_RGB888)
        return image

    def __load_objs(self, path):
        '''
            read file from json file
            obj_dicts =  {'name':[], 'bboxes':[], 'category_name':[],
                         'name_pattern': '', 'height':height, 'width':width, 'path':path, 'polygons':[],
                         'filename':path, 'difficult':[]}
        :param path:
        :return:
        '''
        context = self.jl.load_json(path)
        obj_dicts = self.jl.get_objects(context)
        return obj_dicts

    def __get_distance(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        return (x1 - x2)^2 + (y1 - y2)^2

    def set_selected(self, index):
        """
        if the click position has bbox:
            1. if the bbox has not selected, it will selected.
            2. otherwise, the selection will cancel
        :param index:
        :return:
        """
        self.obj_dicts['difficults'][index] = not self.obj_dicts['difficults'][index]

    def select_bbox(self, x, y):
        selected_bbox = []
        for index, value in enumerate(self.obj_dicts['bboxes']):
            xmin, ymin, xmax, ymax = value['bbox']
            center = (xmax - xmin, ymax - ymin)
            if xmin <= x <= xmax and ymin <= y <= ymax:
                selected_bbox.append((center, index))

        nearest_one, min_distance = None, sys.maxsize
        for center, index in selected_bbox:
            distance = self.__get_distance(center, (x, y))
            if distance < min_distance:
                min_distance = distance
                nearest_one = index
        self.set_selected(nearest_one)

    def get_objs(self):
        return self.obj_dicts



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


    def __get_image(self, index):
        name, img_path, json_path = self.image_paths[index]
        assert osp.exists(img_path), 'Image path should exist !'
        assert osp.exists(json_path), 'Image path should exist !'
        return name, img_path, json_path


    def __init__(self, save_method, image_paths, config_obj, canZoom = False, canPan = False):
        self.save = save_method
        self.image_paths = image_paths
        self.config_obj = config_obj
        self.index = config_obj.config['index']


        # Store a local handle to the scene's current image pixmap.
        self._pixmapHandle = None
        # Image aspect ratio mode.
        # !!! ONLY applies to full image. Aspect ratio is always ignored when zooming.
        #   Qt.IgnoreAspectRatio: Scale image to fit viewport.
        #   Qt.KeepAspectRatio: Scale image to fit inside viewport, preserving aspect ratio.
        #   Qt.KeepAspectRatioByExpanding: Scale image to fill the viewport, preserving aspect ratio.
        self.aspectRatioMode = Qt.KeepAspectRatio

        img_path, json_path = self.init()

        self.scene = QGraphicsScene()
        self.zoomStack = []

        # Image is displayed as a QPixmap in a QGraphicsScene attached to this QGraphicsView.
        QGraphicsView.__init__(self)
        self.loadImageFromFile(img_path, json_path)

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

        # self.leftMouseButtonDoubleClicked.connect(self.handleDoubleLeftClick)
        # self.rightMouseButtonDoubleClicked.connect(self.handleDoubleRightClick)


    def init(self):
        name, img_path, json_path = self.__get_image(self.index)
        self.name = name
        return img_path, json_path

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


    def loadImageFromFile(self, img_path="", json_path = None):
        """ Load an image from file.
        Without any arguments, loadImageFromFile() will popup a file dialog to choose the image file.
        With a fileName argument, loadImageFromFile(fileName) will attempt to load the specified image file directly.
        """

        imgUtil = ImageUtil(img_path, json_path, self.config_obj.config['target_path'])
        self.setImage(imgUtil.get_QimgWithMask())

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
        return QGraphicsView.mouseDoubleClickEvent(self, event)



    def image_change(self):
        """
        change visible image according to index
        """
        name, img_path, json_path = self.__get_image(self.index)
        self.name = name
        # self.save((self.name, self.index), (self.name, self.index))
        self.loadImageFromFile(img_path, json_path)

        self.config_obj.update({'name':self.name, 'index':self.index})

    def set_current_index(self, index):
        self.index = min(max(0, index), len(self.image_paths) - 1)


    def get_current_index(self):
        return self.index

