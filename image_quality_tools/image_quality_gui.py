# coding=utf-8
import os
import os.path
import sys
import traceback

from tqdm import tqdm
from PyQt5 import QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QSlider, QMessageBox, QFileDialog, QProgressDialog, \
    QMainWindow
import os.path as osp
from PyQt5.QtWidgets import QApplication

from components import CheckLabel, QSlideReadOnly, TitleLabel, UpdateLineEdit, QtImageViewer
from misc import ConfigUtil, get_resource_path


class Window(QWidget):

    def get_names(self, path):

        get_name_dialog = QProgressDialog()
        get_name_dialog.setWindowModality(Qt.WindowModal)
        get_name_dialog.setMinimumDuration(0)
        get_name_dialog.setWindowTitle('Warning')
        get_name_dialog.setLabelText('Reading files, please wait a moment...')
        get_name_dialog.setRange(0, 100)

        assert osp.exists(path), 'Input path must exist! %s' % path
        name_paths = []
        files = os.listdir(path)
        for i, file in enumerate(files):
            if file.lower().split('.')[-1] in ['jpg', 'png', 'jpeg', 'tif'] and '_vis' not in file.lower().split('.')[
                0]:
                name = file.split('.')[0]
                img_path = osp.join(path, file)
                json_path = osp.join(path, '%s.json' % name)
                # assert osp.exists(json_path), 'The json file %s must exist!'%json_path
                # name, img_path, json_path
                if osp.exists(json_path):
                    name_paths.append((name, img_path, json_path))

                if get_name_dialog.wasCanceled():
                    sys.exit(0)

                get_name_dialog.setValue(int(i * 100.0 / len(files)))
        get_name_dialog.close()

        return name_paths


    def init_above(self):

        self.titleLabel = TitleLabel(name=self.image_viewer.name)
        self.checkedLabel = CheckLabel(name=self.image_viewer.name, assessment=self.config_obj.config['assessment'])


        TitleHBLayer = QHBoxLayout()
        # TitleHBLayer.setAlignment(Qt.AlignRight)
        TitleHBLayer.addWidget(self.titleLabel)
        TitleHBLayer.addWidget(self.checkedLabel)
        TitleHBLayer.setStretchFactor(self.titleLabel, 5)
        TitleHBLayer.setStretchFactor(self.checkedLabel, 5)

        return TitleHBLayer

    @property
    def init_bottom(self):
        self.input_index = UpdateLineEdit(index = self.config_obj.config['index'], total = self.config_obj.config[
            'total'])


        self.slide = QSlideReadOnly(Qt.Horizontal, self)
        self.slide.setMinimum(0)
        self.slide.setMaximum(len(self.image_viewer.image_paths) -1)
        self.slide.setSingleStep(1)
        self.slide.setValue(self.config_obj.config['index'])
        self.slide.setTickPosition(QSlider.TicksBelow)

        #设置刻度间距
        self.slide.setTickInterval(max(10, len(self.image_viewer.image_paths) // 10))
        self.slide.valueChanged.connect(self.slideChange)

        BottomHBLayer = QHBoxLayout()
        BottomHBLayer.setAlignment(Qt.AlignRight)
        BottomHBLayer.addWidget(self.slide)
        BottomHBLayer.addWidget(self.input_index)
        BottomHBLayer.setStretchFactor(self.slide, 9)
        BottomHBLayer.setStretchFactor(self.input_index, 1)

        return BottomHBLayer

    def __init__(self, source_path, target_path):
        super(Window, self).__init__()
        self.old_hook = sys.excepthook
        sys.excepthook = self.catch_exceptions

        self.setWindowIcon(QtGui.QIcon(get_resource_path('logo.ico')))

        self.image_paths = self.get_names(source_path)
        assert len(self.image_paths) > 0, 'Selected folder must include the image and json file'
        self.config_obj = ConfigUtil({'source_path': source_path, 'index': 0, 'target_path': target_path,
                                         'total':len(self.image_paths),
                                        'name':self.image_paths[0][0], 'assessment': {}})
        self.image_viewer = QtImageViewer(None, self.image_paths, self.config_obj)
        self.image_viewer.setFocusPolicy(Qt.NoFocus)
        os.makedirs(target_path, exist_ok=True)


        self.current_index = self.config_obj.config['index']
        AboveHBLayer = self.init_above()
        BottomHBLayer = self.init_bottom

        self.layout = QVBoxLayout()
        self.layout.addLayout(AboveHBLayer)
        self.layout.addWidget(self.image_viewer)
        self.layout.addLayout(BottomHBLayer)

        self.setLayout(self.layout)


        self.config_obj.add_update_widget(self.checkedLabel, ('name', 'assessment'))
        self.config_obj.add_update_widget(self.titleLabel, ('name', ))
        self.config_obj.add_update_widget(self.input_index, ('index', 'total'))




    def keyPressEvent(self, e):
        # print(e.key())
        if e.key() == Qt.Key_QuoteLeft:
            self.config_obj.update({'name':self.image_viewer.name, 'index':self.image_viewer.index, 'assessment':(self.image_viewer.name,
                                                                                                                  0)})
            self.slide.setValue(min(self.image_viewer.index + 1, self.config_obj.config['total'] - 1))
            # print(0)
        elif e.key() == Qt.Key_1:
            self.config_obj.update({'name':self.image_viewer.name, 'index':self.image_viewer.index, 'assessment':(self.image_viewer.name,
                                                                                                                  1)})
            self.slide.setValue(min(self.image_viewer.index + 1, self.config_obj.config['total'] - 1))
            # print(1)
        elif e.key() == Qt.Key_2:
            self.config_obj.update({'name':self.image_viewer.name, 'index':self.image_viewer.index, 'assessment':(self.image_viewer.name,
                                                                                                                  2)})
            self.slide.setValue(min(self.image_viewer.index + 1, self.config_obj.config['total'] - 1))
            # print(2)
        elif e.key() == Qt.Key_3:
            self.config_obj.update({'name':self.image_viewer.name, 'index':self.image_viewer.index, 'assessment':(self.image_viewer.name,
                                                                                                                  3)})
            self.slide.setValue(min(self.image_viewer.index + 1, self.config_obj.config['total'] - 1))
            # print(3)
        elif e.key() == Qt.Key_4:
            self.config_obj.update({'name':self.image_viewer.name, 'index':self.image_viewer.index, 'assessment':(self.image_viewer.name,
                                                                                                                  4)})
            self.slide.setValue(min(self.image_viewer.index + 1, self.config_obj.config['total'] - 1))
            # print(4)
        # elif e.key() == Qt.Key_5:
        #     self.slide.setValue(min(self.viewer.index + 1, self.config_obj.config['total']))
        #     print(5)
        else:
            # assert ValueError()
            QMessageBox().information(None, "Warning", "Please press `1234 to give score!", QMessageBox.Yes)

            self.config_obj.update()
            
            super(Window, self).keyPressEvent(e)



    def slideChange(self):
        index = min(max(0, self.slide.value()), self.config_obj.config['total'] - 1)
        if index != self.current_index:
            # if check current image has been check or not
            current_name, _, _ = self.image_paths[self.current_index]
            next_name, _, _ = self.image_paths[index]
            if current_name in self.config_obj.config['assessment'] or next_name in self.config_obj.config['assessment']:
                self.current_index = index
                self.image_viewer.set_current_index(index)
                self.image_viewer.image_change()
            else:
                QMessageBox().warning(None, "Warning", "Please label current image !", QMessageBox.Yes)

                self.slide.setValue(self.current_index)


    def closeEvent(self, event):
        print('closed!', '*'*10)
        self.config_obj.update()
        return super().closeEvent(event)


    def catch_exceptions(self, exc_type, exc_value, exc_tb):
        print(exc_type, exc_value, traceback)
        traceback_string = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        QMessageBox.critical(None, 'An exception was raised', "%s"%(traceback_string), QMessageBox.NoButton)
        self.old_hook(exc_type, exc_value, exc_tb)
        self.close()


if __name__ == '__main__':


    # Create the application.
    # sys.excepthook = catch_exceptions
    app = QApplication(sys.argv)

    # Create image viewer and load an image file to display.
    # viewer = Window(source_path="/Volumes/home/temp/pwd2022_test_select",
    #                 target_path="/Users/sober/Downloads/tp")


    oper_message = QMessageBox()
    oper_message.setWindowTitle('Help')
    oper_message.setIconPixmap(QPixmap(get_resource_path('keyboard.png')))
    # oper_message.setStandardButtons(QMessageBox.Ok)
    oper_message.setDefaultButton(QMessageBox.Ok)
    oper_message.show()
    oper_message.exec_()

    while True:
        # print('Select the path')
        path  = QFileDialog.getExistingDirectory()
        print('Select the path:', path)
        if path == '':
            press_button = QMessageBox().information(None, "Error", "Please select one folder!", QMessageBox.Yes |
                                           QMessageBox.No, QMessageBox.No)
            if press_button == QMessageBox.No:
                sys.exit(0)
        else:
            break



    window = Window(source_path=path, target_path="./")
    # window = Window(source_path=r'H:/tp/quality', target_path="./")

    window.setFixedSize(800, 800)
    # Show viewer and run application.
    window.show()


    sys.exit(app.exec_())

