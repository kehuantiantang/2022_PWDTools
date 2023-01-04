import os
import json
import os.path as osp
import sys


class ConfigUtil(object):
    DEFAULT_CONFIG = {'source_path': '', 'index': 0, 'target_path': '',
                      'total':0,
                      'name':'', 'assessment': {}}

    def __init__(self, config = None):
        self.update_widges = []
        if config is None:
            self.config = ConfigUtil.DEFAULT_CONFIG
        else:
            self.config = config
        self.get_record(config)

    def get_record(self, config):
        if osp.exists('gui_config.bk'):
            with open('gui_config.bk', 'r', encoding='utf-8') as f:
                old_config =  json.load(f)
                if config['total'] != old_config['total'] or config['source_path'] != old_config['source_path'] or \
                        config['target_path'] != old_config['target_path']:
                    pass
                else:
                    config['index'], config['name'], config['target_path'], config['assessment'] = old_config['index'], \
                                                                                     old_config['name'], \
                                                                     old_config['target_path'], old_config['assessment']
        self.config = config


    def __update_record(self):
        with open('gui_config.bk', 'w', encoding='utf-8') as f:
            json.dump({'source_path': self.config['source_path'], 'index': self.config['index'], 'target_path':
                self.config['target_path'],
                       'total':self.config['total'],
                       'name':self.config['name'], 'assessment':self.config['assessment']}, f, indent=6)


    # def add_update_widgets(self, widget, keys):

    def add_update_widget(self, widget, keys):
        self.update_widges.append((widget, keys))
        self.__update_widget()

    def __update_widget(self):
        for widget, keys in self.update_widges:
            params = {}
            for k in keys:
                params[k] = self.config[k]
            widget.update_widget(**params)

    def __update_config(self, item = None):
        if item is not None:
            for k, v in item.items():
                if k == 'assessment':
                    self.config['assessment'][v[0]] = v[1]
                else:
                    self.config[k] = v


    def update(self, item = None):
        self.__update_config(item)
        self.__update_widget()
        self.__update_record()


def get_resource_path(relative_path):
    if getattr(sys, 'frozen', False): #是否Bundle Resource
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)