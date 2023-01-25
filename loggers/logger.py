import logging
import os

class AutoMLLog:

    def __init__(self, name):
        self.log_obj = logging.getLogger(name)
        self.fileHandler = None
        self.consoleHandler = None

    def set_handler(self, log_name):
        self.log_obj.setLevel(logging.INFO)
        file_path  = f'./logs/{log_name}.log'

        fileHandler = logging.FileHandler(file_path, mode = "w",  encoding='utf-8')
        fileHandler.setLevel(logging.INFO)
        self.fileHandler = fileHandler

        consoleHandler = logging.StreamHandler()
        consoleHandler.setLevel(logging.INFO)
        self.consoleHandler = consoleHandler


    def set_formats(self):
        formatter = logging.Formatter('%(asctime)s %(name)s %(filename)s %(module)s %(funcName)s %(levelname)s: %(message)s')
        self.fileHandler.setFormatter(formatter)


    def addOn(self):
        self.log_obj.addHandler(self.fileHandler)
        self.log_obj.addHandler(self.consoleHandler)

        return self.log_obj