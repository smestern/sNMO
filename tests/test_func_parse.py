from sNMO.b2_model.brian2_model import brian2_model, genModel
import logging
#set up logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
#test loading a file
model = optModel(file='C:\\Users\\SMest\\Dropbox\\PVN_MODELLING_WORK\\CADEX_MODEL\\CADEX_SPATIAL copy.py')