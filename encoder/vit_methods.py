"""Registry of tactile encoder pretraining objectives."""
from ResNet.model import ResNet
from VAE.model import VAE
from MAE.model import MAE
from DINOv2.model import DINOv2
from importlib import import_module
IJEPA = import_module('I-JEPA.model').IJEPA
VJEPA = import_module('V-JEPA.model').VJEPA
METHODS = {'ResNet': ResNet, 'VAE': VAE, 'MAE': MAE, 'DINOv2': DINOv2,
           'I-JEPA': IJEPA, 'V-JEPA': VJEPA}
