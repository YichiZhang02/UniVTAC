"""Registry of tactile encoder pretraining objectives."""
from ResNet.model import ResNet
from RVAE.model import RVAE
from RAE.model import RAE
from MAE.model import MAE
from DINOv2.model import DINOv2
from importlib import import_module
SDVAE = import_module('SD-VAE.model').SDVAE
IJEPA = import_module('I-JEPA.model').IJEPA
VJEPA = import_module('V-JEPA.model').VJEPA
METHODS = {'ResNet': ResNet, 'RVAE': RVAE, 'RAE': RAE, 'SD-VAE': SDVAE, 'MAE': MAE, 'DINOv2': DINOv2,
           'I-JEPA': IJEPA, 'V-JEPA': VJEPA}

from encoder.catalog import MULTIFRAME_METHODS
for _name in MULTIFRAME_METHODS:
    METHODS[_name] = import_module(f'encoder.{_name}.model').Model
