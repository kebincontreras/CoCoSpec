from enum import Enum


class CamerasEnum(str, Enum):
    EOS_M50 = "eos_m50"
    SPECIM_IQ = "specim_iq"
    TOUCAN = "toucan"
    ULTRIS_SR5 = "ultris_sr5"


class CocoaConditionsEnum(str, Enum):
    OPEN = "open"
    CLOSE = "close"


class ImageFormatsEnum(str, Enum):
    JPG = "jpg"
    NPY = "npy"
    NUMPY = "numpy"
    TIFF = "tiff"
    TIF = "tif"
    DAT = "dat"
    HDR = "hdr"
    ENVI = "envi"
