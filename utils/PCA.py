
import os  
import numpy as np  
import matplotlib.pyplot as plt  
from skimage.io import imread  
from Utils_pca import *


# Definición del directorio raíz del proyecto
root_dir = os.path.dirname(os.path.abspath(__file__))


metadata_dir = os.path.join(root_dir, "Metadata") 

metadata_paths = {
    "Specim_IQ": os.path.join(root_dir, "Metadata", "Metadata_Specim_IQ.hdr"),
    "EOS_M50": os.path.join(root_dir, "Metadata", "Metadata_EOS_M50.hdr"),
    "Ultris_SR5": os.path.join(root_dir, "Metadata", "Metadata_Ultris_SR5.hdr"),
    "Toucan": os.path.join(root_dir, "Metadata", "Metadata_Toucan.hdr")   
}


wavelengths = {camera: extract_wavelengths(path) for camera, path in metadata_paths.items()}

plot_pca_spectral_signatures("Specim_IQ", wavelengths, root_dir)
plot_pca_spectral_signatures("EOS_M50", wavelengths, root_dir)
plot_pca_spectral_signatures("Ultris_SR5", wavelengths, root_dir)
plot_pca_spectral_signatures("Toucan", wavelengths, root_dir)






