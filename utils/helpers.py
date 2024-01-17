import os
import numpy as np
    

def create_dir(class_name):
    if not os.path.exists(class_name):
        os.makedirs(class_name)
