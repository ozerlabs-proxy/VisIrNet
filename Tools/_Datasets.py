import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import json
from pathlib import Path

class Dataset():
    def __init__(self, dataset, split = "train"):
        
        self.set_paths(dataset = dataset, split=split)
        print(f"[INFO] loading {split} dataset")
        self.dataset = tf.data.Dataset.list_files(str(self.INPUTS_DIR / "*.*"))
        print(f"[INFO] {split} _dataset: ", len(self.dataset))
        self.dataset = self.dataset.map(lambda x: tf.py_function(self._preprocess_instance, 
                                                        [x], [tf.float32, tf.float32, tf.float32, tf.string]))
        
        
    def set_paths(self, dataset, split):
        
        """
        Set the paths to the dataset
        """
        DATA_DIR = Path('data')
        DATASET_DIR = DATA_DIR / dataset


        assert DATASET_DIR.exists(), f"dataset {dataset} not found"
        assert split in ["train", "val", "test"], f"split {split} not found"


        input_images_dir = f"{split}_input"
        template_images_dir=f"{split}_template"
        labels_dir=f"{split}_label"

        self.INPUTS_DIR = DATASET_DIR / input_images_dir
        self.TEMPLATES_DIR = DATASET_DIR / template_images_dir
        self.LABELS_DIR = DATASET_DIR / labels_dir        
        
        
    def _get_image(self,image_path):
        """
        Given a path to an image, return the image as a numpy array
        Args:
            image_path (str): path to the image
        Returns:
            np.array: image as a numpy array
        """
        image = PIL.Image.open(image_path)
        image = np.array(image)       
        
        if len(image.shape) < 2:
            raise ValueError("image has less than two dimensions")

        elif len(image.shape) == 2:
            # print("image is grayscale")
            # convert grayscale to RGB
            image = PIL.Image.fromarray(image)
            image = image.convert('RGB')
            # image = tf.image.grayscale_to_rgb(image)
            image = np.array(image)
            # print("new image.shape: ", image.shape)
            
        elif image.shape[2] > 3:
            # print("image is RGBA")
            # convert RGBA to RGB
            image = PIL.Image.fromarray(image, 'RGBA').convert('RGB')
            image = np.array(image)
            
            # print("new image.shape: ", image.shape)

        # assert len(image.shape) == 3, "image is not RGB"
        # assert image.shape[2] == 3, "image is not RGB"
        
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image = image / np.float32(255.0)
        
        # assert len(image.shape) == 3, f"image shape is not (H,W,C) but {image.shape}"
        
        return image

    def _parse_label(self,label_path):
        """
        Given a path to a label, return the label as a numpy array
        Args:
            label_path (str): path to the label
        Returns:
            np.array: label as a numpy array
        """
        
        with open(str(label_path), 'r') as input_file:
            data = json.load(input_file)
            
        location = data['location']
        """" >>> location
            top_left_u"
            "top_left_v"
            "top_right_u"
            "top_right_v"
            "bottom_left_u"
            "bottom_left_v"
            "bottom_right_u"
            "bottom_right_v"
        """

        u_list=[list(x.values())[0] for x in location]
        v_list=[list(x.values())[1] for x in location]
        
        u_list = tf.constant(u_list, dtype="float")
        v_list = tf.constant(v_list, dtype ="float")
                
        label = tf. convert_to_tensor(tf.concat([u_list, v_list], -1), dtype=tf.float32)
        # assert len(label.shape) == 1 and label.shape == 8 , f"label shape is not (uv_list)" 
        
        return label

    def _preprocess_instance(self,_instance):
        """
        Given a _instance, return a tuple of (input, template, label , _instance)
        Args:
            _instance (_str_): pa
        Returns:
            tuple: (input, template, label , _instance)
        """
        
        # assuming _instance is a string with the format "00001"
        # 
        # _instance = tf.strings.as_string(_instance).numpy().decode("utf-8")
        
        
        _instance_path = Path(tf.strings.as_string(_instance).numpy().decode("utf-8"))
        _instance = _instance_path.stem
        _extension = _instance_path.suffix
        input_path = self.INPUTS_DIR / f"{_instance}{_extension}"
        template_path = self.TEMPLATES_DIR / f"{_instance}{_extension}"
        label_path = self.LABELS_DIR / f"{_instance}_label.txt"
        
        input_image = self._get_image(input_path)
        template_image = self._get_image(template_path)
        label = self._parse_label(label_path)
        

        
        
        # 
        # assert input_image.shape == (192,192,3), f"input_image.shape is not (192,192,3) but {input_image.shape}"
        # assert template_image.shape == (128,128,3), f"template_image.shape is not (192,192,3) but {template_image.shape}"
        # assert label.shape == (8,), f"label.shape is not (8,) but {label.shape}"
        # assert _instance == _instance_path.stem, f"_instance is not {_instance_path.stem} but {_instance}"
        
        # # assert all are contain finate values
        # assert tf.reduce_all(tf.math.is_finite(input_image)) , f"input_image contain non finite values ----- {tf.strings.as_string(str(_instance))}"
        # assert tf.reduce_all(tf.math.is_finite(template_image)) , f"template_image contain non finite values ----- {tf.strings.as_string(str(_instance))}"
        # assert tf.reduce_all(tf.math.is_finite(label)) , f"label contain non finite values {label} ----- {tf.strings.as_string(str(_instance))}"
        _instance = tf.constant(str(_instance)+"" , dtype=tf.string)
        
        return tf.cast(input_image,dtype="float"), tf.cast(template_image,dtype="float"), tf.cast(label, dtype="float") , _instance