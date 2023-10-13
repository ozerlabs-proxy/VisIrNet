import json
import numpy as np
from pathlib import Path
import tensorflow as tf
from pathlib import Path

def tensor_has_nan(some_tensor):
        """
        check if the tensor has nans
        """

        
        return not np.isfinite(some_tensor).all()


# dumping json needs to convert numpy types to python types
def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()

# save coco format to json file 
def save_json(data, 
                save_dir:str,
                file_name:str):
        """
        Save the json file
        """

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / file_name
        
        assert save_path.suffix == '.json', "file name must have .json extension"
        
        print("[INFO] saving json file to {}".format(save_path))
        try:
                with open(save_path, 'w') as fp:
                        json.dump(data, fp,default=np_encoder)
        except Exception as e:
                print("[ERROR] error saving json file...")
                print(e)
                return
        print(f"[INFO] file saved at {save_dir}.")
                

def save_model_weights(model, 
                        save_path, 
                        save_as,
                        tag = None,
                        ):
        """
        save model weights
        """
        tag = "" if tag is None else tag
        try:
                print(f"[INFO] Saving model at {save_path}")
                save_path = Path(save_path)
                save_path.mkdir(parents=True, exist_ok=True)
                save_path = str(save_path / f"{save_as}-{tag}.keras")
                model.save(save_path)
        except Exception as e:
                print(f"[ERROR] model couldn't be saved at {save_path}")
                
def save_logs(logs,save_path,save_as):
        """
                save logs to json file given the logs (metrics) dictionary
        """
        save_path = Path(save_path)
        try:
                print(f"[INFO] Saving logs at {save_path}")
                save_path.mkdir(parents=True, exist_ok=True)
                save_path = save_path / f"{save_as}"
                assert save_path.suffix == '.json', "file name must have .json extension"
                save_path = str(save_path)
                with open(save_path, 'w') as fp:
                        json.dump(logs, fp,default=np_encoder)
        except Exception as e:
                print(f"[ERROR] model couldn't be saved at {save_path}")
        

# get latest file from a directory
def latest_file(path: Path, 
                pattern: str = "*"):
        """
        Given a path and a pattern return the latest file in the directory
        """

        files = list(path.glob(pattern))
        assert any(files), f"could not find any file with pattern {pattern} in {path}"
        return max(files, key=lambda x: Path(x).stat().st_ctime)

def load_model(model_path:Path = None):
        """
        Given a path load the model
        Args:
                mode_path (str): path to the model
        Returns:
                model (tf.keras.Model): the loaded model
        """
        try:
                print(f"[INFO] loading model from {model_path}")
                model = tf.keras.models.load_model(model_path)
        except:
                print(f"[ERROR] model not found at {model_path}")
                raise Exception("Model not found")
        return model

def get_summary_writter(log_dir : str = "logs/tensorboard",
                        log_id : str = "",
                        suffix : str = ""):
        
        """
        Given a log directory return a summary writer
        Args:
                log_dir (str): log directory
                
        Returns:
                _summary_writer (tf.summary.SummaryWriter): summary writer
        """
        log_dir = log_dir + "/" + log_id + "/" + suffix
        _summary_writer = tf.summary.create_file_writer(log_dir)
        return _summary_writer

def tb_write_summary(_summary_writer,
                        logs,
                        epoch):
        """
        Given a summary writer and logs write the logs to tensorboard
        Args:
                _summary_writer (tf.summary.SummaryWriter): summary writer
                logs (dict): logs to write
                epoch (int): epoch number
        Returns:
                None
        """
        with _summary_writer.as_default():
                for key, value in logs.items():
                        # print(f"key: {key}, value: {value}")
                        tf.summary.scalar(f"{key}", value, epoch)
                        _summary_writer.flush()