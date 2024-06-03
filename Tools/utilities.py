import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import pandas as pd

def tensor_has_nan(some_tensor):
        """
        check if the tensor has nans
        """

        
        return not tf.reduce_all(tf.math.is_finite(some_tensor))


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
         
def save_predictions(predictions,save_path,save_as):
        """
                make dataframe and save predictions to csv file
        """
        
        
        save_path = Path(save_path)
        
        predictions = pd.DataFrame(predictions)
        try:
                print(f"[INFO] Saving predictions at {save_path}")
                save_path.mkdir(parents=True, exist_ok=True)
                save_path = save_path / f"{save_as}"
                assert save_path.suffix == '.csv', "file name must have .csv extension"
                save_path = str(save_path)

                predictions.to_csv(save_path)

        except Exception as e:
                print(f"[ERROR] couldn't save {save_path}")
        

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
                        

def plot_showcase_and_save_backbone_results( all_data_to_plot,
                                        _instances,
                                        save_path = "resources/backbone-showcase",
                                        dataset_name = "SkyData",
                                        loss_functions_used = "mse_pixel"
                                        ):

    

    # input_images = all_data_to_plot[0]
    template_images = all_data_to_plot[1]
    warped_inputs = all_data_to_plot[2]
    # rgb_fmaps = all_data_to_plot[4]
    ir_fmaps = all_data_to_plot[5]
    warped_fmaps = all_data_to_plot[6]

    for i_th in range(len(all_data_to_plot[0])):
        data_to_plot = {k:np.array(v[i_th]).clip(0,1) for k,v in all_data_to_plot.items()}
        
        summed_data = {
                3: 0.9 * warped_inputs[i_th] + .2 * warped_fmaps[i_th],
                7: 0.05 *template_images[i_th] + 1 *  tf.cast(ir_fmaps[i_th],tf.float32)
        }
        data_to_plot.update(summed_data)
        
        fig, axs = plt.subplots(2, 4, figsize=(8, 5), constrained_layout=True)
        axs = axs.ravel()

        # fig = plt.figure(figsize=(20, 20))
        # axs = fig.subplots(3, 2)
        titles=["input_images","template_images","warped_inputs","summed_rgb","rgb_fmaps","ir_fmaps","warped_fmaps","summed_ir"]
        for i, ax in enumerate(axs):
            ax.axis('off')
            ax.set_title(titles[i])
            


        for i, data_i in data_to_plot.items():
            axs[i].imshow(np.array(data_i).clip(0,1))
        # fig.tight_layout()
        # plt.show()

        #saving showcase
        
        save_path = Path(f"{save_path}/{dataset_name}/{loss_functions_used}")
        save_name = f"{_instances.numpy()[i_th].decode('utf-8')}.png"
        save_path.mkdir(parents=True, exist_ok=True)
        
        save_as= str(save_path/save_name)
        
        plt.savefig(save_as, dpi=300, bbox_inches='tight')
        plt.close()
            
