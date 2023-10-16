"""
will contan some important utility functions for the project
"""

import tensorflow as tf
from pathlib import Path


def plot_and_save_model_structure(model,
                                    save_path="resources/",
                                    save_as=f"model"):
    """
    Plot and save the model structure
    """
    model_name = save_as if not model.name else f'{save_as}-{model.name}'
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    save_path = save_path / f"{model_name }.png"
    save_as = str(save_path)
    print(f"[INFO] Saving model structure to {save_as}")
    tf.keras.utils.plot_model(model,
                            to_file=save_as,
                            show_shapes=False,
                            show_dtype=False,
                            show_layer_names=True,
                            rankdir="TB",
                            expand_nested=False,
                            dpi=96,
                            layer_range=None,
                            show_layer_activations=False,
                            show_trainable=False,
                        )
    print(f"[INFO] Model structure saved to {save_as}")
    

def save_model(model: tf.keras.Model,
                target_dir: str,
                model_name: str):
    """Saves a tf model to a target directory.

    Args:
    model: A target tf model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model,
               target_dir="models",
               model_name=".pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,exist_ok=True)

    # Create model save path
    assert model_name.endswith(".h5") or model_name.endswith(".h5"), "model_name should end with '.h5' "
    model_save_path = str(target_dir_path / model_name)

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    try:
        model.save(model_save_path)
        print(f"[INFO] Model saved to: {model_save_path}")
    except Exception as e:
        print(f"[ERROR] Error while saving model to {model_save_path}")
    