"""Contains auxiliary methods for the application."""
import logging
import os
import pathlib
import time

# mute tf logs below warning level and import it
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf  # isort:skip  # noqa: E402


def init_logger(log_dir: pathlib.Path, run_time: str = None) -> None:
    """Initialise logger."""
    # create log filename path
    if run_time is None:
        run_time = time.strftime("%d-%m-%Y_%H-%M-%S", time.localtime())
    log_path = log_dir.joinpath(f"{run_time}.log")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if os.path.exists(log_path):
        os.remove(log_path)

    # set logging file handler
    fh = logging.FileHandler(log_path)
    formatter = logging.Formatter(
        fmt="[%(asctime)-19s] [%(levelname)s] - in: %(name)s.%(funcName)s "
        'line: %(lineno)d, msg: "%(message)s"',
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(formatter)

    # set logging to console
    ch = logging.StreamHandler()

    # init logging basic config, which is available for all scripts
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)-8s] [%(levelname)s] msg: "%(message)s"',
        datefmt="%H:%M:%S",
        handlers=[ch, fh],
    )


def prepare_gpu() -> str:
    """Set up hardware settings for tensorflow."""
    avail_num_gpus = len(tf.config.list_physical_devices("GPU"))
    if avail_num_gpus > 0:
        gpus = tf.config.experimental.list_physical_devices("GPU")
        GPU_ID = 0
        tf.config.experimental.set_visible_devices(gpus[GPU_ID], "GPU")
        tf.config.experimental.set_memory_growth(gpus[GPU_ID], True)
        selected_hardware = "I'm using GPU"
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        selected_hardware = "I'm using CPU"

    return f"Available number of GPUs is {avail_num_gpus}, {selected_hardware}"
