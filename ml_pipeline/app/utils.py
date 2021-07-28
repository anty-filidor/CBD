"""Contains auxiliary methods for the application."""
import logging
import os
import pathlib
import time


def init_logger(log_dir: pathlib.Path) -> None:
    """Initialise logger."""
    run_time = time.strftime("%d-%m-%Y_%H-%M-%S", time.localtime())

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir.joinpath(f"CBD_{run_time}.log")

    # set logging file handler
    fh = logging.FileHandler(log_path)
    formatter = logging.Formatter(
        fmt="[CBD APP] [%(asctime)-19s] [%(levelname)s] - in: %(name)s.%(funcName)s "
        'line: %(lineno)d, msg: "%(message)s"',
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(formatter)

    # set logging to console
    ch = logging.StreamHandler()

    # init logging basic config, which is available for all scripts
    logging.basicConfig(
        level=logging.INFO,
        format='[CBD APP] [%(asctime)-8s] [%(levelname)s] msg: "%(message)s"',
        datefmt="%H:%M:%S",
        handlers=[ch, fh],
    )


def prepare_gpu() -> str:
    """Set up hardware settings for tensorflow."""
    import tensorflow as tf

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
