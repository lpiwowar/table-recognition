import argparse

from table_recognition.config import Config
from table_recognition.data_preparation import data_preparation
from table_recognition.train import train_pipeline


def check_arguments(arg):
    """
    A function that checks whether valid combination of
    arguments is used.

    :type arg:  ArgumentParser
    :param arg: Parsed arguments that should be checked.
    :return:     True when valid combination of arguments is used
                 False otherwise
    """
    return not (not arg.train ^ arg.infer) ^ arg.data_preparation


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Table recognition")
    parser.add_argument("--train",
                        help="Train model for table recognition (--config-file required)",
                        action="store_true")
    parser.add_argument("--infer",
                        help="Use trained model for table recognition (--config-file required)",
                        action="store_true")
    parser.add_argument("--data-preparation",
                        help="Prepare dataset for training (--config-file required)",
                        action="store_true")
    parser.add_argument("--config-file",
                        help="Path to configuration file",
                        default="./config.ini")
    args = parser.parse_args()

    if not check_arguments(args):
        raise Exception("ERROR: Either --train, --infer or --data-preparation must be specified.")

    config = Config(args.config_file)

    if args.data_preparation:
        data_preparation(config)
    elif args.train:
        train_pipeline(config)

    # import wandb
    # import datetime
    # import os

    # run = wandb.init(project="table-recognition",
    #                 name=datetime.datetime.now().strftime("%Y-%m-%d-%H:%M"),
    #                 entity="lpiwowar")

    # artifact = wandb.Artifact('ocr-output', type='dataset')
    # dir_path = "/home/lpiwowar-personal/PycharmProjects/master-thesis/data_preparation/ocr_output"
    # for file in os.listdir(dir_path):
    #     artifact.add_file(os.path.join(dir_path, file))

    # run.log_artifact(artifact)
