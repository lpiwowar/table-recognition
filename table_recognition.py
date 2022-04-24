import argparse

from table_recognition.config import Config
from table_recognition.data_preparation import data_preparation
from table_recognition import Trainer
from table_recognition import Infer
from table_recognition import evaluate


def check_arguments(arg):
    """
    A function that checks whether valid combination of
    arguments is used.

    :type arg:  ArgumentParser
    :param arg: Parsed arguments that should be checked.
    :return:     True when valid combination of arguments is used
                 False otherwise
    """
    return (not (not arg.train ^ arg.infer) ^ arg.data_preparation) ^ args.evaluate


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
    parser.add_argument("--evaluate",
                        help="Evaluate performance of the model",
                        action="store_true")
    parser.add_argument("--config-file",
                        help="Path to configuration file",
                        default="./config.ini")
    args = parser.parse_args()

    if not check_arguments(args):
        raise Exception("ERROR: Either --train, --infer or --data-preparation must be specified.")

    if args.data_preparation:
        config = Config(args.config_file, task="data-preparation")
        data_preparation(config)
    elif args.train:
        config = Config(args.config_file, task="train")
        Trainer(config)
    elif args.infer:
        config = Config(args.config_file, task="infer")
        Infer(config)
    elif args.evaluate:
        config = Config(args.config_file, task="evaluate")
        evaluate(config)
