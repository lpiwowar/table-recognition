import argparse

from config_parser import Config

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Table recognition")
    parser.add_argument("--train",
                        help="Train model for table recognition (--config_file required)",
                        action="store_true")
    parser.add_argument("--infer",
                        help="Use trained model for table recognition (--config_file required)",
                        action="store_true")
    parser.add_argument("--config_file",
                        help="Path to configuration file",
                        default="./table_recognition_config.ini")
    args = parser.parse_args()

    if not args.train ^ args.infer:
        raise Exception("ERROR: Either --train or --infer must be specified.")

    config = Config(args.config_file)
