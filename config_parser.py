import configparser
import os


class Config(object):
    """
    Class that represents parsed configuration file.

    The configuration file is an INI file. The file should/can
    define following values:

    ! - Marks required values in a given block.

    [train]
    learning_rate        = <learning rate that should be used for training>
    test_input_data_dir  = ! <path to directory containing test data>
    test_gt_data_dir     = ! <path to directory containing ground truth for test data>
    train_input_data_dir = ! <path to directory containing train data>
    train_gt_data_dir    = ! <path to directory containing ground truth for training data>

    [infer]
    input_data_dir = ! <path to directory containing table images>
    """

    def __init__(self, config_file_path):
        """
        The constructor of Config class.

        :param config_file_path: Path to INI configuration file
        """

        self.config_file_path = config_file_path

        # Train parameters
        self.learning_rate = None
        self.test_input_data_dir = None
        self.test_gt_data_dir = None
        self.train_input_data_dir = None
        self.train_gt_data_dir = None

        # Infer section
        self.input_data_dir = None

        self.parse_ini_config_file()

    def __str__(self):
        return f"<Config config_file_path={self.config_file_path} " \
               f"learning_rate={self.learning_rate} " \
               f"test_input_data_dir={self.test_input_data_dir} " \
               f"test_gt_data_dir={self.test_gt_data_dir} " \
               f"train_input_data_dir={self.train_input_data_dir} " \
               f"train_gt_data_dir={self.train_gt_data_dir} " \
               f"input_data_dir={self.input_data_dir}>"

    def parse_ini_config_file(self):
        """Function that parses the INI configuration file."""

        config_parser = configparser.ConfigParser()

        if os.path.exists(self.config_file_path):
            config_parser.read(self.config_file_path)
        else:
            raise Exception(f"ERROR: {self.config_file_path} does not exist!")

        if "train" in config_parser:
            self.parse_train_section(config_parser)

        if "infer" in config_parser:
            self.parse_infer_section(config_parser)

    def parse_infer_section(self, config_parser):
        """
        Function that parses the infer section in the INI configuration file.

        :type config_parser:  ConfigParser
        :param config_parser: ConfigParser class that contains the parsed INI
                              configuration file.
        """

        infer_config = config_parser["infer"]
        self.input_data_dir = Config.validate_dir(infer_config["input_data_dir"])

    def parse_train_section(self, config_parser):
        """
        Function that parses the train section in the INI configuration file.

        :type config_parser: ConfigParser
        :param config_parser: ConfigParser class that contains the parsed INI
                              configuration file.
        """

        train_config = config_parser["train"]
        self.learning_rate = Config.validate_int(train_config["learning_rate"], mandatory=False)
        self.test_input_data_dir = Config.validate_dir(train_config["test_input_data_dir"])
        self.test_gt_data_dir = Config.validate_dir(train_config["test_gt_data_dir"])
        self.train_input_data_dir = Config.validate_dir(train_config["train_input_data_dir"])
        self.train_gt_data_dir = Config.validate_dir(train_config["train_gt_data_dir"])

    @staticmethod
    def validate_dir(directory_path, mandatory=True):
        """
        A function that checks whether a given directory path exists.

        :type directory_path:  String
        :param directory_path: Directory path that should be checked
        :type mandatory:       Bool
        :param mandatory:      Specifies whether given directory path is mandatory.
        :return:               String that describes path to the directory if it exists.
        :raises Exception:     When directory path does not exist and :param mandatory is
                               set to True
        """
        if os.path.exists(directory_path):
            return directory_path
        elif mandatory:
            raise Exception(f"ERROR: {directory_path} does not exist!")
        else:
            return ""

    @staticmethod
    def validate_int(str_integer, mandatory=True):
        """
        A function that checks whether a given string is a valid integer

        :type str_integer:  Integer
        :param str_integer: String containing an integer.
        :type mandatory:    Bool
        :param mandatory:   Specifies whether it is mandatory that the :param str_integer
                            contains valid value.
        :return:            Integer value obtained by parsing the :param str_integer
        :raises Exception:  When the :param str_integer does not contain valid value and
                            :param mandatory is set to True
        """
        try:
            return int(str_integer)
        except Exception as e:
            if mandatory:
                raise e
            else:
                return None
