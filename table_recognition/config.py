import configparser
import logging
import os


class Config(object):
    """
    Class that represents parsed configuration file.

    The configuration file is an INI file. The file should/can
    define following values:

    ! - Marks required values in a given block.

    [data_preparation]
    ocr_output_path      = ! <directory containing XML files that represent the OCR output>
    dataset_img_path     = ! <directory containing table images>
    dataset_gt_path      = ! <directory containing GT data in XML CTDAR format>
    train_list           = <text file defining which files should be used for training>
    test_list            = <text file defining which files should be used for testing>
    randomize            = <RANDOM: randomly select which files will be used for training/testing>
    train_ratio          = <RANDOM: what percentage of the data should be used for training>
    test_ratio           = <RANDOM: what percentage of the data should be used for testing>
    visualize_graph      = <VISUALIZE: create visualization of the created graph and store it>
    visualize_dir        = <VISUALIZE: directory in which the visualizations should be stored>

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

        # Dataset preparation
        self.ocr_output_path = None
        self.dataset_img_path = None
        self.dataset_gt_path = None
        self.train_list = None
        self.test_list = None
        self.randomize = None
        self.train_ratio = None
        self.test_ratio = None
        self.visualize_graph = None
        self.visualize_dir = None
        self.prepared_data_dir = None
        self.input_graph_colorer = None
        self.visual_features = None

        # Train parameters
        self.wandb_mode = None
        self.learning_rate = None
        self.batch_size = None
        self.epochs = None
        self.test_input_data_dir = None
        self.test_gt_data_dir = None
        self.train_input_data_dir = None
        self.train_gt_data_dir = None
        self.data_dir = None
        self.model_path = None
        self.visualize_path = None
        self.preload_model = None
        self.train_percentage = None
        self.test_percentage = None

        # Infer section
        self.input_data_dir = None

        # Logger
        self.logger = None
        self.tqdm_disable = None

        self.parse_ini_config_file()

    def __str__(self):
        return f"<Config config_file_path={self.config_file_path} " \
               f"[train].learning_rate={self.learning_rate} " \
               f"[train].test_input_data_dir={self.test_input_data_dir} " \
               f"[train].test_gt_data_dir={self.test_gt_data_dir} " \
               f"[train].train_input_data_dir={self.train_input_data_dir} " \
               f"[train].train_gt_data_dir={self.train_gt_data_dir} " \
               f"[train].model_path={self.model_path} " \
               f"[train].train_percentage={self.train_percentage} " \
               f"[train].test_percentage={self.test_percentage} " \
               f"[train/infer].input_data_dir={self.input_data_dir} " \
               f"[dataset-preparation].ocr_output_path={self.ocr_output_path} " \
               f"[dataset-preparation].input_path={self.dataset_img_path} " \
               f"[dataset-preparation].output_path={self.dataset_gt_path} " \
               f"[dataset-preparation].train_list={self.train_list} " \
               f"[dataset-preparation].test_list={self.test_list} " \
               f"[dataset-preparation].randomize={self.randomize} " \
               f"[dataset-preparation].train_ratio={self.train_ratio} " \
               f"[dataset-preparation].test_ratio={self.test_ratio} " \
               f"[dataset-preparation].visualize_graph={self.visualize_graph} " \
               f"[dataset-preparation].visualize_dir={self.visualize_dir} " \
               f"[logging].logger={self.logger}>"

    def parse_ini_config_file(self):
        """Function that parses the INI configuration file."""

        config_parser = configparser.ConfigParser()

        if os.path.exists(self.config_file_path):
            config_parser.read(self.config_file_path)
        else:
            raise Exception(f"ERROR: {self.config_file_path} does not exist!")

        if "dataset-preparation":
            self.parse_dataset_preparation_section(config_parser)

        if "train" in config_parser:
            self.parse_train_section(config_parser)

        if "infer" in config_parser:
            self.parse_infer_section(config_parser)

        if "logging" in config_parser:
            self.parse_logging_section(config_parser)

    def parse_logging_section(self, config_parser):
        log_level = config_parser["logging"]["log_level"]
        self.tqdm_disable = Config.validate_bool(config_parser["logging"]["tqdm_disable"])

        log_level_dict = {"INFO": logging.INFO, "DEBUG": logging.DEBUG, "WARNING": logging.WARNING}
        logging.basicConfig(format='[%(levelname)s] %(asctime)s => %(message)s',
                            datefmt='(%Y-%m-%d %H:%M:%S)')
        self.logger = logging.getLogger("table_recognition")
        self.logger.setLevel(log_level_dict[log_level])

    def parse_dataset_preparation_section(self, config_parser):
        """
        Function that parser the dataset_preparation section in the INI
        configuration file.

        :type config_parser:  ConfigParser
        :param config_parser: ConfigParser class that contains the parsed INI
                              configuration file.
        """

        dataset_prep_config = config_parser["data_preparation"]
        self.ocr_output_path = Config.validate_file(dataset_prep_config["ocr_output_path"], mandatory=False)
        self.dataset_img_path = Config.validate_file(dataset_prep_config["dataset_img_path"], mandatory=False)
        self.dataset_gt_path = Config.validate_file(dataset_prep_config["dataset_gt_path"], mandatory=False)
        self.visualize_dir = Config.validate_file(dataset_prep_config["visualize_dir"], mandatory=False)
        self.prepared_data_dir = Config.validate_file(dataset_prep_config["prepared_data_dir"], mandatory=False)
        self.input_graph_colorer = dataset_prep_config["input_graph_colorer"]
        self.visual_features = Config.validate_bool(dataset_prep_config["visual_features"])

    def parse_infer_section(self, config_parser):
        """
        Function that parses the infer section in the INI configuration file.

        :type config_parser:  ConfigParser
        :param config_parser: ConfigParser class that contains the parsed INI
                              configuration file.
        """

        infer_config = config_parser["infer"]
        self.input_data_dir = Config.validate_file(infer_config["input_data_dir"])

    def parse_train_section(self, config_parser):
        """
        Function that parses the train section in the INI configuration file.

        :type config_parser: ConfigParser
        :param config_parser: ConfigParser class that contains the parsed INI
                              configuration file.
        """

        train_config = config_parser["train"]
        self.wandb_mode = train_config["wandb_mode"]
        self.learning_rate = Config.validate_float(train_config["learning_rate"], mandatory=False)
        self.batch_size = Config.validate_int(train_config["batch_size"], mandatory=False)
        self.epochs = Config.validate_int(train_config["epochs"], mandatory=False)
        self.model_path = train_config["model_path"]
        self.visualize_path = train_config["visualize_path"]
        self.data_dir = Config.validate_file(train_config["data_dir"], mandatory=False)
        self.preload_model = Config.validate_file(train_config["preload_model"], mandatory=False)
        self.train_percentage = Config.validate_float(train_config["train_percentage"], mandatory=False)
        self.test_percentage = Config.validate_float(train_config["test_percentage"], mandatory=False)
        self.model_name = train_config["model_name"]

    @staticmethod
    def validate_bool(bool_value, mandatory=True):
        """
        A function that checks whether a given string value contains
        a bool value (either True or False).

        :type bool_value:  String
        :param bool_value: String that should contain the bool value
        :type mandatory:   Bool
        :param mandatory:  Specifies whether given bool value is mandatory
        :raises Exception: When :param bool_value: does not contain valid bool value
                           and :param mandatory: is set to True.
        """
        if bool_value == "True":
            return True
        elif bool_value == "False":
            return False
        else:
            raise Exception(f"ERROR: {mandatory} is not valid bool value!")

    @staticmethod
    def validate_file(file_path, mandatory=True):
        """
        A function that checks whether a given directory path exists.

        :type file_path:   String
        :param file_path:  Directory path that should be checked
        :type mandatory:   Bool
        :param mandatory:  Specifies whether given directory path is mandatory.
        :return:           String that describes path to the directory if it exists.
        :raises Exception: When directory path does not exist and :param mandatory: is
                           set to True
        """
        if os.path.exists(file_path):
            return file_path
        elif mandatory:
            raise Exception(f"ERROR: {file_path} does not exist!")
        else:
            return ""

    @staticmethod
    def validate_float(str_float, mandatory=True):
        """
        A function that checks whether a given string is a valid integer

        :type str_float:    Integer
        :param str_float:   String containing an integer.
        :type mandatory:    Bool
        :param mandatory:   Specifies whether it is mandatory that the :param str_integer:
                            contains valid value.
        :return:            Integer value obtained by parsing the :param str_integer:
        :raises Exception:  When the :param str_integer: does not contain valid value and
                            :param mandatory: is set to True
        """
        try:
            return float(str_float)
        except Exception as e:
            if mandatory:
                raise e
            else:
                return None

    @staticmethod
    def validate_int(str_int, mandatory=True):
        try:
            return int(str_int)
        except Exception as e:
            if mandatory:
                raise e
            else:
                return None