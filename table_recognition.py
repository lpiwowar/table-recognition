import argparse

from table_recognition.config import Config
from table_recognition.data_preparation import data_preparation
from table_recognition import Trainer


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

    #if args.data_preparation:
    #    data_preparation(config)
    #elif args.train:
    #    Trainer(config)

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

    from table_recognition.dataset import TableDataset
    from torchvision.ops import RoIAlign
    import cv2
    import torch
    import numpy as np

    dataset = TableDataset(config)
    data = dataset[0]
    img_name = data.img_path.split("/")[-1]
    for id, img in enumerate(data.edge_image_regions):
        h, w, c = img.shape
        roi_align = RoIAlign((10, 10), 1.0, -1)
        # new_img = roi_align(img, torch.tensor(torch.tensor([0, 0, 0, w, h])))
        #print(torch.tensor([[0, 0, 0, w, h]]).shape)
        coords = torch.cat((torch.Tensor([0]), torch.Tensor([0, 0, h, w]))).view(1, 5)
        #print(coords)
        #h, w, c = img.numpy()
        #print(img.shape)
        #img = torch.from_numpy(img.numpy()).permute(2, 1, 0)
        #print(img.shape)
        #exit(0)
        # img = torch.tensor([img])
        #img = img.numpy()
        #img = torch.from_numpy(np.array([img])).permute(3,2,0,1)
        #img = torch.tensor([img])
        # img = [img]
        #img = torch.tensor(img)

        cv2.imshow("test", img.numpy())
        cv2.waitKey(0)

        img = img.numpy().transpose(2,1,0)
        img = np.array([img])
        img = torch.from_numpy(img)
        print(img.shape)
        # exit(0)

        new_img = roi_align(img.type('torch.FloatTensor'), coords)
        print(new_img.shape)
        # new_img = roi_align(img, {[(0, 0, w-1, h-1)]}, 10)
        print(type(new_img.numpy().astype(int).dtype))

        new_img = new_img.numpy().transpose(3,2,1,0)[:,:,:,0]
        print(new_img.shape)
        # exit(0)
        print(new_img.astype(int).dtype)
        cv2.imshow("test", new_img.astype(np.uint8))
        cv2.waitKey(0)
        #cv2.imwrite("./cuts/" + str(id) + img_name, img.numpy())
