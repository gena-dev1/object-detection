import os
import argparse
import torch


parser = argparse.ArgumentParser(description="Folder To Parse")
parser.add_argument("--path", required=True, default=None, type=str)
args = parser.parse_args()

model = torch.hub.load("ultralytics/yolov5", "yolov5l")


def list_directory(path):
    origin_folder = os.path.abspath(path)
    dir_list = os.listdir(path)
    folder = []

    for dir in dir_list:
        path = os.path.join(origin_folder, dir)
        folder.append(path)

    return folder


def create_manifest(folder):
    for img in folder:
        results = model(img)
        result_info = results.pandas().xyxy[0]["name"]
        try:
            result = img + "  " + result_info[0]
            with open("object.txt", 'a') as f:
                f.write(result)

        except:
            pass


def main():
    directory = args.path
    listed_dir = list_directory(directory)
    create_manifest(listed_dir)


if __name__ == "__main__":
    main()
