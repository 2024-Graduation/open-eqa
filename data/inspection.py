import argparse
import json

from collections import Counter
from pathlib import Path

ANSWER_CATEGORY = ["functional reasoning", "object state recognition", "spatial reasoning", "attribute recognition", 
                   "object localization", "object recognition", "world knowledge", "functional reasoning"]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=Path,
        default="data/open-eqa-v0.json",
        help="path to EQA dataset (default: data/open-eqa-v0.json)",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="hm3d",
        choices=["hm3d", "scannet", "all"],
        help="Type of dataset"
    )
    args = parser.parse_args()
    return args

# Load dataset
def load_hm3d(dataset_path):
    dataset = json.load(dataset_path.open("r"))
    hm3d_data = [item for item in dataset if item["episode_history"].startswith("hm3d")]
    print("found {:,} questions".format(len(hm3d_data)))
    return hm3d_data

def load_scannet(dataset_path):
    dataset = json.load(dataset_path.open("r"))
    scannet_data = [item for item in dataset if item["episode_history"].startswith("scannet")]
    print("found {:,} questions".format(len(scannet_data)))
    return scannet_data

def load_all(dataset_path):
    dataset = json.load(dataset_path.open("r"))
    print("found {:,} questions".format(len(dataset)))
    return dataset

""" Inpect a dataset
    1. Questions by Category
    2. Questions per Setting
    3. Do extra answers exist
    4. Which episodes were used to answer a question
""" 
class Inspection:
    def __init__(self, dataset):
        self.dataset = dataset

    def inspect(self):
        self.questions_by_category()
        self.questions_per_setting()
        self.do_extra_answers_exist()
        self.which_episodes_were_uesd()

    def questions_by_category(self):
        category_counts = Counter([item["category"] for item in self.dataset])

        print("-------- Questions by Category --------")
        for key, value in category_counts.items():
            print(f"{key}: {value}")
        print("---------------------------------------")

    def questions_per_setting(self):
        pass

    def do_extra_answers_exist(self):
        has_extra_answers = Counter("true" if "extra_answers" in item else "false" for item in self.dataset)

        print("-------- Do extra answers exist --------")
        for key, value in has_extra_answers.items():
            print(f"{key}: {value}")
        print("----------------------------------------")

    def which_episodes_were_uesd(self):
        used_episodes = Counter([item["episode_history"] for item in self.dataset])

        print("-------- Which episodes were used --------")
        for key, value in used_episodes.items():
            print(f"{key}: {value}")
        print("------------------------------------------")

def main(args: argparse.Namespace):
    if args.dataset == "hm3d":
        dataset = load_hm3d(args.dataset_path)
    elif args.dataset == "scannet":
        dataset = load_scannet(args.dataset_path)
    elif args.dataset == "all":
        dataset = load_all(args.dataset_path)

    print(dataset[0])
    inspection = Inspection(dataset)
    inspection.inspect()

if __name__ == "__main__":
    main(parse_args())