import argparse
import json
import os
import matplotlib.pyplot as plt

from collections import Counter
from pathlib import Path

ANSWER_CATEGORY = ["functional reasoning", "object state recognition", "spatial reasoning",
                   "attribute recognition", "object localization", "object recognition", "world knowledge"]

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

def make_autopct(values):
    value_iter = iter(values)
    def my_autopct(pct):
        value = next(value_iter)
        return f"{pct:.1f}%\n({value:d})"
    return my_autopct

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
        os.makedirs(os.path.join(os.getcwd(), "data", "results"), exist_ok=True)
        
    def draw_pie_chart(self, data, title, filename):
        plt.figure(figsize=(8, 5))
        plt.pie(data.values(), labels=data.keys(), autopct=make_autopct(data.values()), explode=[0.05]*len(data))
        plt.suptitle(title, fontweight='bold', fontsize=14)
        plt.savefig(os.path.join(os.getcwd(), "data", "results", filename))
        plt.close()

    def draw_bar(self, data, title, filename):
        data_counter = Counter(list(data.values()))

        plt.figure(figsize=(8, 5))
        bar = plt.bar(data_counter.keys(), data_counter.values(), color='blue', edgecolor='black', alpha=0.7)
        plt.suptitle(title, fontweight='bold', fontsize=14)
        plt.xticks(range(1, 15))
        plt.xlabel("Number of questions")

        # Annotate the counts above the bars
        for i, rect in enumerate(bar):
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2.0, height, str(int(height)), ha='center', va='bottom')

        plt.savefig(os.path.join(os.getcwd(), "data", "results", filename))
        plt.close()

    def inspect(self):
        self.overall_inspection()
        self.questions_by_category()
        self.do_extra_answers_exist()
        self.which_episodes_were_used()
        
    def overall_inspection(self):
        print("-------- Overall Inspection --------")
        print(f"Total number of questions: {len(self.dataset)}")
        print("------------------------------------")

    def questions_by_category(self):
        category_counts = Counter([item["category"] for item in self.dataset])
        
        # Draw a pie chart (Questions by Category)
        self.draw_pie_chart(category_counts, f"Questions by Category", "questions_by_category.svg")

    def do_extra_answers_exist(self):
        has_extra_answers = Counter("true" if "extra_answers" in item else "false" for item in self.dataset)

        # Draw a pie chart (Do extra answers exist)
        self.draw_pie_chart(has_extra_answers, f"Do extra answers exist", "extra_answers_exist.svg")

    def which_episodes_were_used(self):
        used_episodes = Counter([item["episode_history"] for item in self.dataset])

        # Draw a bar chart (The number of questions per episode)
        self.draw_bar(used_episodes, f"The number of questions per episode", "which_episodes_were_used.svg")

def main(args: argparse.Namespace):
    if args.dataset == "hm3d":
        dataset = load_hm3d(args.dataset_path)
    elif args.dataset == "scannet":
        dataset = load_scannet(args.dataset_path)
    elif args.dataset == "all":
        dataset = load_all(args.dataset_path)

    inspection = Inspection(dataset)
    inspection.inspect()

if __name__ == "__main__":
    main(parse_args())