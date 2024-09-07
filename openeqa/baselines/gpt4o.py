# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import List, Optional
import numpy as np
import tqdm

from openeqa.utils.openai_utils import (
    call_openai_api,
    call_openai_assitant_api,
    prepare_openai_vision_messages,
    set_openai_key,
)
from openeqa.utils.prompt_utils import load_prompt
from openeqa.utils.scenegraph_utils import ScenegraphManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default="data/open-eqa-mini-hm3d-v0.json",
        help="path to EQA dataset (default: data/open-eqa-mini-hm3d-v0.json)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model (default: gpt-4o)",
    )
    parser.add_argument(
        "--frames-directory",
        type=Path,
        default="data/frames/",
        help="path image frames (default: data/frames/)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=50,
        help="num frames in gpt4o (default: 50)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="image size (default: 512)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="gpt seed (default: 1234)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="gpt temperature (default: 0.2)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="gpt maximum tokens (default: 128)",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default="data/results",
        help="output directory (default: data/results)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="continue running on API errors (default: false)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="only process the first 5 questions",
    )
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    args.output_path = args.output_directory / (
        args.dataset.stem + "-{}-{}.json".format(args.model, args.seed)
    )
    return args

def ask_question( #* called when there is no scenegraph for the episode of question
    question: str,
    image_paths: List,
    image_size: int = 512,
    openai_key: Optional[str] = None,
    openai_model: str = "gpt-4o",
    openai_seed: int = 1234,
    openai_max_tokens: int = 128,
    openai_temperature: float = 0.2,
    force: bool = False,
) -> Optional[str]:
    try:
        set_openai_key(key=openai_key)

        prompt = load_prompt("gpt4o-base-sg") # TODO: load the prompt for creating scenegraph
        prefix, suffix = prompt.split("User Query:")
        suffix = "User Query:" + suffix.format(question=question)

        messages = prepare_openai_vision_messages(
            prefix=prefix, suffix=suffix, image_paths=image_paths, image_size=image_size
        )
        output = call_openai_api(
            messages=messages,
            model=openai_model,
            seed=openai_seed,
            max_tokens=openai_max_tokens,
            temperature=openai_temperature,
        )
        return output
    except Exception as e:
        if not force:
            traceback.print_exc()
            raise e


def main(args: argparse.Namespace):
    # check for openai api key
    assert "OPENAI_API_KEY" in os.environ

    # load dataset
    dataset = json.load(args.dataset.open("r"))
    print("found {:,} questions".format(len(dataset)))

    # load results
    results = []
    if args.output_path.exists():
        results = json.load(args.output_path.open())
        print("found {:,} existing results".format(len(results)))
    completed = [item["question_id"] for item in results]
    
    scenegraph_manager = ScenegraphManager()
    
    # process data
    for idx, item in enumerate(tqdm.tqdm(dataset)): #* 각 question에 대해 반복
        if args.dry_run and idx >= 2:
            break

        # skip completed questions
        question_id = item["question_id"]
        if question_id in completed:
            continue  # skip existing

        #* extract scene paths
        # TODO: call function to get indices - after merge other branch
        episode_id = item["episode_history"]
        folder = args.frames_directory / episode_id
        frames = sorted(folder.glob("*-rgb.png"))
        indices = np.round(np.linspace(0, len(frames) - 1, args.num_frames)).astype(int)
        paths = [str(frames[i]) for i in indices]

        # TODO: check whether all frames are seen - after merge other branch
        #* check for existence of the episode's scenegraph
        is_there = scenegraph_manager.has_episode(episode_id)

        # generate answer
        question = item["question"]

        if is_there: #* if there is a scenegraph for the episode
            print("Scenegraph for episode: {}".format(episode_id))
            output = ask_question(
                question=question,
                image_paths=paths,
                image_size=args.image_size,
                openai_model=args.model,
                openai_seed=args.seed,
                openai_max_tokens=args.max_tokens,
                openai_temperature=args.temperature,
                force=args.force,
            )
        else: #* if there is no scenegraph for the episode
            print("No scenegraph for episode: {}".format(episode_id))
            output = ask_question(
                question=question,
                image_paths=paths,
                image_size=args.image_size,
                openai_model=args.model,
                openai_seed=args.seed,
                openai_max_tokens=args.max_tokens,
                openai_temperature=args.temperature,
                force=args.force,
            )
        
        #* save updated_scenegraph
        print("output: {}".format(output))
        # TODO: string to dict - after engineering prompt
        SCENEGRAPH_SEPARATOR = "User Query:"
        # output = json.loads(output)
        prefix, suffix = output.split(SCENEGRAPH_SEPARATOR)
        print("suffix: ", suffix)
        # scenegraph_manager.update_scenegraph(episode_id, json.loads(suffix))
        # answer = output["answer"]
        answer = output
    
        # store results
        results.append({"question_id": question_id, "answer": answer})
        json.dump(results, args.output_path.open("w"), indent=2)

    # save at end (redundant)
    json.dump(results, args.output_path.open("w"), indent=2)
    print("saving {:,} answers".format(len(results)))


if __name__ == "__main__":
    main(parse_args())