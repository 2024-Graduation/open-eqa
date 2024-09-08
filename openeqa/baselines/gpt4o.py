# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import random
import traceback
from pathlib import Path
from typing import List, Optional
import numpy as np
import tqdm

from openeqa.utils.extract_indices import IndicesExtractor
from openeqa.utils.openai_utils import (
    call_openai_api,
    prepare_openai_vision_messages,
    set_openai_key,
)
from openeqa.utils.prompt_utils import load_prompt
from openeqa.utils.scenegraph_utils import ScenegraphManager

DEFAULT_MAX_TOKENS = 4096 #* maximum tokens for gpt4o
INPUT_SCENEGRAPH_SEPARATOR = "Output: " #* input scenegraph separator
OUTPUT_SCENEGRAPH_SEPARATOR = "Output: "

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
        default=DEFAULT_MAX_TOKENS,
        help="gpt maximum tokens (default: DEFAULT_MAX_TOKENS)",
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
    parser.add_argument( #* test base prompt
        "--test-base-prompt",
        action="store_true",
        help="test prompt performance",
    )
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    args.output_path = args.output_directory / (
        args.dataset.stem + "-{}-{}.json".format(args.model, args.seed)
    )
    return args

def ask_question(
    question: str,
    image_paths: List,
    previous_scenegraph: json,
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

        prompt = load_prompt("gpt4o-update")
        prefix, suffix = prompt.rsplit(INPUT_SCENEGRAPH_SEPARATOR, 1)
        suffix_scenegraph, suffix_query = suffix.split("User Query:")
        suffix = "Scenegraph: " + suffix_scenegraph.format(scenegraph=json.dumps(previous_scenegraph)) + "\nUser Query: " + suffix_query.format(question=question)

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

def extract_frames(
    episode_id: str,
    num_frames: int,
    frames_directory: Path,
    extractor: IndicesExtractor,
) -> List[str]:
    folder = frames_directory / episode_id
    frames = sorted(folder.glob("*-rgb.png"))
    extractor.add_episode(episode_id, len(frames))
    extract_indices, indices_status = extractor.extract_indices(episode_id, num_frames)

    paths = [str(frames[i]) for i in extract_indices]
    return paths

def create_base_scenegraph(
    episode_id: str,
    image_paths: List[str],
    scenegraph_manager: ScenegraphManager,
    image_size: int = 512,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    force: bool = False,
) -> Optional[str]:
    try:
        set_openai_key(key=None)
    
        prompt = load_prompt("gpt4o-base")
        print("Creating base scenegraph for episode: {}".format(episode_id))
        messages = prepare_openai_vision_messages(
            prefix=prompt, image_paths=image_paths, image_size=image_size
        )
        output = call_openai_api(
            messages=messages,
            model="gpt-4o",
            seed=1234,
            max_tokens=max_tokens,
            temperature=0.2,
        )
        scenegraph = output.split("Scenegraph:")
        scenegraph = json.loads(scenegraph[1])
        scenegraph_manager.create_scenegraph(episode_id, scenegraph)

        return scenegraph

    except Exception as e:
        if not force:
            traceback.print_exc()
            raise e

def main(args: argparse.Namespace):
    # check for openai api key
    assert "OPENAI_API_KEY" in os.environ
    
    #* test base prompt
    if args.test_base_prompt:
        dataset = Path("data/mini-hm3d-episodes/")
        questions = []
        for episode in os.listdir(dataset):
            # 한 episode 당 하나의 question만 추출. 랜덤
            with open(dataset / episode, "r") as file:
                data = json.load(file)
                random.shuffle(data)
                questions.append(data[0])

    else:
        # load questions dataset
        questions = json.load(args.dataset.open("r"))
    
    print("found {:,} questions".format(len(questions)))

    # load results
    results = []
    if args.output_path.exists():
        results = json.load(args.output_path.open())
        print("found {:,} existing results".format(len(results)))
    completed = [item["question_id"] for item in results]
    
    scenegraph_manager = ScenegraphManager()
    extractor = IndicesExtractor()
    
    # process data
    for idx, item in enumerate(tqdm.tqdm(questions)): #* 각 question에 대해 반복
        if args.dry_run and idx >= 5:
            break

        # skip completed questions
        question_id = item["question_id"]
        if question_id in completed:
            continue  # skip existing

        print("\n\n========================== Processing question: {} ==========================\n".format(item["question"]))

        #* extract scene paths
        episode_id = item["episode_history"] # mini-hm3d-v0/episode_name
        paths = extract_frames(episode_id, args.num_frames, args.frames_directory, extractor)

        if args.test_base_prompt:
            print("testing base prompt")
            # delete existing scenegraph
            if scenegraph_manager.has_episode(episode_id):
                scenegraph_manager.delete_episode_file(episode_id)
            create_base_scenegraph(episode_id=episode_id,
                                      image_paths=paths,
                                      scenegraph_manager=scenegraph_manager,
                                      max_tokens=args.max_tokens,
                                      image_size=args.image_size,
                                      force=args.force)
            continue

        #* check for existence of the episode's scenegraph and create if not found
        if scenegraph_manager.has_episode(episode_id) == False:
            print("No scenegraph for episode: {}".format(episode_id))
            create_base_scenegraph(episode_id=episode_id,
                                      image_paths=paths,
                                      scenegraph_manager=scenegraph_manager,
                                      max_tokens=args.max_tokens,
                                      image_size=args.image_size,
                                      force=args.force)
        
        print("\nscenegraph exists for episode: {}".format(episode_id))

        # generate answer
        question = item["question"]
        output = ask_question( #* answer with updating scenegraph
            question=question,
            image_paths=paths,
            previous_scenegraph=scenegraph_manager.get_scenegraph(episode_id),
            image_size=args.image_size,
            openai_model=args.model,
            openai_seed=args.seed,
            openai_max_tokens=args.max_tokens,
            openai_temperature=args.temperature,
            force=args.force,
        )
        
        print("output: {}".format(output))

        #* save updated_scenegraph
        if output.__contains__("text: "):
            print("has text - output: {}".format(output))
            output_parsed_json = json.loads(output.split("text: ")[0])
        output_parsed_json = json.loads(output.split(OUTPUT_SCENEGRAPH_SEPARATOR)[1])
        scenegraph = output_parsed_json["scenegraph"]
        scenegraph_manager.update_scenegraph(episode_id, scenegraph)

        answer = output_parsed_json["answer"]
    
        # store results
        results.append({"question_id": question_id, "answer": answer})
        json.dump(results, args.output_path.open("w"), indent=2)

    # save at end (redundant)
    json.dump(results, args.output_path.open("w"), indent=2)
    print("saving {:,} answers".format(len(results)))


if __name__ == "__main__":
    main(parse_args())