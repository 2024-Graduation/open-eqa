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
    prepare_openai_vision_messages,
    set_openai_key,
)
from openeqa.utils.caption_utils import Captions, create_captions
from openeqa.utils.prompt_utils import load_prompt
from openeqa.utils.videoagent import first_step, second_step


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


def ask_question(
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

        prompt = load_prompt("gpt4o")
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

    cached_captions = Captions()

    # process data
    #* question 기준으로 iterate
    for idx, item in enumerate(tqdm.tqdm(dataset)):
        if args.dry_run and idx >= 5:
            break

        # skip completed questions
        question_id = item["question_id"]
        if question_id in completed:
            continue  # skip existing

        episode_id = item['episode_history']

        # extract scene paths
        folder = args.frames_directory / item["episode_history"]
        frames = sorted(folder.glob("*-rgb.png"))
        indices = np.round(np.linspace(0, len(frames) - 1, args.num_frames)).astype(int)
        paths = [str(frames[i]) for i in indices]

        #* 0. question 과 무관하게 추출된 프레임에 대해 image captioning
        #! if cached_captions.has_caption == True -> continue
        for image in paths:
            # create caption
            single_caption = create_captions(image_paths=[image])
            # save the caption to memory
            cached_captions.add_caption(caption=single_caption, episode_id=episode_id, image_path=image)

        # generate answer
        question = item["question"]
        trial = 1

        '''answer = ask_question(
            question=question,
            image_paths=paths,
            image_size=args.image_size,
            openai_model=args.model,
            openai_seed=args.seed,
            openai_max_tokens=args.max_tokens,
            openai_temperature=args.temperature,
            force=args.force,
        )'''
        while (trial < 2):
            if trial==1:
                 #* caption 주고 answer 및 confidence 도출
                answer, confidence = first_step(
                    question=question,
                    captions_data=cached_captions.captions_data[episode_id]
                )

            if confidence < 3:
                trial+=1
                print("confidence is less than 3. try second step.")
                # answer, confidence = second_step(
                #     question=question,
                #     episode_id=episode_id,
                #     num_frames=5, # select multiple best frames
                #     saved_captions=cached_captions.captions_data
                # )

            else:
                break

        # store results
        results.append({"question_id": question_id, "answer": answer})
        json.dump(results, args.output_path.open("w"), indent=2)

    # save at end (redundant)
    json.dump(results, args.output_path.open("w"), indent=2)
    print("saving {:,} answers".format(len(results)))


if __name__ == "__main__":
    main(parse_args())