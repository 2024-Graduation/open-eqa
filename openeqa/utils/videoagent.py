import json
import os
import traceback
from typing import Optional, Tuple

from openai import OpenAI

from openeqa.utils.openai_utils import call_openai_api, prepare_openai_messages, prepare_openai_vision_messages, set_openai_key
from openeqa.utils.prompt_utils import load_prompt

client = OpenAI()

def parse_description_output(output: str) -> str:
    # print("let me parse description: ", output)
    if output.startswith("```json"):
        output = output[output.find("```json") + len("```json"):].strip()
        if output.endswith("```"):
            output = output[:-3].strip()
    output = json.loads(output)
    return output["description"]

def parse_final_answer(output: str) -> str:
    output = json.loads(output)
    return output["final_answer"]

def get_segment_descriptions(episode_id, segments, cached_descriptions):
    segment_descriptions = []
    for segment in segments: # tuple
        segment_description = cached_descriptions.get_description(episode_id=episode_id, segment=segment)
        segment_descriptions.append(segment_description) # ["", "", ...]
    
    return segment_descriptions

def select_best_segment(
        question: str,
        episode_id: str,
        segments: list, # list of tuple
        cached_descriptions
    ) -> Tuple[int, int]:
    prompt = load_prompt("gpt4o-segment-selection")
    segment_descriptions = get_segment_descriptions(episode_id=episode_id, segments=segments, cached_descriptions=cached_descriptions)

    messages = prepare_openai_messages(content = prompt.format(
        question=question,
        segment_description=segment_descriptions
    ))

    output = call_openai_api(
        messages=messages,
        model="gpt-4o",
        seed=1234,
        temperature=0.2,
        # verbose=True
    )
    
    # print("select_best_segment output: ", parse_description_output(output)) # {segment: [0, 1]}
    segment = cached_descriptions.find_segment(episode_id=episode_id, my_description=parse_description_output(output))
    return segment

def get_final_answer(
    question:str,
    segment_paths: list,
    episode_id: str = None,
    cached_captions: list = None,
    cached_descriptions: list = None,
    image_size: int = 512,
    openai_key: Optional[str] = None,
    openai_model: str = "gpt-4o",
    openai_seed: int = 1234,
    openai_temperature: float = 0.2,
    force: bool = False
) -> Optional[str]:
    openai_key = os.environ["OPENAI_API_KEY"]

    try:
        set_openai_key(key=openai_key)
        prompt = load_prompt("gpt4o")
        content = prompt.format(question=question)
        suffix = ""

        if cached_captions:
            suffix = suffix.format(captions=cached_captions)
        
        if cached_descriptions:
            suffix = f"Descriptions: {get_segment_descriptions(episode_id=episode_id, segments=segment_paths, cached_descriptions=cached_descriptions)}"

        messages = prepare_openai_vision_messages(
            prefix=content, suffix=suffix, image_paths=segment_paths, image_size=image_size
        )

        output = call_openai_api(
            messages=messages,
            model=openai_model,
            seed=openai_seed,
            temperature=openai_temperature,
        )

        print("final answer output: ", output) # {final_answer: "xxx"}

        return parse_final_answer(output)
        # return output
    
    except Exception as e:
        if not force:
            traceback.print_exc()
            raise e