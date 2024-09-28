
import base64
import os
import traceback
from typing import List, Optional, Tuple

import cv2
import openai
from openeqa.utils.prompt_utils import *
from tenacity import retry, stop_after_attempt, wait_random_exponential

from openeqa.utils.openai_utils import *

class Captions():
    captions_data = {}
    
    def add_caption(self, episode_id: str,
            image_path: int,
            caption: str
        ) -> None:

        caption = {
            "image_idx" : image_path,
            "caption" : caption
        }

        if episode_id not in self.captions_data.keys():
            self.captions_data[episode_id] = []
        
        captions_for_episode = self.captions_data[episode_id] # list
        captions_for_episode.append(caption)

        self.captions_data[episode_id] = captions_for_episode

        print("add_caption: ", caption)
        print("episode_captions " , self.captions_data[episode_id])
    
    def has_caption(self, episode_id: str, image_path: int) -> bool:
        if episode_id in self.captions_data.keys():
            captions_for_episode = self.captions_data[episode_id]
            for caption in captions_for_episode:
                if caption["image_idx"] == image_path:
                    return True
        return False

    def get_caption(self, episode_id: str, segment: Tuple[int, int]) -> Optional[str]:
        first_caption = ""
        second_caption = ""
        if episode_id in self.captions_data.keys():
            captions_for_episode = self.captions_data[episode_id]
            for caption in captions_for_episode:
                # segment[0], segment[1] 에 해당하는 숫자가 caption["image_idx"](str)에 포함되어 있으면 caption["caption"]을 반환
                if str(segment[0]) in caption["image_idx"]:
                    first_caption = caption["caption"]
                if str(segment[1]) in caption["image_idx"]:
                    second_caption = caption["caption"]

        return first_caption, second_caption

def create_captions(
    image_paths: List,
    image_size: int = 512,
    openai_key: Optional[str] = None,
    openai_model: str = "gpt-4o",
    openai_seed: int = 1234,
    openai_max_tokens: int = 128,
    openai_temperature: float = 0.2,
    force: bool = False
) -> Optional[str]:

    openai_key = os.environ["OPENAI_API_KEY"]

    try:
        set_openai_key(key=openai_key)

        prompt = load_prompt("gpt4o-captioning")
        # prefix, suffix = prompt.split("User Query:")
        # suffix = "User Query:" + suffix.format(question=question)

        messages = prepare_openai_vision_messages(
            prefix=prompt, image_paths=image_paths, image_size=image_size
        )
        output = call_openai_api(
            messages=messages,
            model=openai_model,
            seed=openai_seed,
            max_tokens=openai_max_tokens,
            temperature=openai_temperature,
        )

        print("caption output: ", output)
        return output
    
    except Exception as e:
        if not force:
            traceback.print_exc()
            raise e