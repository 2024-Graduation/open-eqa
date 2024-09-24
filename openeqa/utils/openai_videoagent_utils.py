
import base64
import os
import traceback
from typing import List, Optional

import cv2
import openai
from openeqa.utils.prompt_utils import *
from tenacity import retry, stop_after_attempt, wait_random_exponential

from openai_utils import *

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

    openai_key = os.environ("OPENAI_API_KEY")

    try:
        set_openai_key(key=openai_key)

        prompt = load_prompt("gpt4o")
        prefix, suffix = prompt.split("User Query:")
        # suffix = "User Query:" + suffix.format(question=question)

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