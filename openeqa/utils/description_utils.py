from typing import Tuple
import os
import traceback
from openeqa.utils.caption_utils import Captions
from openeqa.utils.prompt_utils import *
from openeqa.utils.openai_utils import *
from openeqa.utils.videoagent import parse_description_output

class Descriptions():
    descriptions_data = {}

    def add_description(self, episode_id: str,
            segment: Tuple[int, int],
            description: str,
        ) -> None:

        description = {
            "segment" : segment,
            "description" : description,
        }

        if episode_id not in self.descriptions_data.keys():
            self.descriptions_data[episode_id] = []
        
        descriptions_for_episode = self.descriptions_data[episode_id]
        descriptions_for_episode.append(description)
    
    def get_description(self, episode_id: str, segment: Tuple[int, int]) -> Optional[str]:
        if episode_id in self.descriptions_data.keys():
            descriptions_for_episode = self.descriptions_data[episode_id]
            for description in descriptions_for_episode:
                if description["segment"] == segment:
                    return description["description"]
        return None
    
    def find_segment(self, episode_id: str, my_description: str) -> Optional[Tuple[int, int]]:
        if episode_id in self.descriptions_data.keys():
            descriptions_for_episode = self.descriptions_data[episode_id]
            for description in descriptions_for_episode:
                if description["description"] == my_description:
                    return description["segment"]
        
        print("my_description: ", my_description)
        print("episode_id: ", episode_id)
        print("descriptions_for_episode: ", descriptions_for_episode)
        raise ValueError("Description not found")

def create_descriptions(
    episode_id: str,
    segment: Tuple[int, int],
    cached_captions: Captions,
    image_size: int = 512,
    openai_key: Optional[str] = None,
    openai_model: str = "gpt-4o",
    openai_seed: int = 1234,
    openai_max_tokens: int = 128,
    openai_temperature: float = 0.2,
    force: bool = False
) -> str:
    
    openai_key = os.environ["OPENAI_API_KEY"]

    try:
        set_openai_key(key=openai_key)
        prompt = load_prompt("gpt4o-description")

        first_caption, second_caption = cached_captions.get_segment_captions(episode_id=episode_id, segment=segment)
        content = prompt.format(
            first_caption=first_caption,
            second_caption=second_caption
        )
        # print("description content: ", content)

        messages = prepare_openai_messages(content=content)
        output = call_openai_api(
            messages=messages,
            model=openai_model,
            seed=openai_seed,
            temperature=openai_temperature,
        )

        output = parse_description_output(output)

        return output
    
    except Exception as e:
        if not force:
            traceback.print_exc()
            raise e