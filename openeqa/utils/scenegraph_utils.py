import json
import os
from pathlib import Path

base_path = 'data/scenegraphs/'

class ScenegraphManager():
    scenegraphs_per_episode = {}

    def __init__(self) -> None:
        self.__set_scenegraphs()

    def __set_scenegraphs(self) -> None: # set scenegraphs from json files
        for filename in os.listdir(f'{base_path}/mini-hm3d-v0'): # filename = episode_id
            with open(f'{base_path}mini-hm3d-v0/{filename}', 'r') as file:
                data = json.load(file)
                episode_name = filename.split('.')[0]
                self.scenegraphs_per_episode[episode_name] = data
                print("Each Scenegraph loaded: ", episode_name)
        
        print("Scenegraphs loaded: ", self.scenegraphs_per_episode.keys())

    def __get_episode_name(self, episode_id) -> str: # get episode name from episode_id
        return episode_id.split('/')[-1]
    
    def has_episode(self, episode_id) -> bool: # check if episode exists
      episode_name = self.__get_episode_name(episode_id)
      print("Checking if episode exists: ", episode_name)
      return episode_name in self.scenegraphs_per_episode.keys()
        
    def get_scenegraph_path(self, episode_id) -> Path: # get path of scenegraph of episode
        return Path(f'{base_path}{episode_id}.json')

    def __add_episode(self, episode_id) -> None: # add new episode
        episode_name = self.__get_episode_name(episode_id)
        if self.has_episode(episode_name):
            raise Exception("Episode already exists")
        else:
            print(f"Adding new episode: {episode_name}")
            self.scenegraphs_per_episode[episode_name] = {}
    
    def create_scenegraph(self, episode_id, scenegraph) -> None:
        self.__add_episode(episode_id)
        self.scenegraphs_per_episode[episode_id] = scenegraph

        with open(self.get_scenegraph_path(episode_id), 'w') as file:
            json.dump(scenegraph, file)

    def update_scenegraph(self, episode_id: str, new_scenegraph: json) -> None: # update scenegraph of episode and save to file
        print("Updating scenegraph for episode: ", episode_id)
        has = self.has_episode(episode_id)
        if not has:
            raise Exception(f"Updating scenegraph failed: Episode {episode_id} does not exist.")
        self.scenegraphs_per_episode[episode_id] = new_scenegraph
        print("Scenegraph updated: ", self.scenegraphs_per_episode[episode_id])

        with open(self.get_scenegraph_path(episode_id), 'w') as file:
            json.dump(new_scenegraph, file)
            print("Scenegraph saved to file: ", self.get_scenegraph_path(episode_id))
    
    def delete_episode_file(self, episode_id) -> None:
        if not self.has_episode(episode_id):
            raise Exception(f"Deleting episode failed: Episode {episode_id} does not exist.")
        else:
            print("Deleting episode file: ", episode_id)
            episode_name = self.__get_episode_name(episode_id)
            os.remove(self.get_scenegraph_path(episode_id))
            del self.scenegraphs_per_episode[episode_name]

        

'''scenegraph_per_episode 의 구성
{
  "episode 1" : {returned scenegraph from prompt},
  "episode 2" : {"},
  "episode 3" : {"},
  ...
}
'''