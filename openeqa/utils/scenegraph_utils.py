import json
import os

class ScenegraphManager():
    scenegraphs_per_episode = {}
   
    def set_scenegraphs(self) -> None: # set scenegraphs from json files
        for filename in os.listdir('data/scenegraphs'):
            with open(f'data/scenegraphs/{filename}', 'r') as file:
                data = json.load(file)
            
        for episode_id, scenegraph in data.items(): # mapping data to scenegraphs_per_episode
            self.scenegraphs_per_episode[episode_id] = scenegraph

    def has_episode(self, episode_id) -> bool: # check if episode exists
      return episode_id in self.scenegraphs_per_episode.keys()
        
    def get_scenegraph_path(self, episode_id) -> str: # get path of scenegraph of episode
        return f'data/scenegraphs/{episode_id}.json'

    def add_episode(self, episode_id) -> None: # add new episode
        if episode_id in self.scenegraphs_per_episode.keys():
            raise Exception("Episode already exists")
        else:
            self.scenegraphs_per_episode[episode_id] = {}

    def update_scenegraph(self, episode_id, new_scenegraph) -> None: # update scenegraph of episode and save to file
        self.has_episode(episode_id)
        self.scenegraphs_per_episode[episode_id] = new_scenegraph

        with open(self.get_scenegraph_path(episode_id), 'w') as file:
            json.dump(new_scenegraph, file)

        

'''scenegraph_per_episode 의 구성
{
  "episode 1" : {유정이가 만드는 scenegraph},
  "episode 2" : {"},
  "episode 3" : {"},
  ...
}
'''