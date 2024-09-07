import json
import os

base_path = 'data/scenegraphs/'

class ScenegraphManager():
    scenegraphs_per_episode = {}
   
    def set_scenegraphs(self) -> None: # set scenegraphs from json files
        for filename in os.listdir(base_path): # filename = episode_id
            with open(f'{base_path}{filename}', 'r') as file:
                data = json.load(file)
                print(data)
            
        for episode_id, scenegraph in data.items(): # mapping data to scenegraphs_per_episode
            print(f"Loading episode: {episode_id}")
            self.scenegraphs_per_episode[episode_id] = scenegraph

    def has_episode(self, episode_id) -> bool: # check if episode exists
      return episode_id in self.scenegraphs_per_episode.keys()
        
    def get_scenegraph_path(self, episode_id) -> str: # get path of scenegraph of episode
        return f'{base_path}{episode_id}.json'

    def add_episode(self, episode_id) -> None: # add new episode
        if episode_id in self.scenegraphs_per_episode.keys():
            raise Exception("Episode already exists")
        else:
            print(f"Adding new episode: {episode_id}")
            self.scenegraphs_per_episode[episode_id] = {}

    def update_scenegraph(self, episode_id, new_scenegraph) -> None: # update scenegraph of episode and save to file
        print("Updating scenegraph for episode: ", episode_id)
        has = self.has_episode(episode_id)
        if not has:
            raise Exception(f"Updating scenegraph failed: Episode {episode_id} does not exist.")
        self.scenegraphs_per_episode[episode_id] = new_scenegraph

        with open(self.get_scenegraph_path(episode_id), 'w') as file:
            json.dump(new_scenegraph, file)
        


        

'''scenegraph_per_episode 의 구성
{
  "episode 1" : {returned scenegraph from prompt},
  "episode 2" : {"},
  "episode 3" : {"},
  ...
}
'''