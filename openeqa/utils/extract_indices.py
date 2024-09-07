import numpy as np

K = 30 # number of frames to extract
BASE_PROB_RATIO = 0.1

class IndicesExtractor():
    """
    Extract indices for a given episode
    
    Attributes:
        indices_counter (dict): the counter for each index for each episode
        base_prob_ratio (float): the ratio of the probability for random selection for all indices
        residual_prob_ratio (float): the ration of the probability for random selection for indices with the minimum count
    """
    def __init__(self, base_prob_ratio = BASE_PROB_RATIO):
        super(IndicesExtractor, self).__init__()
        np.random.seed(1234) # seed for reproducibility
        
        self.indices_counter = {}
        self.__set_prob_ratio(base_prob_ratio)
    
    def add_episode(self, episode_id: str, episode_len: int):
        """
        Add indices for a new episode to the dictionary. If the episode already exists, do nothing
        
        Args:
            episode_id (str): the id of the episode
            episode_len (int): the length of the episode
            
        Returns:
            None
        """
        if episode_id not in self.indices_counter:
            self.indices_counter[episode_id] = np.zeros(episode_len, dtype=np.int8)

    def extract_indices(self, episode_id: str, num_frames: int):
        """
        Extract indices for a given episode
        
        Args:
            episode_id (str): the id of the episode
            num_frames (int): the number of frames to extract
            
        Returns:
            extracted_indices (numpy): the extracted indices
            indices_status (dict): the dictionary for unseen / seen image frame indices ({'unseen': (numpy), 'seen': (numpy)})
        """
        if episode_id not in self.indices_counter:
            raise ValueError("Episode not found")
        
        episode_len = len(self.indices_counter[episode_id])
        pool_sizes = self.__calculate_pool_size(episode_len, num_frames)
        extracted_indices = np.zeros(num_frames, dtype=np.int32) # initialize extracted indices
        
        start_idx = 0
        for i in range(num_frames):
            end_idx = start_idx + pool_sizes[i]
            pooled_indices = np.arange(start_idx, end_idx)
            
            probabilities = self.__calculate_probabilities(episode_id, pooled_indices)
            extracted_index = np.random.choice(pooled_indices, p=probabilities) # select an index based on probabilities
            
            extracted_indices[i] = extracted_index # store the extracted index
            self.indices_counter[episode_id][extracted_index] += 1 # increment the count
            
            start_idx = end_idx
        
        unseen_indices = np.where(self.indices_counter[episode_id] == 0)[0]
        seen_indices = np.where(self.indices_counter[episode_id] > 0)[0]
        indices_status = {"unseen": unseen_indices, "seen": seen_indices}

        return extracted_indices, indices_status
    
    def __calculate_pool_size(self, episode_len: int, num_frames: int):
        """
        Calculate the pool sizes for each frame extraction
        
        Args:
            episode_len (int): the length of the episode
            num_frames (int): the number of frames to extract
            
        Returns:
            pool_sizes (numpy): the pool sizes for each frame extraction
        """
        base_size = episode_len // num_frames
        remainder = episode_len % num_frames
        
        pool_sizes = np.full(num_frames, base_size, dtype=np.int32)
        if 0 < remainder:
            indices_to_increment = np.linspace(0, num_frames - 1, remainder, dtype=np.int32)
            pool_sizes[indices_to_increment] += 1
        
        return pool_sizes
    
    def __calculate_probabilities(self, episode_id: str, pooled_indices: np.ndarray):
        """
        Calculate the probabilities for each available pooled index
        
        Args:
            episode_id (str): the id of the episode
            pooled_indices (numpy): the available pooled indices
            
        Returns:
            probabilities (numpy): the probabilities for each pooled index
        """
        base_probabilities = np.full(len(pooled_indices), self.base_prob_ratio / len(pooled_indices), dtype=np.float32)
        residual_probabilities = np.zeros(len(pooled_indices), dtype=np.float32)
        
        indices_counts = np.array([self.indices_counter[episode_id][i] for i in pooled_indices], dtype=np.int32)
        min_count = np.min(indices_counts)
        residual_probability = self.residual_prob_ratio * (1 / np.sum(indices_counts == min_count))
        residual_probabilities = np.where(indices_counts == min_count, residual_probability, 0)
        
        return base_probabilities + residual_probabilities
    
    def __set_prob_ratio(self, base_prob_ratio: float):
        """
        Set the probabilities for random selection
        
        Args:
            base_prob_ratio (float): the ratio of the probability for random selection for all indices
            residual_prob_ratio (float): the ratio of the probability for random selection for indices with the minimum count
            
        Returns:
            None
        """
        self.base_prob_ratio = base_prob_ratio
        self.residual_prob_ratio = 1.0 - base_prob_ratio
    
if __name__ == "__main__":
    extractor = IndicesExtractor(BASE_PROB_RATIO)
    
    extractor.add_episode("episode1", 162)
    extractor.add_episode("episode2", 170)
    extractor.add_episode("episode2", 20)
    
    for i in range(20):
        extractor.extract_indices("episode1", K)
        extractor.extract_indices("episode2", K)
        
    print(extractor.indices_counter["episode1"])
    print()
    print(extractor.indices_counter["episode2"])