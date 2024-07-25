import os
import habitat_sim

DATASET_PATH = os.path.join(os.getcwd(), "data/versioned_data/hm3d-0.2/hm3d/val/")

def set_init():
    scene_path = os.path.join(DATASET_PATH, "00800-TEEsavR23oF/TEEsavR23oF.basis.glb")
    config_path = os.path.join(DATASET_PATH, "hm3d_annotated_basis.scene_dataset_config.json")

    sim_settings = {
        "scene": scene_path,
        "config": config_path,
        "default_agent": 0,
        "sensor_height": 1.5,
        "width": 1280,
        "height": 720,
    }

    return sim_settings

def set_environment(sim_settings):
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = sim_settings["scene"]
    backend_cfg.scene_dataset_config_file = sim_settings["config"]
    backend_cfg.enable_physics = False

    return backend_cfg

def set_agent(sim_settings):
    sem_cfg = habitat_sim.CameraSensorSpec()
    sem_cfg.uuid = "color_sensor"
    sem_cfg.sensor_type = habitat_sim.SensorType.COLOR
    sem_cfg.resolution = [sim_settings["height"], sim_settings["width"]]
    sem_cfg.position = [0.0, sim_settings["sensor_height"], 0.0]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sem_cfg]

    return agent_cfg

def main():
    sim_settings = set_init()
    env = set_environment(sim_settings)
    agent = set_agent(sim_settings)

    sim_cfg = habitat_sim.Configuration(env, [agent])
    sim = habitat_sim.Simulator(sim_cfg)

if __name__ == "__main__":
    main()