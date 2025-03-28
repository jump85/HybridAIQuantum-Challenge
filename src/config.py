# src/config.py
import os
import yaml
from decouple import config as decouple_config

def load_config(config_path="yaml_default_config.yaml"):
    """
    Load configuration from the YAML file.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_full_path = os.path.join(base_dir, config_path)
    with open(config_full_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def create_scaleway_session(cfg):
    """
    Create a Scaleway QaaS session based on the config and environment variables.
    Expects SCW_PROJECT_ID and SCW_SECRET_KEY to be set in the environment.
    """
    import perceval.providers.scaleway as scw
    # Get credentials from the .env file using python-decouple.
    proj_id = decouple_config("SCW_PROJECT_ID") # 
    token = decouple_config("SCW_SECRET_KEY") # 
    platform = cfg["quantum"].get("scaleway_platform", "sim:sampling:p100")
    if not proj_id or not token:
        raise RuntimeError("Scaleway credentials not found in environment.")
    session = scw.Session(
        project_id=proj_id,
        token=token,
        platform=platform,
        max_idle_duration_s=1200,
        max_duration_s=3600
    )
    return session