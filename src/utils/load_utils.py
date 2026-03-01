import os
import pickle
from pathlib import Path
from functools import wraps
import inspect
import hashlib
from collections import Counter
import copy
import yaml
import pandas as pd


DATA_DIR = Path("data/cache")
PICKLES_DIR = DATA_DIR / "pickles"
PICKLES_DIR.mkdir(parents=True, exist_ok=True)

def pickle_cache(cache_dir):
    """
    Cache each function call's result in its own pickle file inside cache_dir.
    cache_dir will be created if it doesn't exist.
    """

    def decorator(func):
        sig = inspect.signature(func)
        has_params = bool(sig.parameters)

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            
            key_data = tuple(sorted(bound.arguments.items()))
            key_bytes = pickle.dumps(key_data)
            key_hash = hashlib.sha256(key_bytes).hexdigest()

            # if the function has no arguments, store in root of data directory
            if not has_params:
                cache_path = os.path.join(PICKLES_DIR, f"{cache_dir}.pkl")
            else:
                func_cache_dir = os.path.join(PICKLES_DIR, cache_dir)
                os.makedirs(func_cache_dir, exist_ok=True)
                cache_path = os.path.join(
                    func_cache_dir,
                    f"{func.__name__}_{key_hash}.pkl"
                )

            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    print(f"Loading cached result for {func.__name__} from {cache_path}")
                    return pickle.load(f)

            result = func(*args, **kwargs)

            with open(cache_path, "wb") as f:
                print(f"Caching result for {func.__name__} to {cache_path}")
                pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

            return result

        return wrapper
    return decorator

def get_model_state(model):
    """Captures the current state of the model and its optimizer."""
    # We use deepcopy to ensure we have a truly independent snapshot in RAM
    return copy.deepcopy(model.state_dict())

def reset_model(model, state_dict):
    """Restores the model to a previous state."""
    model.load_state_dict(state_dict)
    print("Model state reset to clean source weights.")

def load_config(config_file_path):
    with open(config_file_path, 'r') as stream:
        try:
            # Use safe_load for security when dealing with untrusted sources
            config_data = yaml.safe_load(stream)
            return config_data
        except yaml.YAMLError as exc:
            # Handle potential YAML parsing errors
            print(exc)
            return None

def save_result_to_csv(result_dict, output_path):
    """Appends a single result row to the CSV. Creates file/header if it doesn't exist."""
    df = pd.DataFrame([result_dict])
    df.to_csv(output_path, mode='a', index=False, header=not os.path.exists(output_path))
