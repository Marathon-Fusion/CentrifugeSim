import os
import pickle

def save_obj(object_name, obj, step, out_dir="checkpoints"):
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, object_name+f"_{step:07d}.pkl")
    with open(fname, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_obj(object_name, step, out_dir="checkpoints"):
    fname = os.path.join(out_dir, object_name+f"_{step:07d}.pkl")
    with open(fname, "rb") as f:
        return pickle.load(f)