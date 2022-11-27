from pathlib import Path
import json

import numpy as np


def load_history(run_dir: str):
    dir_path = Path("learning_gravity") / "runs" / run_dir

    assert dir_path.exists() and dir_path.is_dir()
    history_path = dir_path / "history.txt"
    assert history_path.exists() and history_path.is_file()

    with open(history_path, "r", encoding="utf-8") as hist_file:

        hist_dict = {}

        for i, line in enumerate(hist_file.readlines()):
            if i % 100 == 0:
                print(f"reading row {i}")
            line_dict = json.loads(line.strip("\n").replace("'", '"'))
            for key in line_dict:
                if key in hist_dict:
                    hist_dict[key].append(line_dict[key])
                else:
                    hist_dict[key] = [line_dict[key]]

    for key, val in hist_dict.items():
        hist_dict[key] = np.array(val)

    return hist_dict
