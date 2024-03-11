# Authors: Vasisht Duddu
# Copyright 2024 Secure Systems Group, University of Waterloo & Aalto University, https://crysp.uwaterloo.ca/research/SSG/
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pickle
from pathlib import Path
from typing import Optional, Any
import logging


def create_dir_if_doesnt_exist(path_to_dir: Path, log: logging.Logger) -> Optional[Path]:
    """Create directory using provided path.
    No explicit checks after creation because pathlib verifies it itself.
    Check pathlib source if errors happen.

    Args:
        path_to_dir (Path): Directory to be created
        log (logging.Logger): Logging facility

    Returns:
        Optional[Path]: Maybe Path to the created directory
    """
    resolved_path: Path = path_to_dir.resolve()

    if not resolved_path.exists():
        log.info("{} does not exist. Creating...")

        try:
            resolved_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            log.error("Error occurred while creating directory {}. Exception: {}".format(resolved_path, e))
            return None

        log.info("{} created.".format(resolved_path))
    else:
        log.info("{} already exists.".format(resolved_path))

    return resolved_path


def save_to_pickle_file(object: Any, save_file: Path, log: logging.Logger) -> Optional[Path]:
    """Save any file to a pickle binary.
    This does not check that required folder structure exists and will fail.
    Perform path yourself checks before calling.

    Args:
        object (Any): Python object to be saved
        save_file (Path): Absolute path (with file name)
        log (logging.Logger): Logging facility

    Returns:
        Optional[Path]: Path where saved if successful, None otherwise
    """
    resolved_path: Path = save_file.resolve()
    try:
        with resolved_path.open(mode="wb") as f:
            pickle.dump(object, f, protocol=pickle.HIGHEST_PROTOCOL)

    except Exception as e:
        log.error("Error occurred while saving to binary pickle file {}. Exception: {}".format(resolved_path, e))
        return None

    if not resolved_path.exists():
        log.error("Saving went fine but file doesn't exist anyway {}.".format(resolved_path))
        return None

    return resolved_path


def load_from_pickle_file(load_file: Path, log: logging.Logger) -> Optional[Any]:
    """Load any file from a pickle binary.

    Args:
        load_file (Path): Absolute path (with file name)
        log (logging.Logger): Logging facility

    Returns:
        Optional[Any]: Loaded file if successful, None otherwise
    """
    resolved_path: Path = load_file.resolve()

    if not resolved_path.exists():
        log.error("Provided file doesn't exist: {}".format(resolved_path))
        return None

    try:
        with resolved_path.open(mode="rb") as f:
            loaded_object: Any = pickle.load(f)

    except Exception as e:
        log.error("Error occurred while loading from binary pickle file {}. Exception {}".format(resolved_path, e))
        return None

    return loaded_object
