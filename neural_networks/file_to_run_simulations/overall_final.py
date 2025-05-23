# This script loops the value of rho from 0.1 to 1.2 in the in.mp file that
# implements the rNEMD algorithm of Muller-Plathe.
# The output of this script is listed below
# To use this script, on the ssh terminal connecting the aristotle, type in python overall_final.py
# To modify the range of the rho values, change it at the beginning of the for loop
# The pathes to save files should be adapted to the suitable ones
##########

import subprocess
from pathlib import Path
import shutil

import numpy as np


def load_profile(file: Path, initial_offset=3, chunk_size=20,)->np.ndarray:
    """
    load profile.mp from lammps

    Args:
        file: Path, file path for profile.mp
        initial_offset: int, number of comment lines
        chunk_size: int, chunk size of data at each timestep
    
    Returns:
        ndarray 
    """
    index = 0
    result = {}
    while True:
        try:
            info = np.loadtxt(
                file,
                skiprows=initial_offset + chunk_size * index+(index),
                max_rows=1,
            )
            data = np.loadtxt(file, usecols=(0, 3),
                              skiprows=initial_offset +
                              chunk_size * index+(index + 1),
                              max_rows=chunk_size,
                              )
            result[int(info[0])] = data
            index += 1
        except Exception as e:
            # print(e)
            break
    
    matrix = np.array([value[:,1] for _key, value in result.items()])
    
    matrix_after_30000 = matrix[28:] # starts from 30000  timesteps
    return np.average(matrix_after_30000,axis=0)


DATA_ROOT_PATH = Path("./")
DATA_ROOT_PATH.mkdir(exist_ok=True)

LAMMPS_LOG_PATH = Path.joinpath(DATA_ROOT_PATH, "lammps_log")
LAMMPS_LOG_PATH.mkdir(exist_ok=True)

PROFILE_MP_PATH = Path.joinpath(DATA_ROOT_PATH, "profile_mp")
PROFILE_MP_PATH.mkdir(exist_ok=True)

DATA_LOG_EX_PATH = Path.joinpath(DATA_ROOT_PATH, "data_log_ex")
DATA_LOG_EX_PATH.mkdir(exist_ok=True)

PROFILE_DATA_PATH = Path.joinpath(DATA_ROOT_PATH, "profile_data")
PROFILE_DATA_PATH.mkdir(exist_ok=True)


for index, rho in enumerate(np.arange(0.1, 1.2+0.02, 0.02)):
    print(f"************{index} {rho}")

    log_lammps_path = Path.joinpath(
        LAMMPS_LOG_PATH, f"log.rho.{rho:.2f}.lammps").absolute()
    profile_mp_path = Path.joinpath(
        PROFILE_MP_PATH, f"profile.rho.{rho:.2f}.mp").absolute()
    data_log_ex_path = Path.joinpath(
        DATA_LOG_EX_PATH, f"data{rho:.2f}.txt").absolute()
    profile_data_path = Path.joinpath(
        PROFILE_DATA_PATH, f"ave_temp_{rho:.2f}").absolute()

    with open(Path("./in.mp.template"), "r") as template_file:
        in_mp = template_file.read()
        in_mp = in_mp.replace("__rho__", f"{rho:.2f}")
    with open(Path("./in.mp"), "w") as write_file:
        write_file.write(in_mp)

    call_lammps = subprocess.run([
        "lmp_aristotle",
        "-in",
        "in.mp"
    ],
    )

    if call_lammps.returncode != 0:
        raise Exception("return code not zero")

    shutil.copy("./log.lammps", log_lammps_path)
    shutil.copy("./profile.mp", profile_mp_path)

    # extract data from log.lammps for each rho
    # The output is a txt file that contains Step TotEng as columns respectively
    call_exdata = subprocess.run([
        "python",
        "log2txt.py",
        log_lammps_path,
        data_log_ex_path,
        "Step",
        "TotEng"
    ],
    )
    if call_exdata.returncode != 0:
        raise Exception("return code not zero")

    # extract data from profile.mp for each rho
    # The output should be [the number of chunks, Temperature averaged over all timesteps]
    np.savetxt(
        profile_data_path,
        load_profile(profile_mp_path),
    )