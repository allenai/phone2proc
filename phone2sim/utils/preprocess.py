import filecmp
import os
import shutil
from zipfile import ZipFile


def extract_scene_usdz(scene_usdz: str, output_zip_path: str) -> str:
    # NOTE: The usdz is actually just a zip file. So we're copying it to be one.
    # create the directory
    base_path = os.path.dirname(output_zip_path)
    os.makedirs(base_path)

    # make a copy of file_id.usdz to the base_path
    shutil.copyfile(scene_usdz, output_zip_path)

    with ZipFile(output_zip_path, "r") as zip_ref:
        zip_ref.extractall(base_path)

    scene_usda = os.path.join(base_path, "Room.usda")
    assert os.path.exists(scene_usda), f"Could not find Room.usda in {scene_usdz}"


def get_scene_usda(scene_usdz: str) -> str:
    # remove the extension and directory from file_id, if it has one
    scene_id = os.path.splitext(os.path.basename(scene_usdz))[0]

    base_path = os.path.join(os.path.expanduser("~"), ".phone2sim", scene_id)
    usdz_zip_path = os.path.join(base_path, f"{scene_id}.zip")
    if not os.path.exists(base_path):
        # extract the usdz file
        extract_scene_usdz(scene_usdz, usdz_zip_path)
    elif not filecmp.cmp(scene_usdz, usdz_zip_path):
        raise Exception(f"Cannot overwrite existing scene {scene_id}.")

    room_usda = os.path.join(base_path, "Room.usda")
    return room_usda
