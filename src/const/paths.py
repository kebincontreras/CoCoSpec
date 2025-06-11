"""Module that composes and returns frequently used project, data, and resources Paths."""

from pathlib import Path


def project_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    return project_dir() / "data"


def scenes_dir() -> Path:
    return data_dir() / "scenes"


def scene_dir(scene: int) -> Path:
    folder_name = f"scene_{scene:02}"
    return scenes_dir() / folder_name


def camera_dir(
        scene: int,
        camera_name: str,
) -> Path:
    folder_name = camera_name.lower()
    return scene_dir(scene=scene) / folder_name


def image_filepath(
        scene: int,
        camera_name: str,
        extension: str,
        cocoa_condition: str,
) -> Path:
    filename = f"hsi_{cocoa_condition.lower()}.{extension}"
    return camera_dir(scene=scene, camera_name=camera_name) / filename


def annotations_filepath(
        scene: int,
        camera_name: str,
        cocoa_condition: str,
) -> Path:
    filename = f"annotations_{cocoa_condition.lower()}.txt"
    return camera_dir(scene=scene, camera_name=camera_name) / filename


def res_dir() -> Path:
    return data_dir() / "resources"


def dark_fields_dir() -> Path:
    return res_dir() / "dark_fields"


def dark_field_filepath(
        camera_name: str,
        extension: str,
) -> Path:
    filename = f"{camera_name.lower()}.{extension}"
    return dark_fields_dir() / filename


def flat_fields_dir() -> Path:
    return res_dir() / "flat_fields"


def flat_field_filepath(
        camera_name: str,
        extension: str,
) -> Path:
    filename = f"{camera_name.lower()}.{extension}"
    return flat_fields_dir() / "spectralon_halogen" / filename


def metadata_dir() -> Path:
    return res_dir() / "metadata"


def wavelengths_dir() -> Path:
    return res_dir() / "wavelengths"
