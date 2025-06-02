from pathlib import Path

from src.const.enums import CamerasEnum, CocoaConditionsEnum


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
        camera_name: CamerasEnum,
) -> Path:
    folder_name = camera_name.value.lower()
    return scene_dir(scene=scene) / folder_name


def image_filepath(
        scene: int,
        camera_name: CamerasEnum,
        extension: str,
        cocoa_condition: CocoaConditionsEnum,
) -> Path:
    filename = f"hsi_{cocoa_condition.value.lower()}.{extension}"
    return camera_dir(scene=scene, camera_name=camera_name) / filename


def annotations_filepath(
        scene: int,
        camera_name: CamerasEnum,
        cocoa_condition: CocoaConditionsEnum,
) -> Path:
    filename = f"annotations_{cocoa_condition.value.lower()}.txt"
    return camera_dir(scene=scene, camera_name=camera_name) / filename


def res_dir() -> Path:
    return data_dir() / "resources"


def metadata_dir() -> Path:
    return res_dir() / "metadata"


def cameras_config_path() -> Path:
    return metadata_dir() / "cameras.json"


def main():
    print(project_dir())
    print(data_dir())


if __name__ == "__main__":
    main()
