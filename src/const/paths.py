from pathlib import Path


def project_dir():
    return Path(__file__).resolve().parents[2]


def data_dir():
    root_dir = project_dir()
    return root_dir / "data"


def scenes_dir():
    return data_dir() / "scenes"


def res_dir():
    root_dir = project_dir()
    return root_dir / "resources"


def main():
    print(project_dir())
    print(data_dir())


if __name__ == "__main__":
    main()
