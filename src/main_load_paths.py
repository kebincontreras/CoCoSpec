from src.const.paths import project_dir, data_dir, flat_field_filepath


def main():
    print(project_dir())
    print(data_dir())
    print(flat_field_filepath(camera_name="toucan", extension="tiff"))


if __name__ == "__main__":
    main()
