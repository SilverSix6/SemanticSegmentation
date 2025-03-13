import cv2
import os


def load_single_image(image_path):
    full_image_path = get_full_path_from_root(image_path)

    print(f'Loaded single image: {full_image_path}')
    return cv2.imread(full_image_path, cv2.IMREAD_COLOR)


def load_list_image(file_paths, full_path):
    # If file paths are not relative to the project root update them
    if not full_path:
        new_file_paths = []
        for file_path in file_paths:
            new_file_paths.append(get_full_path_from_root(file_path))
        file_paths = new_file_paths

    # Load images into list
    image_list = []
    for file_path in file_paths:
        image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        if image is not None:
            image_list.append(image)
    return image_list


def load_folder_image(folder_path):
    print(f'Loaded folder of images: {folder_path}')

    # Get full file paths of all files in the given folder
    file_paths = []
    full_folder_path = get_full_path_from_root(folder_path)
    possible_file_paths = os.listdir(full_folder_path)

    # Sort file paths to be in order
    possible_file_paths = sorted(possible_file_paths)

    for file in possible_file_paths:
        full_file_path = os.path.join(full_folder_path, file)
        # Check if path is a file
        if os.path.isfile(full_file_path):
            file_paths.append(full_file_path)
        if os.path.isdir(full_file_path):
            file_paths += load_folder_image(full_file_path)

    return file_paths


def get_full_path_from_root(path):
    project_root = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(project_root, '../..'))
    return os.path.join(project_root, path)
