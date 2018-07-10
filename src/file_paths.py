image_path = ''
output_path = '../stl-files/'


def get_image_path():
    return image_path


def set_image_path(path):
    global image_path
    image_path = path


def get_output_path():
    return output_path


def set_output_path(path):
    global output_path
    output_path = path