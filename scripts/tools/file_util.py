import os


def make_if_not_exists(path):
    """
    Creates 'path' unless it exists, in which case do nothing.
    """
    os.makedirs(path, exist_ok=True)
