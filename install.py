from typing import List
from sys import platform, executable
from subprocess import check_call

common = [
    'pygame',
    'pygame_menu',
    'python-decouple',
    'numpy',
    'pandas',
    'tqdm',
    'simple-chalk',
    'tensorflow'
]

windows = [
]

# remember to install tensorflow-deps!!!
darwin = [
]


def install(packages: List[str]) -> None:
    for package in packages:
        check_call([executable, '-m', 'pip', 'install', '--no-cache-dir', package])


if __name__ == '__main__':
    install(common)
    if platform == 'windows':
        install(windows)
    elif platform == 'darwin':
        install(darwin)
