from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'scene_graph_pkg'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='3D Scene Graph from fixed RGBD camera',
    license='MIT',
    entry_points={
        'console_scripts': [
            'scene_graph_node = scene_graph_pkg.scene_graph_node:main',
            'realsense_node = scene_graph_pkg.realsense_node:main',
        ],
    },
)
