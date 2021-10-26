from setuptools import setup
from setuptools import find_packages

package_name = 'cable_observer'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/params.yaml', 'config/d435i.yaml']),
        ('share/' + package_name + '/launch', ['launch/cable_observer.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Amadeusz Szymko',
    maintainer_email='amadeusz.szymko@put.poznan.pl',
    description='Deformable linear objects tracking',
    license='Apache License, Version 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'cable_observer_node = cable_observer.cable_observer_node:main'
        ],
    },
)
