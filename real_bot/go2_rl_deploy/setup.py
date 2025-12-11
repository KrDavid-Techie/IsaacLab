from setuptools import setup
import os
from glob import glob

package_name = 'go2_rl_deploy'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'unitree_sdk2py',
        'numpy',
        'scipy',
        'onnxruntime',
        'paramiko',
        ],
    zip_safe=True,
    maintainer='User',
    maintainer_email='user@todo.todo',
    description='RL Policy Deployment for Unitree Go2',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'deploy_node = go2_rl_deploy.deploy_node:main',
        ],
    },
)
