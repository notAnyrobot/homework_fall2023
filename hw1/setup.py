# setup.py
from setuptools import setup

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "mujoco-mjx",   #==2.2.0
    "gym",  #==0.25.2
    "tensorboard",  #==2.10.0
    "tensorboardX", #==2.5.1
    "matplotlib",   #==3.5.3
    "ipython==7.34.0",
    "moviepy",  #==1.0.3
    "pyvirtualdisplay", #==3.0
    "torch",    #==1.12.1
    "opencv-python",    #==4.6.0.66
    "ipdb", #==0.13.9
    "swig", #==4.0.2
    "box2d-py", #==2.3.8
]

setup(
    name='cs285',
    version='0.1.0',
    packages=['cs285'],
    install_requires=INSTALL_REQUIRES,
)