from setuptools import setup, find_packages

with open("requirements.txt", 'r') as file:
    libs = file.readlines()

setup(
    name = "image_toolbox",
    version = "0.2",
    author = "Kareem Janou",
    description = ("Collection of usefull tools, that can make the developement of neural networks easier."),
    license = "BSD",
    keywords = "ML tools AI CNN pytorch",
    packages=find_packages(),
    install_requires= libs,
    classifiers=[
        "Development Status :: 1 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.8',
    ],
)