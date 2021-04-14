import setuptools
import os


def _process_requirements():
    packages = """torch>=1.6.0
    tensorboard>=2.3.0
    function==1.2.0
    numpy>=1.19.4
    opencv-python>=4.4.0.44""".split('\n')

    requires = []
    for pkg in packages:
        if pkg.startswith('git+ssh'):
            return_code = os.system('pip install {}'.format(pkg))
            assert return_code == 0, 'error, status_code is: {}, exit!'.format(return_code)
        else:
            requires.append(pkg)
    return requires


setuptools.setup(
    name="dxtorchutils",
    version="0.2.8",
    author="IanDx",
    author_email="IanDxSSXX@gmail.com",
    description="Some utils and models in pytorch",
    url="https://github.com/Ian-Dx/DxTorchUtils",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=_process_requirements()
)


