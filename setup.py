import setuptools
import os


def _process_requirements():
    packages = open('requirements.txt').read().strip().split('\n')
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
    version="0.0.46",
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


