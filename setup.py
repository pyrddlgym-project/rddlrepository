# This file is part of rddlrepository

# rddlrepository is free software: you can redistribute it and/or modify
# it under the terms of the MIT License as published by
# the Free Software Foundation.

# rddlrepository is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# MIT License for more details.

# You should have received a copy of the MIT License
# along with rddlrepository. If not, see <https://opensource.org/licenses/MIT>.

from setuptools import setup, find_packages

setup(
      name='rddlrepository',
      version='0.1',
      author="Ayal Taitler, Scott Sanner, Michael Gimelfarb",
      author_email="ataitler@gmail.com, ssanner@mie.utoronto.ca, mike.gimelfarb@mail.utoronto.ca",
      description="Home for all things RDDL",
      license="MIT License",
      url="https://github.com/ataitler/rddlrepository",
      packages=find_packages(),
      install_requires=['pillow>=9.2.0', 'matplotlib>=3.5.0', 'numpy>=1.22'],
      python_requires=">=3.8",
      include_package_data=True,
      package_data={'': [ ]},
      classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

