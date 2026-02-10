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

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
      name='rddlrepository',
      version='2.1',
      author="Ayal Taitler, Michael Gimelfarb, Scott Sanner",
      author_email="ataitler@gmail.com, mike.gimelfarb@mail.utoronto.ca, ssanner@mie.utoronto.ca",
      description="rddlrepository: a repository of RDDL problem description files",
      #long_description=long_description,
      license="MIT License",
      url="https://github.com/pyrddlgym-project/rddlrepository",
      packages=find_packages(),
      install_requires=['numpy'],
      python_requires=">=3.8",
      include_package_data=True,
      entry_points={ 
          'console_scripts': [ 'rddlrepo=rddlrepository.entry_point:main' ],
      },
      classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

