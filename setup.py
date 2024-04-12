#!/usr/bin/env python
import os

from setuptools import find_packages, setup

version = "v0.1.0"

# sys.path.append("./tests")

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

import os

setup(
    name="case_pipeline",
    version=version,
    package_dir={"": "src"},
    packages=find_packages("src"),
    zip_safe=False,
)
