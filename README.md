**This repo builds upon the powerplantmatching repo from the PyPSA community and processes the PyPSA powerplantmatching dataset (powerplants.csv via PyPSA_processing.py) and uses an inflow-driven model to determining the age cohort composition of the EU power plant stock on a single country and granular technology level for a desired target year (via age_cohort_assignment_via_inflow_driven_model.py). See folder and realted files in _pypsa_age_cohort_**

**Why an inflow-driven cohort survival model rather than direct capacity accounting?**
While the PyPSA dataset contains explicit decommissioning records compiled from real-world statistics, we opt for an inflow-driven stock model combined with a lognormal lifetime distribution rather than simply reading the surviving fleet directly from the data. The primary motivation is that decommissioning records in aggregated statistical databases are systematically less complete and less reliable than installation records. Capacity additions are typically well-documented at the time of commissioning, whereas retirements are often recorded with a delay, omitted entirely for older plants that were quietly phased out, or inconsistently captured across different national statistics that feed into the database. This asymmetry means that the surviving fleet inferred directly from the data tends to be biased toward older cohorts that should have been retired but lack a recorded decommissioning date.
The inflow-driven approach instead takes only the capacity addition time series as input and applies a lognormal lifetime distribution to estimate the probability that a unit installed in year t is still operational by the reference year. This separates the more reliable empirical signal — when capacity was built — from the less reliable one — when it was retired — and replaces the latter with a physically motivated distributional assumption. The lognormal distribution is a standard choice for component lifetimes as it is right-skewed, strictly positive, and well-supported in the industrial ecology and material flow analysis literature for energy infrastructure. The resulting age distribution is therefore expected to be more robust to the known data quality asymmetry between installations and retirements in historical plant-level statistics.




<!--
SPDX-FileCopyrightText: Contributors to powerplantmatching <https://github.com/pypsa/powerplantmatching>

SPDX-License-Identifier: MIT
-->

# powerplantmatching

[![pypi](https://img.shields.io/pypi/v/powerplantmatching.svg)](https://pypi.org/project/powerplantmatching/)
[![conda](https://img.shields.io/conda/vn/conda-forge/powerplantmatching.svg)](https://anaconda.org/conda-forge/powerplantmatching)
![pythonversion](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FPyPSA%2Fpowerplantmatching%2Fmaster%2Fpyproject.toml)
[![Tests](https://github.com/PyPSA/powerplantmatching/actions/workflows/test.yml/badge.svg)](https://github.com/PyPSA/powerplantmatching/actions/workflows/test.yml)
[![doc](https://readthedocs.org/projects/powerplantmatching/badge/?version=latest)](https://powerplantmatching.readthedocs.io/en/latest/)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/PyPSA/powerplantmatching/master.svg)](https://results.pre-commit.ci/latest/github/PyPSA/powerplantmatching/master)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![REUSE status](https://api.reuse.software/badge/github.com/pypsa/powerplantmatching)](https://api.reuse.software/info/github.com/pypsa/powerplantmatching)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3358985.svg)](https://zenodo.org/record/3358985#.XUReFPxS_MU)
[![Stack Exchange questions](https://img.shields.io/stackexchange/stackoverflow/t/pypsa)](https://stackoverflow.com/questions/tagged/pypsa)

A toolset for cleaning, standardizing and combining multiple power
plant databases.

This package provides ready-to-use power plant data for the European power system.
Starting from openly available power plant datasets, the package cleans, standardizes
and merges the input data to create a new combined dataset, which includes all the important information.
The package allows to easily update the combined data as soon as new input datasets are released.

You can directly [download the current version of the data](https://downgit.github.io/#/home?url=https://github.com/PyPSA/powerplantmatching/blob/master/powerplants.csv) as a CSV file.

Initially, powerplantmatching was developed by the
[Renewable Energy Group](https://fias.uni-frankfurt.de/physics/schramm/complex-renewable-energy-networks/)
at [FIAS](https://fias.uni-frankfurt.de/) and is now maintained by the [Digital Transformation in Energy Systems Group](https://tub-ensys.github.io/) at the Technical University of Berlin to build power plant data
inputs to [PyPSA](http://www.pypsa.org/)-based models for carrying
out simulations.

### Main Features

- clean and standardize power plant data sets
- aggregate power plant units which belong to the same plant
- compare and combine different data sets
- create lookups and give statistical insight to power plant goodness
- provide cleaned data from different sources
- choose between gross/net capacity
- provide an already merged data set of multiple different open data sources
- scale the power plant capacities in order to match country-specific statistics about total power plant capacities
- visualize the data
- export your powerplant data to a [PyPSA](https://github.com/PyPSA/PyPSA)-based model

## Map

![powerplants.png](docs/assets/images/powerplants.png)

## Installation

 Using pip

```bash
pip install powerplantmatching
```

or conda

```bash
conda install -c conda-forge powerplantmatching
```

# Contributing and Support
We strongly welcome anyone interested in contributing to this project. If you have any ideas, suggestions or encounter problems, feel invited to file issues or make pull requests on GitHub.
-   In case of code-related **questions**, please post on [stack overflow](https://stackoverflow.com/questions/tagged/pypsa).
-   For non-programming related and more general questions please refer to the [PyPSA mailing list](https://groups.google.com/group/pypsa).
-   To **discuss** with other PyPSA & technology-data users, organise projects, share news, and get in touch with the community you can use the [discord server](https://discord.gg/JTdvaEBb).
-   For **bugs and feature requests**, please use the [powerplantmatching Github Issues page](https://github.com/PyPSA/powerplantmatching/issues).


## Citing powerplantmatching

If you want to cite powerplantmatching, use the following paper

- F. Gotzens, H. Heinrichs, J. Hörsch, and F. Hofmann, [Performing energy modelling exercises in a transparent way - The issue of data quality in power plant databases](https://www.sciencedirect.com/science/article/pii/S2211467X18301056?dgcid=author), Energy Strategy Reviews, vol. 23, pp. 1–12, Jan. 2019.

with bibtex

```
@article{gotzens_performing_2019,
 title = {Performing energy modelling exercises in a transparent way - {The} issue of data quality in power plant databases},
 volume = {23},
 issn = {2211467X},
 url = {https://linkinghub.elsevier.com/retrieve/pii/S2211467X18301056},
 doi = {10.1016/j.esr.2018.11.004},
 language = {en},
 urldate = {2018-12-03},
 journal = {Energy Strategy Reviews},
 author = {Gotzens, Fabian and Heinrichs, Heidi and Hörsch, Jonas and Hofmann, Fabian},
 month = jan,
 year = {2019},
 pages = {1--12}
}
```

and/or the current release stored on Zenodo with a release-specific DOI:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3358985.svg)](https://zenodo.org/record/3358985#.XURat99fjRY)

## Licence

`powerplantmatching` is released as free software under the [MIT](LICENSES/MIT.txt) license.
The default output data [powerplants.csv](powerplants.csv) generated by the package is released under [CC BY 4.0](LICENSES/CC-BY-4.0.txt).
Parts of the repository may be licensed under different licenses, especially dependent package binaries for `duke` being licensed under [Apache 2.0 license](https://github.com/PyPSA/powerplantmatching/tree/master/LICENSES/Apache-2.0.txt).

This repository uses the [REUSE](https://reuse.software/) conventions to indicate the licenses that apply to individual files and parts of the repository.
For details on the licenses that apply, see the the header information of the respective files and [REUSE.toml](REUSE.toml) for details.

Copyright 2018-2024 Fabian Gotzens (FZ Jülich), Jonas Hörsch (KIT), Fabian Hofmann (FIAS)
Copyright 2025- Contributors to powerplantmatching <https://github.com/pypsa/powerplantmatching>

You can find a list of contributors in the [contributors page](https://github.com/PyPSA/powerplantmatching/graphs/contributors) and in the [contributors file](docs/contributors.md).
