# Prediction of a reactor's isotopic inventory

[![Build Status](https://github.com/ramp-kits/nuclear_inventory_prediction/workflows/build/badge.svg?branch=main&event=push)](https://github.com/ramp-kits/nuclear_inventory_prediction/actions)

Authors : Benjamin Dechenaux, Jean-Baptiste Clavel, Cécilia Damon (IRSN)
with the help of François Caud and Alexandre Gramfort (Université Paris-Saclay)

Institut de Radioprotection et de Sûreté Nucléaire (IRSN)

31 avenue de la division Leclerc

92260 - Fontenay-aux-Roses

## Getting started

### Install

To run a submission and the notebook you will need the dependencies listed
in `requirements.txt`. You can install install the dependencies with the
following command-line:

```bash
pip install -U -r requirements.txt
```

If you are using `conda`, we provide an `environment.yml` file for similar
usage.

### Challenge description

Get started with the [dedicated notebook](starting_kit.ipynb)


### Test a submission

The submissions need to be located in the `submissions` folder. For instance
for `my_submission`, it should be located in `submissions/my_submission`.

To run a specific submission, you can use the `ramp-test` command line:

```bash
ramp-test --submission my_submission
```

You can get more information regarding this command line:

```bash
ramp-test --help
```

### To go further

You can find more information regarding `ramp-workflow` in the
[dedicated documentation](https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/stable/using_kits.html)
