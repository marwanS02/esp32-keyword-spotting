# Install and Setup

In this first step, you will prepare and install TVM and its support tools.

## Prerequisites

As a prerequisite, make sure that the following dependencies are installed on your system:
- Python 3.8 with virtualenv support or make sure that you are inside an anaconda (miniconda) environment
- LLVM version 11 or higher (see https://apt.llvm.org/ for Debian/Ubuntu)
- Utility to decompress `.tar` archives

On Ubuntu, the following command will be helpful:

```bash
sudo apt-get install python3 python3-dev python3-venv python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev
```

## Installing TVM (NEW)

Make sure to enter your `micro-kws` python environment first!

TVM can be installed in a straightforward fashion via `pip install apache-tvm`. It should be included in `2_deploy/requirements.txt` and be installed via

```bash
pip install -r requirements.txt
```

## Installing TVM (OLD)

*Please ignore this unless you want to learn how to build TVM from source!*

### Using pre-compiled Python packages

~~This is the recommended approach for this lab as it ensures that the used version of TVM is compatible with this repository. See [`whl/README.md`](whl/README.md) for more details.~~

### Compile from source manually

If you want to do your own build of the TVM compiler framework, please follow the instructions at https://tvm.apache.org/docs/install/from_source.html. The step is entirely optional and can be skipped if you have already completed the previous step.

Make sure to use a fixed commit (e.g. `d6632070a01e23270f9f480efc39d09fc38eb55f`) while checking out the repository to work on the lab exercises.

To be able to follow the tutorial, make sure that you enable `USE_MICRO`, `USE_MICRO_STANDALONE_RUNTIME` and `USE_LLVM` in your `build/config.cmake`. In addition, configure the `PYTHONPATH` environment variable properly, e.g. run `export PYTHONPATH=$(pwd)/python` inside the top-level directory of the TVM repository. For custom builds (compiled from source), you must run `python -m tvm.driver.tvmc` instead of `tvmc` in later steps.
