
# Requirements
Install git and python 3.6 on your machine.
Also, you will need to install the following libraries:
      - gfortran
      - libblas-dev
      - liblapack-dev

# Getting the code
To get the latest release of GAMTL run:
```
git clone https://github.com/shgo/gamtl.git
cd gamtl/
```

# Preparing the environment
Create the python environment:
```
python3.6 -m venv gamtl_env
```
Using an isolated environment has the advantage of not messing with your python OS installation.
When it is done, activate the environment with:
```
source gamtl_env/bin/activate
```

Now we will install all python dependencies for GAMTL, that are listed on the `requirements.txt` file.
The following command will install everything for you:
```
python -m pip install -r requirements.txt
```

To test if your environment is properly set up, run:
```
python test_environment.py
```

This will create an artificial dataset and run some methods a few times with minimal configuration.
You should see no error by the end of the process.

# Reproducing experiments on Artificial Dataset

To reproduce the experiments on 'Art1' dataset, run:

```
python exp_var_m_art.py
```
