

# Getting the code
Install git and python 3.6 on your machine.
Run the following command:
```
git clone https://github.com/shgo/gamtl.git
cd gamtl/
```

# Preparing the environment

To create the python environment run
```
python3.6 -m venv gamtl_env
```

This will create a python environment, without messing with your OS installation.
When finish, we need to activate the environment. On linux (assuming bash):
```
source gamtl_env/bin/activate

```

To make things easy for us, the `requirements.txt` file has all the dependencies required for running the code.
The following command will install everything for you:
```
python -m pip install -r requirements.txt
```

