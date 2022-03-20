# numan
Numerical analysis python library 🔢

![Tests](https://github.com/LemuelPuglisi/numan/actions/workflows/tests.yml/badge.svg)

## Package installation 

Create a virtual environment (you need `virtualenv`): 

```
virtualenv venv
```

Then activate the virtual environment: 

```
source venv/bin/activate
```

Install `numan` package:

```
pip install -e .
```

## Testing 

Install the requirements for development 

```
pip install -r ./requirements_dev.txt
```

Run `mypy`

```
mypy src
```

Run `flake8`

```
flake8 src
```

Run `pytest`:

```
pytest
```



