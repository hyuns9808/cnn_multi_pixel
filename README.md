# cnn_multi_pixel

Project aimed to run training/testing on CNNs using power traces from our multiple pixel ADC design.

## Main Goals
1. Utilize the Poetry Python library to mitigate any potential dependency/environment issues.
2. Modularize all code for better version control and debugging.
3. Apply dynamic value normalization using average exponent values of all traces during initial setup.
4. Implement overall cleaner logic structure with better documentation.

## User Guideline

### Initial Setup
This project is created via [Poetry][https://python-poetry.org/] for Python packaging and dependency management.
The minimum required Python version is **3.12**. If you do NOT have a version of Python higher than 3.12, install it prior to the following steps.
1. Clone the repo to your local environment.
2. Open a terminal on the cloned directory.
3. Install Poetry according to your environment. Follow instructions of "Installation" section of [this][https://python-poetry.org/docs/] official document.
4. IF you need a .venv folder within the project root directory, run the following command:
```
poetry config virtualenvs.in-project true
```
Note that .venv is included in the .gitignore file of the repo.
5. Run the following command to create a virtual environment and install all specified dependencies:
```
poetry install
```
6. Adjust the hyperparamters from the 'hyperparams.py' file in the project root directory. Follow instructions within the given comments of the .py file.
7. Based on the set hyperparameters, add the power trace files you want to use for training/testing.
8. Run the following command to run the CNN script:
```
poetry run script.py
```

### General Caveats
- The .gitignore file specifies to not push the following files/folders:
    1. .venv
    2. any subdirectories under './trace_files'. Thus, if it is your initial setup on a new environment you will have to manually add trace files to the ./trace_files directory.
- The given poetry.lock is the .lock file obtained from a successful run. If there are any issues, delete it and run 'poetry install' again.