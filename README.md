# Sound Resurrection

Sound Resurrection is a project containing research results from the Sound Processing in ML area. The goal of the research is to design a model performing sound generation based on masked input with the focus put on the sound taken from video conferences. Another part of the project includes audio signal denoising (in this case - deleting background rustle or conversations, if present) and generating sound with richer spectral envelope from downsampled signal (with effort put on audio distortions that result from audio compression).

The following document contains brief description of the project as well as the explanation of the project's structure.

## Project structure

```yaml
root:
    doc:                    # project's documentation
        Documentation.md    # mainly non-technical documentation of the project
    scripts:                # executables connecting the user with software modules
    src:                    # the source code
        preprocessing:      # data preprocessing tools
        utilities:          # project-wide utils
    tests:                  # tests for the software
        unit:               # unit tests
```

## How to use

#### To use the project's functionality, follow these steps:

- In the root project's directory run:

```bash
python3 setup.py setup_venv
```

- Then activate the virtual environment with:
```bash
source venv/bin/activate`
```

- Install necessary python packages:
```bash
pip install -r requirements.txt
```

- Then you can run scripts from the `scripts` folder.

#### If you are a contributor, after activating the environment do:

- Install python packages needed for project managing, testing etc:
```bash
pip install -r requirements-dev.txt
```

- To properly install `tensorflow` library:
```bash
# If you want to use GPU acceleration (recommended)
pip install --extra-index-url https://pypi.nvidia.com tensorrt-bindings==8.6.1 tensorrt-libs==8.6.1
pip install -U tensorflow[and-cuda]

# Otherwise
pip install tensorflow==2.15.0
```

- To use code checks:
```bash
# Install pre-commit hooks that shall be fired at each commit...
pre-commit install
# ...or run the checks manually (especially before marking a feature branch as ready)
pre-commit run --all-files
```

- You have to ensure that all tests from the `tests` directory pass, before merge of a feature branch to master may be considered:
```bash
python3 setup.py run_unit_tests
```

- To properly use the `.editorconfig` file, make sure to install the IDE extension e.g.
```bash
code --install-extension EditorConfig.EditorConfig
```

## Running training

The training data is available at [datashare.ed.ac.uk](https://datashare.ed.ac.uk/handle/10283/2651). The VCTK is a free data set containing short speech samples varying mainly in gender and accent.

To set up the workspace before training follow the above [guide](#how-to-use). Training a specific model involves running a script from the `scripts` folder. For each such executable file there should be provided an external config with comments describing its contents and purpose of each part.