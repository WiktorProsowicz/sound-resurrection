# Sound Resurrection

Sound Resurrection is a project containing research results from the Sound Processing in ML area. The goal of the research is to design a model performing sound generation based on masked input with the focus put on the sound taken from video conferences.

The following document contains brief description of the project as well as the explanation of the project's structure.

## Project structure

```yaml
root:
    doc:                    # contains project's documentation
        Documentation.md    # contains mainly non-technical documentation of the projecte
    src:                    # contains the source code
        preprocessing:      # contains data preprocessing tools
    test:                   # contains tests for the software
```

## How to use

To use the project's functionality, follow these steps:

- In the root project's directory run `python3 setup.py setup_venv`
- Than activate the virtual environment with `source venv/bin/activate`
- Install necessary python packages: `pip install -r requirements.txt`
- Then you can run scripts from the `scripts` folder

If you are a contributor, after activating the environment do:

- Install python packages needed for project managing, testing etc: `pip install -r requirements-dev.txt`
- Before each commit you can check whether it is safe to do so with `pre-commit run`
- You have to ensure that all of the tests from the `tests` directory pass: `python3 setup.py run_tests`
- To properly use the `.editorconfig` file, make sure to install the IDE extension e.g. `code --install-extension EditorConfig.EditorConfig`
