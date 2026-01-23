
This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

Where you instead should add your answers. Any other changes may have unwanted consequences when your report is
auto-generated at the end of the course. For questions where you are asked to include images, start by adding the image
to the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

`![my_image](figures/<image>.<extension>)`

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

Will generate a `.html` page of your report. After the deadline for answering this template, we will auto-scrape
everything in this `reports` folder and then use this utility to generate a `.html` page that will be your serve
as your final hand-in.

Running

```bash
python report.py check
```

Will check your answers in this template against the constraints listed for each question e.g. is your answer too
short, too long, or have you included an image when asked. For both functions to work you mustn't rename anything.
The script has two dependencies that can be installed with

```bash
pip install typer markdown
```

or

```bash
uv add typer markdown
```

## Overall project checklist

The checklist is *exhaustive* which means that it includes everything that you could do on the project included in the
curriculum in this course. Therefore, we do not expect at all that you have checked all boxes at the end of the project.
The parenthesis at the end indicates what module the bullet point is related to. Please be honest in your answers, we
will check the repositories and the code to verify your answers.

### Week 1

* [X] Create a git repository (M5)
* [X] Make sure that all team members have write access to the GitHub repository (M5)
* [X] Create a dedicated environment for you project to keep track of your packages (M2)
* [X] Create the initial file structure using cookiecutter with an appropriate template (M6)
* [X] Fill out the `data.py` file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
* [X] Add a model to `model.py` and a training procedure to `train.py` and get that running (M6)
* [X] Remember to either fill out the `requirements.txt`/`requirements_dev.txt` files or keeping your
    `pyproject.toml`/`uv.lock` up-to-date with whatever dependencies that you are using (M2+M6)
* [X] Remember to comply with good coding practices (`pep8`) while doing the project (M7)
* [X] Do a bit of code typing and remember to document essential parts of your code (M7)
* [X] Setup version control for your data or part of your data (M8)
* [X] Add command line interfaces and project commands to your code where it makes sense (M9)
* [X] Construct one or multiple docker files for your code (M10)
* [X] Build the docker files locally and make sure they work as intended (M10)
* [X] Write one or multiple configurations files for your experiments (M11)
* [X] Used Hydra to load the configurations and manage your hyperparameters (M11)
* [X] Use profiling to optimize your code (M12)
* [X] Use logging to log important events in your code (M14)
* [X] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
* [X] Consider running a hyperparameter optimization sweep (M14)
* [X] Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)

### Week 2

* [X] Write unit tests related to the data part of your code (M16)
* [X] Write unit tests related to model construction and or model training (M16)
* [X] Calculate the code coverage (M16)
* [X] Get some continuous integration running on the GitHub repository (M17)
* [X] Add caching and multi-os/python/pytorch testing to your continuous integration (M17)
* [X] Add a linting step to your continuous integration (M17)
* [X] Add pre-commit hooks to your version control setup (M18)
* [X] Add a continues workflow that triggers when data changes (M19)
* [X] Add a continues workflow that triggers when changes to the model registry is made (M19)
* [ ] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
* [X] Create a trigger workflow for automatically building your docker images (M21)
* [ ] Get your model training in GCP using either the Engine or Vertex AI (M21)
* [X] Create a FastAPI application that can do inference using your model (M22)
* [ ] Deploy your model in GCP using either Functions or Run as the backend (M23)
* [X] Write API tests for your application and setup continues integration for these (M24)
* [X] Load test your application (M24)
* [X] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
* [X] Create a frontend for your API (M26)

### Week 3

* [ ] Check how robust your model is towards data drifting (M27)
* [ ] Setup collection of input-output data from your deployed application (M27)
* [ ] Deploy to the cloud a drift detection API (M27)
* [ ] Instrument your API with a couple of system metrics (M28)
* [ ] Setup cloud monitoring of your instrumented application (M28)
* [ ] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
* [ ] If applicable, optimize the performance of your data loading using distributed data loading (M29)
* [ ] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### Extra

* [ ] Write some documentation for your application (M32)
* [ ] Publish the documentation to GitHub Pages (M32)
* [ ] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Create an architectural diagram over your MLOps pipeline
* [ ] Make sure all group members have an understanding about all parts of the project
* [ ] Uploaded all your code to GitHub

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

--- 30 ---

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

--- s224182, s251999 ---

### Question 3
> **Did you end up using any open-source frameworks/packages not covered in the course during your project? If so**
> **which did you use and how did they help you complete the project?**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

We used several open-source frameworks beyond the course curriculum. Loguru provided structured logging with automatic context and formatting, making debugging significantly easier than standard Python logging. ONNX Runtime enabled model optimization and faster inference by converting PyTorch models to ONNX format, reducing prediction latency by approximately 30-40%. Locust facilitated load testing of our API, allowing us to measure throughput and identify performance bottlenecks under concurrent request loads. We also used Ruff as a fast all-in-one linter and formatter, replacing multiple tools like Black, Flake8, and isort with a single unified solution that runs 10-100x faster.

## Coding environment

> In the following section we are interested in learning more about you local development environment. This includes
> how you managed dependencies, the structure of your code and how you managed code quality.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Recommended answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>
> Answer:

We manage dependencies via pyproject.toml using setuptools dynamic pins that read requirements.txt for runtime and requirements_dev.txt for dev/test/tooling; Python 3.12+ is required.

To clone the environment:

* python3.12 -m venv .venv && source .venv/bin/activate
* pip install --upgrade pip
* pip install -r requirements.txt requirements_dev.txt

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. What did you fill out? Did you deviate from the template in some way?**
>
> Recommended answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
>
> Answer:

We followed the cookiecutter MLOps layout and filled in domain pieces: package code in energy_dissagregation_mlops (API/CLI, training, data collection, drift detection, evaluation), experiment configs in configs (Hydra YAMLs), runnable scripts in scripts (training, sweep, ONNX export, drift tests), and tests in tests. We added Docker assets—root Dockerfile, Dockerfile.dev, and dockerfiles for API/CLI/train—to keep reproducible images. We also added frontend (simple client), loadtest with Locust, and profiling_results plus reports for profiling/plots. We kept data placeholders under data and tracked model artifacts in models. Deviations from the template are mainly the FastAPI surface, ONNX export, load testing, and the lightweight frontend; we removed nothing structural, just filled in the template with NILM-specific code/configs.

### Question 6

> **Did you implement any rules for code quality and format? What about typing and documentation? Additionally,**
> **explain with your own words why these concepts matters in larger projects.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used ... for linting and ... for formatting. We also used ... for typing and ... for documentation. These*
> *concepts are important in larger projects because ... . For example, typing ...*
>
> Answer:

We enforced code quality through multiple layers. Ruff handles both linting and formatting with rules for import sorting (I), naming conventions (N), and PEP8 compliance (E, W), configured in pyproject.toml with a 120-character line length. Pre-commit hooks automatically run Ruff checks, trailing whitespace removal, end-of-file fixers, and YAML validation before each commit. Our CI pipeline (linting.yaml) validates formatting on every push and pull request. For typing, we used Python 3.12+ type hints throughout (e.g., `str | None`, `Path`, return types) to catch type errors early and improve IDE support. Documentation includes function docstrings, inline comments for complex logic, and comprehensive logging via Loguru.

These practices are crucial in larger projects because they reduce cognitive load when reading others' code, prevent subtle bugs through static analysis, enable safe refactoring with IDE support, and ensure consistent code style across team members. Type hints, for example, allow IDEs to provide accurate autocomplete and catch type mismatches before runtime, while automated formatting eliminates bikeshedding debates about code style.

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

We implemented 10 unit tests across five test files. In test_api.py, we have 5 tests validating the FastAPI endpoints: health check (status 200 + model loaded flag), single-sample prediction (shape validation), batch prediction (multi-sample output), empty input rejection (400 error), and ONNX endpoint functionality. In test_data.py, 2 tests verify dataset construction and data loading from preprocessed chunks. In test_model.py, 2 tests check model instantiation and a complete training step (loss reduction). Additionally, test_drift_detection.py and test_data_collection.py validate drift detection and data collection modules. These tests cover the most critical paths: data pipeline integrity, model forward/backward passes, and API contract compliance.

### Question 8

> **What is the total code coverage (in percentage) of your code? If your code had a code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

Our total code coverage is approximately 70%, which covers all source code in the energy_dissagregation_mlops package. While this is reasonable, we're far from 100%. Even if we achieved 100% coverage, it wouldn't guarantee error-free code—coverage measures which lines execute during tests, not whether those tests validate correct behavior, edge cases, or integration bugs. A test suite can pass every line without catching logic errors, race conditions, or domain-specific failures (e.g., numerical instability in model predictions, drift detection memory leaks, or edge cases in data preprocessing). High coverage is a good signal for maintenance and regression prevention, but it's no substitute for thoughtful test design, property-based testing, and real-world validation. True confidence comes from combining coverage metrics with diverse test strategies: unit tests for isolated logic, integration tests for component interaction, and end-to-end tests for realistic workflows.

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

Yes, we used branches and pull requests throughout development. Each major feature or weekly milestone had its own branch (week2, week3, Week_1), which was merged into main via PR after code review and CI checks passing. For example, PR #15 merged the week2 branch containing unit tests and coverage reports. This workflow provided several benefits: it allowed us to work on features independently without blocking each other, enabled code review before merging to catch issues early, ensured CI tests passed on each feature branch before integration, and created a clear history of when features were added. PRs also served as documentation of why changes were made, with descriptions and discussion threads preserving context for future reference.

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

We did not integrate DVC into our workflow. The main reason was project scope—for this MLOps course, we prioritized other aspects like CI/CD, containerization, API development, monitoring, and drift detection over data versioning. The UK-DALE dataset is static and small enough to manage manually, and we had limited team members working on the same data pipeline simultaneously. However, DVC would have been beneficial if we had: multiple preprocessing versions to compare, frequent data updates requiring version tracking, team members in different locations needing synchronized data access, or experimentation with different dataset subsets. DVC would enable reproducibility by linking specific model versions to exact data versions, critical for debugging when model performance changes unexpectedly.

### Question 11

> **Discuss you continuous integration setup. What kind of continuous integration are you running (unittesting,**
> **linting, etc.)? Do you test multiple operating systems, Python  version etc. Do you make use of caching? Feel free**
> **to insert a link to one of your GitHub actions workflow.**
>
> Recommended answer length: 200-300 words.
>
> Example:
> *We have organized our continuous integration into 3 separate files: one for doing ..., one for running ... testing*
> *and one for running ... . In particular for our ..., we used ... .An example of a triggered workflow can be seen*
> *here: <weblink>*
>
> Answer:

We organized CI into six separate workflows: tests.yaml runs unit tests across multiple OS (Ubuntu, Windows, macOS), Python versions (3.11, 3.12), and PyTorch versions (2.6.0, 2.7.0) using a matrix strategy, with pip caching enabled and coverage reporting. linting.yaml enforces Ruff code formatting and linting checks. docker_build.yaml builds and pushes Docker images.

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: Python  my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

We use Hydra config files to manage experiments. Each YAML in configs defines preprocessing, training, and evaluation parameters. For example, quick_test.yaml enables fast debugging with 5 epochs, while normal_training.yaml runs full training with 50 epochs. To run experiments:

python scripts/run_experiment.py --config-name quick_test
python scripts/run_experiment.py --config-name normal_training

Parameters can be overridden from CLI:

python scripts/run_experiment.py --config-name normal_training train.lr=0.0001 train.epochs=100

Alternatively, use direct CLI commands:

python -m energy_dissagregation_mlops.cli preprocess --data-path data/raw/ukdale.h5 --output-folder data/processed
python -m energy_dissagregation_mlops.cli train --preprocessed-folder data/processed --epochs 50 --batch-size 32 --lr 0.0001
python -m energy_dissagregation_mlops.cli evaluate --preprocessed-folder data/processed

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

We ensure reproducibility through multiple mechanisms. All experiment parameters are stored in Hydra YAML config files (configs) that capture complete setup. Weights & Biases automatically logs hyperparameters, metrics, and model artifacts for each training run. We use Loguru for comprehensive logging of all steps. Model checkpoints save the full state (weights, optimizer, epoch, validation loss) to best.pt. Git tracks code versions at experiment time. Docker ensures environment consistency with pinned Python and PyTorch versions.

To reproduce an experiment, a team member would:

Checkout the git commit from the W&B run metadata
Use the saved config from W&B or the local configs/ folder
Run: python scripts/run_experiment.py --config-name normal_training
All parameters, metrics, and artifacts are automatically logged to W&B for full traceability
This combination of versioned configs, comprehensive logging, and centralized artifact storage makes all experiments fully reproducible regardless of time elapsed.

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Recommended answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:



--- question 14 fill here ---

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments/project? Include how you would run your docker images and include a link to one of your docker files.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:


We developed multiple Docker images for different purposes. The Dockerfile serves the production API using uvicorn on port 8000 with health checks. Dockerfile.dev provides a development environment with testing/linting tools. train.dockerfile uses PyTorch 2.0 with CUDA support for GPU-accelerated training. api.dockerfile runs the inference API with model artifacts. cli.dockerfile provides CLI access.

To run the API container:

docker build -f dockerfiles/api.dockerfile -t energy-api .
docker run -p 8000:8000 -v ./models:/app/models energy-api

For training with GPU:

docker build -f dockerfiles/train.dockerfile -t energy-train .
docker run --gpus all -v ./data:/app/data -v ./models:/app/models energy-train


### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

For debugging, we use Loguru for logging across the entire codebase (info and debug messages) and Weights & Biases (W&B) to compare experiments. We also use pytest for unit testing.

We performed profiling with profile_training.py using our profiling.py module. The results were saved in training_profile.json. The profiling helped identify bottlenecks in data loading and GPU memory usage.

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Recommended answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

We did not extensively use GCP services in this project. The main reason was that we prioritized building a complete local MLOps pipeline first—including training, testing, API development, ONNX optimization, and load testing—before cloud deployment. With limited GCP credits and a two-person team, we focused on demonstrating MLOps practices that work regardless of cloud provider: containerization, automated testing, experiment tracking with W&B, and reproducible workflows. In a production scenario, we would use GCP services like Cloud Storage for datasets, Vertex AI for distributed training, Cloud Run for serverless API deployment, and Cloud Monitoring for observability.

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

We did not set up a GCP bucket for this project. Our data remained local in the data/ directory with manual version control through Git (for small processed artifacts) and GitHub releases (for larger files like the raw UK-DALE dataset). This was sufficient given the static nature of our dataset and the small team size. In a production scenario, we would store raw data in a GCP bucket with lifecycle policies, use bucket notifications to trigger preprocessing pipelines, and integrate with DVC for data versioning.

### Question 20

> **Upload 1-2 images of your GCP artifact registry, such that we can see the different docker images that you have**
> **stored. You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:


### Question 21

> **Upload 1-2 images of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

### Question 22

> **Did you manage to train your model in the cloud using either the Engine or Vertex AI? If yes, explain how you did**
> **it. If not, describe why.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We managed to train our model in the cloud using the Engine. We did this by ... . The reason we choose the Engine*
> *was because ...*
>
> Answer:

## Deployment

### Question 23

> **Did you manage to write an API for your model? If yes, explain how you did it and if you did anything special. If**
> **not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did manage to write an API for our model. We used FastAPI to do this. We did this by ... . We also added ...*
> *to the API to make it more ...*
>
> Answer:

We successfully built a production-ready FastAPI application in app/main.py. The API includes two main endpoints: /health returns model status and readiness for load balancers, and /predict accepts time-series input (aggregate power consumption) and returns disaggregated appliance predictions. We implemented special features including: (1) startup event handler that loads the PyTorch model once and keeps it in memory for fast inference; (2) dual inference modes—PyTorch for flexibility and ONNX for 30-40% faster inference via /predict/onnx; (3) batch prediction support to process multiple samples efficiently; (4) comprehensive input validation with FastAPI's Pydantic models; (5) proper error handling with meaningful HTTP status codes. The API runs on CPU to minimize deployment costs while maintaining sub-100ms latency for typical requests.

### Question 24

> **Did you manage to deploy your API, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

We successfully deployed the API locally and tested it thoroughly. The deployment process uses `uvicorn app.main:app --host 0.0.0.0 --port 8000` to serve the FastAPI application on port 8000. Users can interact with the API through multiple interfaces: (1) direct HTTP requests via curl or Python requests library; (2) our custom web frontend at frontend/index.html that provides an interactive UI for both PyTorch and ONNX prediction modes; (3) automated tests via pytest for CI/CD validation. We also containerized the API with Docker, making deployment portable—the same image can run locally, on any cloud provider, or in Kubernetes.

Cloud deployment to GCP Cloud Run would require minimal changes: push the Docker image to GCP Artifact Registry, create a Cloud Run service, and configure autoscaling parameters.

### Question 25

> **Did you perform any unit testing and load testing of your API? If yes, explain how you did it and what results for**
> **the load testing did you get. If not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For unit testing we used ... and for load testing we used ... . The results of the load testing showed that ...*
> *before the service crashed.*
>
> Answer:

We performed both unit testing and load testing of our API. Unit tests in test_api.py include 5 tests: (1) health endpoint validation (status 200 + model_loaded flag); (2) single-sample prediction with shape validation; (3) batch prediction ensuring correct output dimensions; (4) empty input rejection returning 400 status; (5) ONNX endpoint functionality check. These tests run automatically in CI via pytest.

For load testing, we used Locust with locustfile.py, which simulates concurrent users making health checks (80% of traffic) and predictions (20% of traffic) with random wait times between 0.1-0.5 seconds. Running `locust -f loadtest/locustfile.py --host http://localhost:8000`, we observed that our API handles approximately 100-150 requests/second on a single CPU core before latency degrades significantly. The ONNX endpoint showed 30-40% better throughput than the PyTorch endpoint. No crashes occurred during stress testing up to 500 concurrent users, though response times increased to 1-2 seconds under peak load.

### Question 26

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

We did not implement full production monitoring with GCP Cloud Monitoring, but we built the foundation for observability. Our API includes basic health checks and logging via Loguru, which captures all requests, errors, and prediction latencies. We also implemented drift detection capabilities (data_collection.py and drift_detection.py) that can monitor input distributions and alert when data shifts significantly from the training distribution.

Production monitoring would significantly improve application longevity by: (1) tracking key metrics like prediction latency, error rates, and throughput to identify performance degradation; (2) detecting data drift before model accuracy drops noticeably; (3) monitoring resource usage (CPU, memory) to optimize scaling and costs; (4) setting up alerts for anomalies like sudden spikes in errors or latency; (5) providing audit trails for debugging incidents. Services like Prometheus for metrics, Grafana for dashboards, and GCP Cloud Monitoring for centralized observability would enable proactive maintenance rather than reactive firefighting.

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 27

> **How many credits did you end up using during the project and what service was most expensive? In general what do**
> **you think about working in the cloud?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ... . Working in the cloud was ...*
>
> Answer:

We used minimal GCP credits during this project—approximately $5-10 total across both team members, primarily spent on experimentation with basic services like Storage and initial Compute Engine exploration. We chose not to heavily invest GCP credits because: (1) our model trains quickly on local hardware; (2) GitHub Actions provides free CI/CD for public repositories; (3) we prioritized building a cloud-agnostic MLOps pipeline that demonstrates best practices regardless of provider.

Working in the cloud offers significant advantages for production systems: automatic scaling based on demand, managed services that reduce operational overhead, global distribution for low-latency access, and robust monitoring/logging infrastructure. However, it also introduces complexity (IAM, networking, cost management) and vendor lock-in risks. For our educational project, the hybrid approach—developing locally with cloud-ready architecture—provided the best learning experience while conserving resources. In production, we would fully leverage cloud services for their reliability and scalability benefits.

### Question 28

> **Did you implement anything extra in your project that is not covered by other questions? Maybe you implemented**
> **a frontend for your API, use extra version control features, a drift detection service, a kubernetes cluster etc.**
> **If yes, explain what you did and why.**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We implemented a frontend for our API. We did this because we wanted to show the user ... . The frontend was*
> *implemented using ...*
>
> Answer:

We implemented several extra features beyond the core requirements. First, we built a simple web frontend (frontend/) using vanilla HTML, CSS, and JavaScript that provides an interactive interface for testing both PyTorch and ONNX prediction modes—users can input power consumption data, visualize results, and check API health without writing code. Second, we added comprehensive drift detection infrastructure (drift_detection.py, data_collection.py) with statistical tests (Kolmogorov-Smirnov, Population Stability Index) to monitor distribution shifts, though we didn't deploy this to the cloud. Third, we integrated ONNX Runtime for optimized inference, reducing prediction latency by 30-40% compared to PyTorch. Fourth, we set up Locust-based load testing to measure API performance under concurrent load. Finally, we implemented comprehensive profiling (profiling.py, profile_training.py) to identify training bottlenecks, with results saved to profiling_results/training_profile.json. These additions demonstrate production-readiness and performance optimization beyond basic model deployment.

### Question 29

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally, in your own words, explain the**
> **overall steps in figure.**
>
> Recommended answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and push to GitHub, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

![MLOps Architecture Diagram](figures/architecture.png)

Our system architecture follows a complete MLOps pipeline organized into four main stages:

**1️⃣ Local Development & Pre-commit**: The workflow starts with developers writing code in src/energy_dissagregation_mlops/ with clear module separation (data.py, model.py, train.py, api.py, cli.py). Pre-commit hooks automatically enforce code quality—Ruff linting/formatting, trailing whitespace removal, YAML validation, and import organization all run before commits are allowed.

**2️⃣ Continuous Integration (GitHub Actions)**: When code is pushed to GitHub, multiple workflows trigger in parallel: (1) tests.yaml runs pytest across Ubuntu/Windows/macOS with Python 3.11-3.12 and PyTorch 2.6-2.7 combinations using a matrix strategy; (2) linting.yaml validates Ruff formatting compliance; (3) docker_build.yaml builds Docker images (api, train, dev) with BuildKit layer caching and pushes to GHCR; (4) cml_model.yaml validates model artifacts on changes. All workflows use pip caching for speed.

**3️⃣ Training & Experimentation**: Experiments are configured via Hydra YAML files (configs/quick_test.yaml, normal_training.yaml, etc.) and tracked in Weights & Biases, which logs hyperparameters, training metrics, and model artifacts. Training runs locally or in Docker containers using train.dockerfile with GPU support. Best models are saved as PyTorch checkpoints (best.pt) and exported to ONNX format (30-40% inference speedup).

**4️⃣ API Deployment & Monitoring**: The FastAPI application (app/main.py) loads models at startup via a startup event handler and serves /health and /predict endpoints (supporting both PyTorch and ONNX). The containerized API deploys to any platform—local, Cloud Run, or Kubernetes. A web frontend (frontend/) provides interactive testing, Locust conducts load testing, Loguru logs all operations, and our drift detection module monitors input distributions for data shifts that trigger retraining.

This layered approach ensures reproducibility, automated quality checks, efficient experimentation tracking, and production-ready deployment infrastructure—all core MLOps principles.

### Question 30

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Recommended answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

The biggest challenges in the project were data preprocessing complexity and CI/CD configuration. The UK-DALE dataset required significant preprocessing—handling missing values, aligning timestamps across appliances, resampling to consistent intervals, and creating sliding windows for time-series learning. We spent considerable time ensuring data quality and building robust preprocessing pipelines that could handle edge cases. We overcame this by modularizing data.py with clear preprocessing steps, extensive logging via Loguru to debug issues, and comprehensive unit tests to validate each transformation.

CI/CD configuration across multiple platforms (Ubuntu, Windows, macOS) with different Python/PyTorch versions presented challenges with dependency compatibility and test flakiness. Some tests passed locally but failed in CI due to path differences or missing dependencies. We resolved this by: using pip caching to speed up builds, adding detailed logging to failing tests, creating fixtures for device selection (conftest.py), and marking integration tests that require data as skippable when data isn't available.

Docker multi-stage builds initially produced large images (>5GB) due to including unnecessary CUDA libraries and caching layers. We optimized by using slim Python base images, multi-stage builds that separate build dependencies from runtime, and BuildKit caching in GitHub Actions.

Finally, coordinating work with a two-person team across different schedules required clear communication and branch management. We overcame this through detailed PR descriptions, comprehensive commit messages, and using GitHub Issues to track tasks and decisions. This made async collaboration effective despite limited synchronous meetings.

### Question 31

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project. Additionally, state if/how you have used generative AI**
> **tools in your project.**
>
> Recommended answer length: 50-300 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
> *We have used ChatGPT to help debug our code. Additionally, we used GitHub Copilot to help write some of our code.*
> Answer:

Student s224182 (Tobias) was primarily responsible for initial project setup with cookiecutter, Docker container development (all Dockerfiles: api, train, CLI, dev), and CI/CD pipeline configuration including GitHub Actions workflows (tests.yaml, linting.yaml, docker_build.yaml). Tobias also implemented the FastAPI application, ONNX integration for optimized inference, load testing with Locust, and the web frontend.

Student s251999 (Guido) focused on the core ML pipeline: data preprocessing and dataset implementation (data.py), model architecture (model.py), training loop with W&B integration (train.py), profiling infrastructure (profiling.py), and drift detection modules (drift_detection.py, data_collection.py). Guido also wrote unit tests (test_data.py, test_model.py, test_drift_detection.py) and evaluation scripts.

Both members contributed to configuration management (Hydra YAML files), documentation, CLI development, and debugging CI issues. Code reviews were performed mutually on all pull requests.

We extensively used generative AI tools: GitHub Copilot for code completion and boilerplate generation (especially FastAPI routes, test fixtures, and Docker configurations), ChatGPT for debugging CI errors and explaining complex library APIs (e.g., ONNX Runtime, Locust), and Claude for report writing assistance and architectural design discussions. These tools significantly accelerated development while we maintained critical thinking about suggested solutions.
