<div align="center">

# Prompted Aspect KPA (PAKPA): Quantitative Review Summarization

</div>

This repository maintains the code, data, and model checkpoints for the paper *Prompted Aspect Key Point Analysis for Quantitative Review Summarization*

[//]: # (# Code to release soon.)

## Installation
### 1. Create new virtual environment by
```bash
conda env create -f environment.yml
conda activate abkpa
```
### 2. Install Pytorch
For other versions, please visit: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
### Linux
##### Using GPUs
```bash
pip3 install torch torchvision torchaudio
conda install -c anaconda cudatoolkit
```
#### Using CPU
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Framework
![Model architecture](Abstractive_KPA_Diagram.png)

We propose a pipeline framework, namely Prompted Aspect Key Point Analysis (PAKPA), for quantitative opinion summarization for business reviews. 
PAKPA employ aspect sentiment analysis to identify aspects in comments as the opinion target and then  generate and quantify KPs grounded in aspects and their sentiment. 
The below figure illustrates PAKPA framework with examples.
Given reviews for a business entity, 
PAKPA performs KPA for reviews and 
generates KPs of distinctive aspects and quantities measuring the prevalence of KPs. PAKPA consists of three components/stages: 
- **Stage 1: Prompted ABSA of Comments**
- **Stage 2: Aspect-Sentiment-based Comment Clustering**
- **Stage 3: Prompted Aspect KP Generation**

## Dataset
We released both the Yelp and SPACE datasets used in our evaluation tasks (dimensions) of KPs. Datasets can be accessed under the ```data/``` folder, 
following the [```yelp/```](/data/yelp) and [```space/```](/data/space) subdirectories for each dataset.

[//]: # (In each dataset directory, )

[//]: # (- ```input_reviews/``` contains the raw input dataset file and also samples from the dataset for experiment)

[//]: # (- ```process_absa_reviews/``` consists of reviews already analyzed for ABSA predictions &#40;Stage 1&#41;)

[//]: # (- ```review_comments_clustered/``` consists of clusters of comments already clustered by their similar aspect terms &#40;Stage 2&#41;)

[//]: # (- ```summaries/``` consists of the final KP summaries produced by the framework &#40;Stage 3&#41;)

Files in each folder:
* ```input_reviews.pkl```: Raw input samples from the dataset for experiment.
* ```reviews_absa_processed.pkl```: Reviews already analyzed for ABSA predictions &#40;Stage 1&#41;
* ```reviews_clustered.pkl```: clusters of comments already clustered by their similar aspect terms &#40;Stage 2&#41;)
* ```summaries.pkl```: final KP summaries produced for the clusters by the framework &#40;Stage 3&#41;)

## Inference
We offer two options to perform inference of our pipeline framework (PAKPA), using Jupyter Notebook files (```notebook```) or Python inference scripts (```script```). 
Each file in the respective folder represent a stage of the pipeline. Below is the directory structures for stages of the pipeline:
```
notebook
├── space (or) yelp
│   ├── Prompted_ABSA.ipynb
│   ├── Aspect_Sentiment_Comment_Clustering.ipynb
│   ├── KP_Generation.ipynb
script
├── prompted_absa.py
├── aspect_sentiment_comment_clustering.py
├── kp_generation.py
```

### Hosting an LLaMas model for Prompted ABSA
We used [FastChat](https://github.com/lm-sys/FastChat/tree/main) to host a LLaMas for prompted in-context learning of ABSA.
FastChat provides OpenAI-compatible APIs for its supported models, so you can use FastChat as a local drop-in replacement for OpenAI APIs.
The FastChat server is compatible with both [openai-python](https://github.com/openai/openai-python) library and cURL commands.

**Note:** *Vicuna-7B is used as the LLMs for Prompted ABSA. This model requires around 14GB of GPU memory.
If you do not have enough memory, you can enable 8-bit compression by adding --load-8bit to commands below. This can reduce memory usage by around half with slightly degraded model quality.
For more information to reduce memory requirement or instruction to run on other architectures, please see this [link](https://github.com/lm-sys/FastChat/tree/main?tab=readme-ov-file#inference-with-command-line-interface)*

First, launch the controller

```bash
python3 -m fastchat.serve.controller
```

Then, launch the model worker(s)

```bash
python3 -m fastchat.serve.model_worker --model-path lmsys/vicuna-7b-v1.3 --load-8bit
```

Finally, launch the RESTful API server

```bash
python3 -m fastchat.serve.openai_api_server --host localhost --port 8000
```

[//]: # (## Inference Scripts)

[//]: # (### Prompted ABSA)

[//]: # (### Aspect-Sentiment-based Comment Clustering)

[//]: # (### KP Generation)

[//]: # (# Evaluation of KP Textual Quality with Aspect-Specific Ground Truth)

# More code coming soon.