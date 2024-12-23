# <font color='Aqua'><b> Hateful Memes Challenge-Team HateDetectron Submissions </b></font>

![GitHub Repo stars](https://img.shields.io/github/stars/rizavelioglu/hateful_memes-hate_detectron?style=social)
![GitHub forks](https://img.shields.io/github/forks/rizavelioglu/hateful_memes-hate_detectron?style=social)
![GitHub](https://img.shields.io/github/license/rizavelioglu/hateful_memes-hate_detectron)
![GitHub repo size](https://img.shields.io/github/repo-size/rizavelioglu/hateful_memes-hate_detectron)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/detecting-hate-speech-in-memes-using/meme-classification-on-hateful-memes)](https://paperswithcode.com/sota/meme-classification-on-hateful-memes?p=detecting-hate-speech-in-memes-using)

Check out the paper on [![arXiv](https://img.shields.io/badge/arXiv-2012.12975-b31b1b.svg)](https://arxiv.org/abs/2012.12975) 
and check out my [![thesis](https://img.shields.io/badge/website-MSc.Thesis-lightgreen)](https://rizavelioglu.github.io/publication/2021-04-msc-thesis)
which offers an in-depth analysis of the approach as well as an overview of Multimodal Research and its foundations.

This repository contains *all* the code used at the [Hateful Memes Challenge](https://ai.facebook.com/tools/hatefulmemes/) by Facebook AI. There are 2 main Jupyter notebooks where all the job is done and documented:
- The *'reproducing results'* notebook --> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1kAYFd50XvFnLO-k9FU9iLM21J8djTo-Q?usp=sharing)
- The *'end-to-end'* notebook --> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1O0m0j9_NBInzdo3K04jD19IyOhBR1I8i?usp=sharing)

The first notebook is only for reproducing the results of Phase-2 submissions by the team `HateDetectron`. In other 
words, just loading the final models and getting predictions for the test set. See the [*end-to-end* notebook](https://colab.research.google.com/drive/1O0m0j9_NBInzdo3K04jD19IyOhBR1I8i?usp=sharing) 
to have a look at the whole approach in detail: how the models are trained, how the image features are extracted, which datasets are used, etc.

---
<h2><b> About the Competition </b></h2>
  The Hateful Memes Challenge and Data Set is a competition and open source data set designed to measure progress in 
multimodal vision-and-language classification.

  Check out the following sources to get more on the challenge:
  - [Facebook AI](https://ai.facebook.com/tools/hatefulmemes/)
  - [DrivenData](https://www.drivendata.org/competitions/64/hateful-memes/)
  - [Competition Paper](https://arxiv.org/pdf/2005.04790.pdf)

<h3><b> Competition Results: </b></h3>
  We are placed the <b>3rd</b> out of <b>3.173</b> participants in total!

  See the official Leaderboard [here!](https://www.drivendata.org/competitions/70/hateful-memes-phase-2/leaderboard/)

---

<h2><b> Repository structure </b></h2>
  The repository consists of the following folders:

  <details>
  <summary><b><i> hyperparameter_sweep/ </i></b>: where scripts for hyperparameter search are.</summary>

  - `get_27_models.py`: iterates through the folders those that were created for hyperparameter search
    and collects the metrics (ROC-AUC, accuracy) on the 'dev_unseen' set and stores them in a pd.DataFrame. Then, it sorts the models according to AUROC metric and moves the best 27 models into a generated folder `majority_voting_models/`
  - `remove_unused_file.py`: removes unused files, e.g. old checkpoints, to free the disk.
  - `sweep.py`: defines the hyperparameters and starts the process by calling `/sweep.sh`
  - `sweep.sh`: is the mmf cli command to do training on a defined dataset, parameters, etc.

  </details>


  <details>
  <summary><b><i> notebooks/ </i></b>: where Jupyter notebooks are stored.</summary>

  - `[GitHub]end2end_process.ipynb`: presents the whole approach end-to-end: expanding data, image feature extraction, hyperparameter search, fine-tuning, majority voting.
  - `[GitHub]reproduce_submissions.ipynb`: loads our fine-tuned (final) models and generates predictions.
  - `[GitHub]label_memotion.ipynb`: a notebook which uses `/utils/label_memotion.py` to label memes from Memotion and to save it in an appropriate form.
  - `[GitHub]simple_model.ipynb`: includes a simple multimodal model implementation, also known as 'mid-level concat fusion'. We train the model and generate submission for the challenge test set.
  - `[GitHub]benchmarks.ipynb`: reproduces the benchmark results.

  </details>


  <details><summary><b><i> utils/ </i></b>: where some helper scripts are stored, such as labeling Memotion Dataset and merging the two datasets.</summary>

  - `concat_memotion-hm.py`: concatenates the labeled memotion samples and the hateful memes samples and saves them in a new `train.jsonl` file.
  - `generate_submission.sh`: generates predictions for 'test_unseen' set (phase 2 test set).
  - `label_memotion.jsonl`: presents the memes labeled by us from memotion dataset.
  - `label_memotion.py`: is the script for labelling Memotion Dataset. The script iterates over the samples in Memotion and labeler labels the samples by entering 1 or 0 on the keyboard. The labels and the sample metadata is saved at the end as a `label_memotion.jsonl`.

  </details>


---