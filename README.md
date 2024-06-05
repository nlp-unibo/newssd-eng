# A Corpus for Sentence-Level Subjectivity Detection on English News Articles

This repository contains the code and resources for the project ["A Corpus for Sentence-Level Subjectivity Detection on English News Articles"](https://aclanthology.org/2024.lrec-main.25.pdf).

## Abstract

_We develop novel annotation guidelines for sentence-level subjectivity detection, which are not limited to language-specific cues.
We use our guidelines to collect NewsSD-ENG, a corpus of 638 objective and 411 subjective sentences extracted from English news articles on controversial topics.
Our corpus paves the way for subjectivity detection in 
English and across other languages without relying on language-specific tools, 
such as lexicons or machine translation.
We evaluate state-of-the-art multilingual transformer-based models on the task in mono-, multi-, and cross-language settings. For this purpose, we re-annotate an existing Italian corpus. We observe that models trained in the multilingual setting achieve the best performance on the task._

## Data

The ``data`` folder contains train, val, and test splits for NewsSD-Eng (``english``) and NewsSD-Ita (``italian``).

## Guidelines

The ``data/guidelines.pdf`` file reports the annotation guidelines used to create NewsSD-Eng.

## Experiments

The ``run_tests.py`` and ``models/BertMultilingual.py`` report the code to reproduce our experiments.

### Usage

The ``run_tests.py`` script accepts the following arguments

- ``-trl | --train-language``: the language to consider for training. Supported: it|en|en+it
- ``-tl  | --test-language``: the language to consider for test. Supported it|en|en+it
- ``-m   | --model``: the model to use. Supported: SBERT | MBERT | SVM | LR.

### Examples

In-domain evaluation on NewsSD-Eng w/ Multilingual BERT (MBERT).

```bash
python run_tests.py --trl en --tl en -m MBERT
```

Cross-language evaluation on NewsSD-Eng (source language) and NewsSD-Ita (target language).

```bash
python run_tests.py --trl en --tl it -m MBERT
```

Multilingual evaluation

```bash
python run_tests.py --trl en+it --tl en+it -m MBERT
```

## Results

We publish all experimental results reported in the paper in ``results`` folder.

## Contact

Federico Ruggeri: federico.ruggeri6@unibo.it

Francesco Antici: francesco.antici@unibo.it

## Cite

You can cite our work as follows:

```
@inproceedings{antici-etal-2024-corpus-sentence,
    title = "A Corpus for Sentence-Level Subjectivity Detection on {E}nglish News Articles",
    author = "Antici, Francesco  and
      Ruggeri, Federico  and
      Galassi, Andrea  and
      Korre, Katerina  and
      Muti, Arianna  and
      Bardi, Alessandra  and
      Fedotova, Alice  and
      Barr{\'o}n-Cede{\~n}o, Alberto",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.25",
    pages = "273--285",
}

}
```

## Credits

Many thanks to Paolo Torroni for their valuable feedback.

Additional thanks to all reviewers for improving the manuscript.


## License

This project is licensed under CC.BY License.
