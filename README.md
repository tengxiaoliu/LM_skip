# LM_skip

Implementation for NeurIPS 2024 [paper]():

Can Language Models Learn to Skip Steps?

<div style="width:80%; text-align:center">
    <figure >
        <img src="static/intro.png">
    </figure>
</div>

## Resources

### Data
The data folder contains the datasets used in our work. Each folder includes four data files: train, in-domain test, OOD-easy, and OOD-hard.

```
data/
└── TASK/
    ├── train.jsonl
    ├── id_test.jsonl
    ├── ood_easy.jsonl
    └── ood_hard.jsonl
```

## Setup

Install all the dependencies in the requirements.txt file:
```
pip -r requirements.txt
```

## Quick Start

To run the task in a single command, use the following script:
```
TASK="aoa"
bash scripts/${TASK}/run_${TASK}.sh
```