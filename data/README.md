## Data


This folder contains the datasets used in our work. Each folder includes four data files: train, in-domain test, OOD-easy, and OOD-hard.

```
data/
└── TASK/
    ├── train.jsonl
    ├── id_test.jsonl
    ├── ood_easy.jsonl
    └── ood_hard.jsonl
```

[//]: # (<img src="../static/tasks.png" alt="Alt text" width="500"/>)
<div style="width:80%; text-align:center">
    <figure >
        <img src="../static/tasks.png">
        <figcaption>
            Task description
        </figcaption>
    </figure>
</div>

### Statistics
| Task                        | Train | In-domain test | OOD-easy | OOD-hard |
|-----------------------------|-------|----------------|----------|----------|
| Analog of Algebra (AOA)     | 5,770 | 1,000          | 2,000    | 420      |
| Multi-digit Addition (MA)   | 2,885 | 1,000          | 1,200    | 1,600    |
| Directional Reasoning (DR)  | 2,080 | 1,000          | 500      | 500      |

