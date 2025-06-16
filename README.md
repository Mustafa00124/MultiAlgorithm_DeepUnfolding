# Multi-Algorithm Deep Unfolding (MADU)

This repository contains the official implementation of the **MADU** framework from the paper:

**"Multi-Algorithm Deep Unfolding"**  

> MADU increases the expressivity of deep unfolding models by integrating multiple algorithmic priors (low-rank, frequency, median) using a structured, interpretable architecture. It is evaluated on the task of foreground detection with CDnet2014 and demonstrates state-of-the-art results.

---

## Directory Structure

```
├── configs/
│   └── cdnet.py               # Configuration settings (paths, model params, etc.)
├── dataloader/
│   ├── generate_mnist.py
│   ├── moving_mnist_data_loader.py
│   ├── train-images-idx3-ubyte.gz
│   ├── utils.py
│   └── video_loader.py        # CDNet video loading logic
├── logs/                      # Output logs, visualizations, and checkpoints
├── madu.py                    # Main MADU model (multi-branch unfolding)
├── madu_layers.py             # Definition of MADU layer (L/M updates)
├── main.py                    # Main training + evaluation script
├── make_latex2.py             # Utility to generate LaTeX table of results
├── run_script_madu.bat        # Script to run MADU on selected categories/models
├── run_all_models.bat         # Batch script to run all baselines, ablations, and MADU
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

---

## Setup & Installation

1. **Install Python Requirements**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare the Dataset**
   * Download CDnet2014 and set the dataset path in `configs/cdnet.py` (`cfgs.data_path`)

---

## How to Run Experiments

The easiest way to run all experiments is using the batch script:

### Step-by-step Instructions

1. Open `run_all_models.bat`
2. Modify the models and seeds:

   ```bat
   set models=median svd dft Madu2 Serial2 Ensemble2
   set seeds=1 2 3 4 5
   ```
3. Run:

   ```bash
   run_all_models.bat
   ```

This will:

* Train and evaluate each model across selected CDNet2014 categories
* Log results in the `logs/` directory
* Save F1 scores and losses to CSV files
* Wait 20 minutes between model runs (configurable)

---

## Model Modes and Configurations

Each mode corresponds to a unique algorithmic configuration:

| Mode        | Description                                | Priors Used                   |
| ----------- | ------------------------------------------ | ----------------------------- |
| `Madu2`     | **Main setting used in the paper**         | Median + DFT + SVT (low-rank) |
| `median`    | Ablation using only median filtering       | Median                        |
| `dft`       | Ablation using only frequency thresholding | DFT                           |
| `svd`       | Ablation using only low-rank prior         | SVT                           |
| `Serial2`   | Sequentially applies priors                | Median → DFT → SVT            |
| `Ensemble2` | Output-level ensemble of the three priors  | Median + DFT + SVT            |

> In our paper, we used `Madu2` with `3` unfolding layers and a seed of `5` as the main evaluation setup.

---

## Exporting LaTeX Tables

After training, you can generate a LaTeX-formatted result table:

```bash
python make_latex2.py
```

This will scan the logs and output a summary table ready for your paper.

---

## Results Summary

MADU2 outperforms all baselines and ablation configurations on CDnet2014. It performs especially well in complex categories such as:

* Dynamic Background
* Bad Weather
* Turbulence
* Shadows

The structured, interpretable nature of MADU ensures robustness across variations without resorting to black-box designs.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{Siddiqui2025MADU,
  title={Multi-Algorithm Deep Unfolding},
  author={Mustafa Siddiqui and Muhammad Tahir},
  booktitle={...},
  year={2025}
}
```

---

For any questions or collaborations, please feel free to reach out.

