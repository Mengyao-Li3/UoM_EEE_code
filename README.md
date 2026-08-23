# UoM_EEE — PhysioNet/CinC Challenge 2023

Research code developed by the **UoM_EEE** team for the **George B. Moody PhysioNet Challenge 2023: Predicting Neurological Recovery from Coma After Cardiac Arrest**.

The repository contains the Python/TensorFlow implementation used to investigate EEG preprocessing, time-frequency representations, convolutional neural networks, and the effect of autoencoder-based artefact removal on downstream neurological outcome prediction.

**Authors:** Mengyao Li, Le Xing, Alexander J. Casson

---

## Project overview

The PhysioNet Challenge 2023 focused on predicting neurological recovery following cardiac arrest from physiological recordings.

Our approach used EEG data to develop deep-learning models for:

* binary neurological outcome prediction (good vs. poor outcome);
* five-class Cerebral Performance Category (CPC) prediction;
* EEG preprocessing and bipolar montage construction;
* time-frequency representation using the Short-Time Fourier Transform (STFT); and
* investigation of autoencoder-based EEG artefact removal as an additional preprocessing stage.

A key part of the research was an **evidence-based comparison of model performance with and without autoencoder-based artefact removal during validation**. The validation results were used to determine the preprocessing configuration for the final evaluation pipeline.

---

## Research workflow

The overall workflow can be summarised as:

```text
EEG recordings
      |
      v
EEG channel selection
      |
      v
Six bipolar EEG derivations
      |
      v
Filtering / resampling / normalization
      |
      +-----------------------------+
      |                             |
      v                             v
Without autoencoder          With autoencoder
      |                       artefact removal
      |                             |
      +-------------+---------------+
                    |
                    v
          Validation comparison
          with vs. without
          autoencoder preprocessing
                    |
                    v
          Selected final pipeline
                    |
                    v
                  STFT
                    |
                    v
       Six-channel time-frequency
              representation
                    |
                    v
                  CNN
             /            \
            v              v
     Outcome prediction   CPC prediction
```

---

## EEG preprocessing

The implementation uses six EEG channels:

* F3
* T3
* P3
* F4
* T4
* P4

These are converted into six bipolar derivations:

1. F3–T3
2. T3–P3
3. F3–P3
4. F4–T4
5. T4–P4
6. F4–P4

The preprocessing pipeline includes:

* notch filtering when the utility frequency lies within the passband;
* band-pass filtering from **0.1 to 30 Hz**;
* resampling to **128 Hz** for recordings with even original sampling frequencies and **125 Hz** for odd original sampling frequencies; and
* scaling to approximately **[-1, 1]** using min-max normalization.

Signal processing is implemented using **MNE-Python** and **SciPy**.

---

## Exploration of autoencoder-based artefact removal

An autoencoder-based EEG artefact-removal approach was explored to investigate the hypothesis that removing EEG artefacts could improve downstream machine-learning performance.

During the validation stage, model performance was compared under two conditions:

1. EEG processed with autoencoder-based artefact removal; and
2. EEG processed without autoencoder-based artefact removal.

The purpose of these experiments was exploratory: to assess whether autoencoder-based artefact removal could improve the performance of the downstream neurological outcome prediction model.

Based on the validation results, the hypothesis was not supported sufficiently to justify incorporating the autoencoder into the final evaluation pipeline. Consequently, the final evaluation was performed using the EEG preprocessing pipeline without autoencoder-based artefact removal.

The autoencoder implementation used for these exploratory experiments is available from:

https://github.com/Non-Invasive-Bioelectronics-Lab/Modified_Autoencoder4_challenge

To reproduce the exploratory validation experiments involving autoencoder-based artefact removal, download the folder `Autoencoder_Mengyao_Challenge` from the repository above and place it in the same directory as `team_code.py`.

The expected directory structure is:

```text
.
├── team_code.py
├── helper_code.py
├── Autoencoder_Mengyao_Challenge/
│   └── ...
├── train_model.py
├── run_model.py
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Time-frequency representation

To provide a two-dimensional representation suitable for convolutional neural networks, the Short-Time Fourier Transform (STFT) is applied independently to each bipolar EEG channel.

The resulting representations are resized to **128 × 128** and stacked along the channel dimension.

The final CNN input therefore contains:

* height: 128
* width: 128
* channels: 6

The six input channels correspond to the six bipolar EEG derivations listed above.

---

## Deep-learning models

The repository implements convolutional neural networks using **TensorFlow/Keras**.

Two prediction tasks are considered:

### Neurological outcome prediction

Binary classification of:

* good neurological outcome
* poor neurological outcome

The outcome model uses **Binary Focal Cross-Entropy** and **AUC** as the main training/evaluation metric.

### Cerebral Performance Category prediction

Five-class prediction of:

* CPC 1
* CPC 2
* CPC 3
* CPC 4
* CPC 5

The CPC model uses a five-class softmax output.

The CNN contains progressively wider convolutional blocks followed by fully connected layers and dropout.

---

## Training and validation

The training implementation includes:

* stratified training/validation splitting;
* one-hot encoding of target labels;
* Adam optimization;
* early stopping;
* model checkpointing;
* repeated training trials; and
* validation-based model selection.

The configured training procedure uses up to **150 epochs** with a **batch size of 32**, with early stopping used to terminate training when validation loss no longer improves.

For the binary outcome model, AUC is monitored during training and validation.

The validation experiments were used to compare alternative preprocessing/model configurations, including the presence or absence of autoencoder-based artefact removal.

---

## Repository structure

| File                | Description                                                                                                                                                                     |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `team_code.py`      | Main research implementation, including preprocessing, autoencoder integration for validation experiments, STFT generation, CNN construction, training, and Challenge inference |
| `helper_code.py`    | PhysioNet Challenge data-loading and utility functions                                                                                                                          |
| `train_model.py`    | Challenge training entry point                                                                                                                                                  |
| `run_model.py`      | Challenge inference entry point                                                                                                                                                 |
| `evaluate_model.py` | Model evaluation utilities                                                                                                                                                      |
| `example_code.py`   | Example Challenge implementation                                                                                                                                                |
| `remove_data.py`    | Utility for removing data                                                                                                                                                       |
| `remove_labels.py`  | Utility for removing labels                                                                                                                                                     |
| `truncate_data.py`  | Utility for truncating data                                                                                                                                                     |
| `requirements.txt`  | Python package dependencies                                                                                                                                                     |
| `Dockerfile`        | Container configuration for the computational environment                                                                                                                       |
| `AUTHORS.txt`       | Author information                                                                                                                                                              |
| `LICENSE`           | Repository license                                                                                                                                                              |

---

## Software requirements

The main software dependencies include:

* Python 3.10
* NumPy
* SciPy
* scikit-learn
* MNE-Python
* TensorFlow / Keras
* OpenCV
* Matplotlib
* joblib

The package versions used by the project are specified in `requirements.txt`.

---

## Running the Challenge code

The repository follows the standard PhysioNet Challenge framework.

### Training

```bash
python train_model.py <data_folder> <model_folder>
```

For example:

```bash
python train_model.py data model
```

### Inference

```bash
python run_model.py <model_folder> <data_folder> <output_folder>
```

For example:

```bash
python run_model.py model data outputs
```

The Challenge framework invokes the corresponding functions implemented in `team_code.py`.

---

## Docker

A Dockerfile is provided to support a reproducible computational environment.

The Docker configuration installs the Python dependencies specified in `requirements.txt` and retrieves the autoencoder repository used for the validation-stage artefact-removal experiments.

**Important:** no authentication token or other credential is required in the Dockerfile. The autoencoder repository is referenced using its public GitHub URL.

---

## Data

The Challenge datasets are **not included in this repository**.

The code is intended to operate on data provided through the PhysioNet/CinC Challenge 2023 framework. Users should obtain and use the data according to the applicable PhysioNet and Challenge access, licensing, and usage requirements.

---

## Research contribution

This project demonstrates a biomedical machine-learning workflow involving:

- physiological signal processing;
- EEG analysis;
- bipolar montage construction;
- time-frequency representations;
- convolutional neural networks;
- clinical outcome prediction;
- validation-based model comparison; and
- exploratory investigation of autoencoder-based EEG artefact removal.

A particular focus of the work was exploring the hypothesis that autoencoder-based artefact removal could improve downstream prediction performance. The hypothesis was evaluated empirically during validation by comparing models with and without the additional artefact-removal stage. The validation results informed the selection of the simpler preprocessing configuration used for final evaluation.

---

## Related publication

**Li, M., Xing, L., & Casson, A. J. (2023).**
*Autoencoder Artefact Removal for Brain Signals and Impact on Classification Performance.*
Computing in Cardiology, 50, 1–4.

DOI: https://doi.org/10.22489/CinC.2023.217

The paper describes the development of the six-channel EEG/STFT-based CNN approach and the investigation of autoencoder-based artefact removal for the PhysioNet Challenge 2023.

The UoM_EEE team was listed among the Challenge papers for this work.

---

## Authors

**Mengyao Li**，
**Le Xing**，
**Alexander J. Casson**

Team: **UoM_EEE**

---

## License

Please refer to `LICENSE` for the licensing terms associated with this repository.

The Challenge data and any external dependencies remain subject to their respective licenses and usage conditions.
