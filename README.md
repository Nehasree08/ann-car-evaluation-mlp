# ANN-Based Classification of Car Evaluation Using MLP

## Requirements
- Python 3.10 or newer recommended
- Internet connection for the first run (downloads the original UCI data)

## Setup (Windows)
Open Command Prompt or VS Code terminal in this folder:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python main.py
```

If `python` is not recognized, try `py` instead.

## Outputs
The run creates:
- `results/validation_experiments.csv` — validation metrics for all controlled experiments
- `results/experiment_comparison.png` — validation macro-F1 comparison chart
- `results/test_metrics.csv` — final held-out test metrics
- `results/classification_report.csv` — precision/recall/F1 by class
- `results/test_confusion_matrix.png` — confusion matrix
- `results/training_loss.png` — final model's training loss
- `results/test_predictions.csv` — actual and predicted labels for error analysis
- `results/run_metadata.json` — selected settings, split sizes, and package versions
- `report.md` — report populated with the results from this actual run

## Important methodology notes
- The split is stratified and fixed at 60% train / 20% validation / 20% test.
- The test set is not used to choose hyperparameters.
- The model uses `solver="sgd"`; backpropagation is used internally by MLPClassifier.
- Experiments vary one parameter at a time from the baseline.
- The best configuration is selected by validation macro-F1, with accuracy as a tie-breaker.
- Results are not included in advance because they must come from executing the experiments.

Dataset source: https://archive.ics.uci.edu/dataset/19/car+evaluation
