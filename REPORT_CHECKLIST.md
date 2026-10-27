# Report submission checklist

Run `python main.py` first. It generates `report.md` with real metrics from your run.

Before submitting:
1. Confirm the command completed without errors.
2. Check `results/run_metadata.json` for the selected configuration and convergence warning.
3. Include `results/experiment_comparison.png`, `results/training_loss.png`, and `results/test_confusion_matrix.png` in the report.
4. Discuss which classes the model confuses using the confusion matrix and `results/test_predictions.csv`.
5. Do not claim the model is suitable for real automotive safety decisions; this dataset is a small benchmark.
