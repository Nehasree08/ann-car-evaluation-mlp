
from pathlib import Path
from copy import deepcopy
import urllib.request
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    log_loss,
)

# ============================================================
# ANN-BASED CLASSIFICATION OF CAR EVALUATION USING MLP
# Complete project implementation
# ============================================================

SEED = 42
MAX_EPOCHS = 800
PATIENCE = 50

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "buying",
    "maint",
    "doors",
    "persons",
    "lug_boot",
    "safety",
]
TARGET = "class"

CLASS_NAMES = ["unacc", "acc", "good", "vgood"]

DATA_URLS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data",
    "https://archive.ics.uci.edu/static/public/19/car+evaluation.zip",
]


# ============================================================
# 1. LOAD DATASET
# ============================================================

def load_dataset():
    """Load the UCI Car Evaluation dataset."""

    columns = FEATURES + [TARGET]
    last_error = None

    for url in DATA_URLS:
        try:
            print(f"Loading dataset from: {url}")

            if url.endswith(".zip"):
                import io
                import zipfile

                with urllib.request.urlopen(url, timeout=30) as response:
                    zip_bytes = response.read()

                with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
                    data_file = next(
                        name for name in archive.namelist()
                        if name.lower().endswith("car.data")
                    )
                    with archive.open(data_file) as file:
                        df = pd.read_csv(file, header=None, names=columns)
            else:
                df = pd.read_csv(url, header=None, names=columns)

            if df.shape[1] == 7 and len(df) > 0:
                df = df.dropna().copy()
                print(f"Dataset loaded successfully: {df.shape[0]} rows")
                return df

        except Exception as exc:
            last_error = exc
            print(f"Could not load this source: {exc}")

    raise RuntimeError(
        "Could not download the dataset. Check your internet connection. "
        f"Last error: {last_error}"
    )


df = load_dataset()

# Keep only the expected features and target.
df = df[FEATURES + [TARGET]].copy()

# Remove incomplete rows if any are present.
df = df.dropna()

# Ensure text columns are consistent.
for column in FEATURES + [TARGET]:
    df[column] = df[column].astype(str).str.strip()

# Save a copy of the loaded dataset.
df.to_csv(RESULTS_DIR / "dataset_used.csv", index=False)

print("\nDataset shape:", df.shape)
print("\nClass distribution:")
print(df[TARGET].value_counts().reindex(CLASS_NAMES, fill_value=0))


# ============================================================
# 2. DATASET CLASS-DISTRIBUTION GRAPH
# ============================================================

class_counts = (
    df[TARGET]
    .value_counts()
    .reindex(CLASS_NAMES, fill_value=0)
)

plt.figure(figsize=(8, 5))
plt.bar(class_counts.index, class_counts.values)
plt.title("Car Evaluation Dataset Class Distribution")
plt.xlabel("Car Evaluation Class")
plt.ylabel("Number of Samples")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "dataset_distribution.png", dpi=300)
plt.close()


# ============================================================
# 3. STRATIFIED TRAIN / VALIDATION / TEST SPLIT
# ============================================================

X = df[FEATURES]
y = df[TARGET]

# First, reserve 20% for the final test set.
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
    stratify=y,
)

# Split the remaining 80% into 60% train and 20% validation
# relative to the complete dataset.
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.25,
    random_state=SEED,
    stratify=y_train_val,
)

print("\nDataset split:")
print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))


# ============================================================
# 4. ONE-HOT ENCODING
# ============================================================

# Fit the encoder using only training data for the experiments.
# The same encoder transforms validation and test data.
try:
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )
except TypeError:
    # Compatibility with older scikit-learn versions.
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse=False,
    )

X_train_encoded = encoder.fit_transform(X_train)
X_val_encoded = encoder.transform(X_val)
X_test_encoded = encoder.transform(X_test)

print("\nEncoded input feature count:", X_train_encoded.shape[1])

# The class list is passed explicitly to partial_fit.
classes = np.array(CLASS_NAMES)


# ============================================================
# 5. EXPERIMENT CONFIGURATIONS
# ============================================================

EXPERIMENTS = [
    {
        "experiment": "E1_baseline",
        "hidden_layer_sizes": (32,),
        "learning_rate_init": 0.01,
        "batch_size": 32,
    },
    {
        "experiment": "E2_lr_0.001",
        "hidden_layer_sizes": (32,),
        "learning_rate_init": 0.001,
        "batch_size": 32,
    },
    {
        "experiment": "E2_lr_0.05",
        "hidden_layer_sizes": (32,),
        "learning_rate_init": 0.05,
        "batch_size": 32,
    },
    {
        "experiment": "E3_batch_16",
        "hidden_layer_sizes": (32,),
        "learning_rate_init": 0.01,
        "batch_size": 16,
    },
    {
        "experiment": "E3_batch_64",
        "hidden_layer_sizes": (32,),
        "learning_rate_init": 0.01,
        "batch_size": 64,
    },
    {
        "experiment": "E4_arch_16",
        "hidden_layer_sizes": (16,),
        "learning_rate_init": 0.01,
        "batch_size": 32,
    },
    {
        "experiment": "E4_arch_64-32",
        "hidden_layer_sizes": (64, 32),
        "learning_rate_init": 0.01,
        "batch_size": 32,
    },
]


# ============================================================
# 6. METRIC HELPER
# ============================================================

def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
    }


# ============================================================
# 7. TRAIN ONE EXPERIMENT AND RECORD EPOCH CURVES
# ============================================================

def train_experiment(config, Xtr, ytr, Xva, yva):
    """Train an MLP incrementally and track validation metrics."""

    model = MLPClassifier(
        hidden_layer_sizes=config["hidden_layer_sizes"],
        activation="relu",
        solver="sgd",
        learning_rate="adaptive",
        learning_rate_init=config["learning_rate_init"],
        batch_size=config["batch_size"],
        momentum=0.9,
        nesterovs_momentum=True,
        max_iter=1,
        shuffle=True,
        random_state=SEED,
        early_stopping=False,
        tol=0.0,
    )

    history = {
        "epoch": [],
        "training_accuracy": [],
        "validation_accuracy": [],
        "training_loss": [],
        "validation_loss": [],
        "validation_macro_f1": [],
    }

    best_model = None
    best_f1 = -1.0
    best_accuracy = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        # Each partial_fit call performs one pass through the data.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if epoch == 1:
                model.partial_fit(Xtr, ytr, classes=classes)
            else:
                model.partial_fit(Xtr, ytr)

        train_predictions = model.predict(Xtr)
        val_predictions = model.predict(Xva)

        train_probabilities = model.predict_proba(Xtr)
        val_probabilities = model.predict_proba(Xva)

        train_acc = accuracy_score(ytr, train_predictions)
        val_metrics = calculate_metrics(yva, val_predictions)

        train_loss = log_loss(
            ytr, train_probabilities, labels=classes
        )
        val_loss = log_loss(
            yva, val_probabilities, labels=classes
        )

        history["epoch"].append(epoch)
        history["training_accuracy"].append(train_acc)
        history["validation_accuracy"].append(
            val_metrics["accuracy"]
        )
        history["training_loss"].append(train_loss)
        history["validation_loss"].append(val_loss)
        history["validation_macro_f1"].append(
            val_metrics["macro_f1"]
        )

        # Select the best checkpoint by validation macro-F1.
        # Accuracy is used as a tie-breaker.
        current_f1 = val_metrics["macro_f1"]
        current_accuracy = val_metrics["accuracy"]

        improved = (
            current_f1 > best_f1 + 1e-12
            or (
                abs(current_f1 - best_f1) <= 1e-12
                and current_accuracy > best_accuracy
            )
        )

        if improved:
            best_f1 = current_f1
            best_accuracy = current_accuracy
            best_epoch = epoch
            best_model = deepcopy(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch % 50 == 0:
            print(
                f"  Epoch {epoch:3d} | "
                f"train loss={train_loss:.4f} | "
                f"val accuracy={val_metrics['accuracy']:.4f} | "
                f"val macro-F1={val_metrics['macro_f1']:.4f}"
            )

        if epochs_without_improvement >= PATIENCE:
            print(f"  Stopped after {epoch} epochs (patience reached).")
            break

    history_df = pd.DataFrame(history)

    return best_model, history_df, best_epoch


# ============================================================
# 8. RUN ALL EXPERIMENTS
# ============================================================

experiment_rows = []
experiment_histories = {}
experiment_models = {}

for config in EXPERIMENTS:
    name = config["experiment"]

    print("\n" + "=" * 65)
    print("Running experiment:", name)
    print("Architecture:", config["hidden_layer_sizes"])
    print("Learning rate:", config["learning_rate_init"])
    print("Batch size:", config["batch_size"])

    best_model, history_df, best_epoch = train_experiment(
        config,
        X_train_encoded,
        y_train,
        X_val_encoded,
        y_val,
    )

    validation_predictions = best_model.predict(X_val_encoded)
    metrics = calculate_metrics(y_val, validation_predictions)

    row = {
        "experiment": name,
        "hidden_layer_sizes": str(config["hidden_layer_sizes"]),
        "learning_rate": config["learning_rate_init"],
        "batch_size": config["batch_size"],
        "epochs": best_epoch,
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "auc": np.nan,
    }

    experiment_rows.append(row)
    experiment_histories[name] = history_df
    experiment_models[name] = best_model

    history_df.to_csv(
        RESULTS_DIR / f"{name}_history.csv",
        index=False,
    )

    print("Best epoch:", best_epoch)
    print("Validation metrics:", metrics)


# Save all experiment results.
results_df = pd.DataFrame(experiment_rows)
results_df.to_csv(
    RESULTS_DIR / "experiment_results.csv",
    index=False,
)

print("\nExperiment results:")
print(results_df.to_string(index=False))


# ============================================================
# 9. SELECT THE BEST MODEL USING VALIDATION METRICS
# ============================================================

# Macro-F1 is the primary selection metric.
# Accuracy breaks ties.
results_df = results_df.sort_values(
    by=["macro_f1", "accuracy"],
    ascending=[False, False],
).reset_index(drop=True)

best_experiment_name = results_df.loc[0, "experiment"]
best_config = next(
    config for config in EXPERIMENTS
    if config["experiment"] == best_experiment_name
)

best_validation_model = experiment_models[best_experiment_name]
best_history = experiment_histories[best_experiment_name]

print("\nSelected experiment:", best_experiment_name)
print("Selected configuration:", best_config)
print(
    "Validation accuracy:",
    f"{results_df.loc[0, 'accuracy']:.4f}",
)
print(
    "Validation macro-F1:",
    f"{results_df.loc[0, 'macro_f1']:.4f}",
)


# ============================================================
# 10. FIGURE 4: TRAINING AND VALIDATION ACCURACY
# ============================================================

plt.figure(figsize=(9, 5))
plt.plot(
    best_history["epoch"],
    best_history["training_accuracy"],
    label="Training Accuracy",
)
plt.plot(
    best_history["epoch"],
    best_history["validation_accuracy"],
    label="Validation Accuracy",
)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "training_validation_accuracy.png",
    dpi=300,
)
plt.close()


# ============================================================
# 11. FIGURE 5: TRAINING AND VALIDATION LOSS
# ============================================================

plt.figure(figsize=(9, 5))
plt.plot(
    best_history["epoch"],
    best_history["training_loss"],
    label="Training Loss",
)
plt.plot(
    best_history["epoch"],
    best_history["validation_loss"],
    label="Validation Loss",
)
plt.xlabel("Epoch")
plt.ylabel("Log Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "training_validation_loss.png",
    dpi=300,
)
plt.close()

# Also save the training-loss-only graph.
plt.figure(figsize=(9, 5))
plt.plot(
    best_history["epoch"],
    best_history["training_loss"],
    label="Training Loss",
)
plt.xlabel("Epoch")
plt.ylabel("Log Loss")
plt.title("Final MLP Training Loss Curve")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "training_loss.png", dpi=300)
plt.close()


# ============================================================
# 12. FIGURE 7: VALIDATION MACRO-F1 EXPERIMENT COMPARISON
# ============================================================

plot_df = pd.DataFrame(experiment_rows)

plt.figure(figsize=(12, 6))
bars = plt.bar(
    plot_df["experiment"],
    plot_df["macro_f1"],
)

plt.ylim(0, 1.05)
plt.xlabel("Experiment")
plt.ylabel("Validation macro-F1")
plt.title("Controlled Experiments: Validation Macro-F1")
plt.xticks(rotation=25, ha="right")

for bar, value in zip(bars, plot_df["macro_f1"]):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f"{value:.3f}",
        ha="center",
        va="bottom",
    )

plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "experiment_comparison.png",
    dpi=300,
)
plt.close()


# ============================================================
# 13. FINAL MODEL: REFIT ON TRAINING + VALIDATION DATA
# ============================================================

# The test set is not used for model selection.
# Refit preprocessing on the combined training/validation data.
X_final_train = pd.concat([X_train, X_val], axis=0)
y_final_train = pd.concat([y_train, y_val], axis=0)

try:
    final_encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )
except TypeError:
    final_encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse=False,
    )

X_final_train_encoded = final_encoder.fit_transform(X_final_train)
X_final_test_encoded = final_encoder.transform(X_test)

final_model = MLPClassifier(
    hidden_layer_sizes=best_config["hidden_layer_sizes"],
    activation="relu",
    solver="sgd",
    learning_rate="adaptive",
    learning_rate_init=best_config["learning_rate_init"],
    batch_size=best_config["batch_size"],
    momentum=0.9,
    nesterovs_momentum=True,
    max_iter=1,
    shuffle=True,
    random_state=SEED,
    early_stopping=False,
    tol=0.0,
)

# Train for the selected experiment's best epoch count.
final_epochs = int(
    results_df.loc[
        results_df["experiment"] == best_experiment_name,
        "epochs",
    ].iloc[0]
)

print("\nRefitting final model on training + validation data...")
print("Final training epochs:", final_epochs)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for epoch in range(final_epochs):
        if epoch == 0:
            final_model.partial_fit(
                X_final_train_encoded,
                y_final_train,
                classes=classes,
            )
        else:
            final_model.partial_fit(
                X_final_train_encoded,
                y_final_train,
            )


# ============================================================
# 14. FINAL HELD-OUT TEST EVALUATION
# ============================================================

test_predictions = final_model.predict(X_final_test_encoded)
test_probabilities = final_model.predict_proba(X_final_test_encoded)

test_metrics = calculate_metrics(y_test, test_predictions)
test_loss = log_loss(
    y_test,
    test_probabilities,
    labels=classes,
)

print("\n" + "=" * 65)
print("FINAL HELD-OUT TEST RESULTS")
print("=" * 65)

for metric_name, value in test_metrics.items():
    print(f"{metric_name}: {value:.4f}")

print(f"test_log_loss: {test_loss:.4f}")

pd.DataFrame([{
    **test_metrics,
    "test_log_loss": test_loss,
    "selected_experiment": best_experiment_name,
    "epochs": final_epochs,
}]).to_csv(
    RESULTS_DIR / "test_metrics.csv",
    index=False,
)


# ============================================================
# 15. FIGURE 6: HELD-OUT TEST CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=CLASS_NAMES,
)

fig, ax = plt.subplots(figsize=(8, 7))
image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
fig.colorbar(image, ax=ax)

ax.set(
    xticks=np.arange(len(CLASS_NAMES)),
    yticks=np.arange(len(CLASS_NAMES)),
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    xlabel="Predicted label",
    ylabel="True label",
    title="Held-out Test Confusion Matrix",
)

threshold = cm.max() / 2 if cm.size else 0

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(
            j,
            i,
            format(cm[i, j], "d"),
            ha="center",
            va="center",
            color="white" if cm[i, j] > threshold else "navy",
            fontsize=12,
        )

fig.tight_layout()
fig.savefig(
    RESULTS_DIR / "test_confusion_matrix.png",
    dpi=300,
)
plt.close(fig)

pd.DataFrame(
    cm,
    index=[f"true_{name}" for name in CLASS_NAMES],
    columns=[f"pred_{name}" for name in CLASS_NAMES],
).to_csv(RESULTS_DIR / "test_confusion_matrix.csv")


# ============================================================
# 16. CLASSIFICATION REPORT AND TEST PREDICTIONS
# ============================================================

report = classification_report(
    y_test,
    test_predictions,
    labels=CLASS_NAMES,
    output_dict=True,
    zero_division=0,
)

pd.DataFrame(report).transpose().to_csv(
    RESULTS_DIR / "classification_report.csv"
)

predictions_df = pd.DataFrame({
    "actual": np.asarray(y_test),
    "predicted": test_predictions,
})

predictions_df.to_csv(
    RESULTS_DIR / "test_predictions.csv",
    index=False,
)


# ============================================================
# 17. SAVE RUN CONFIGURATION
# ============================================================

run_info = {
    "random_seed": SEED,
    "dataset_rows": len(df),
    "training_rows": len(X_train),
    "validation_rows": len(X_val),
    "test_rows": len(X_test),
    "encoded_features": X_train_encoded.shape[1],
    "selected_experiment": best_experiment_name,
    "selected_hidden_layer_sizes": str(
        best_config["hidden_layer_sizes"]
    ),
    "selected_learning_rate": best_config["learning_rate_init"],
    "selected_batch_size": best_config["batch_size"],
    "selected_best_epoch": final_epochs,
    "test_accuracy": test_metrics["accuracy"],
    "test_macro_precision": test_metrics["macro_precision"],
    "test_macro_recall": test_metrics["macro_recall"],
    "test_macro_f1": test_metrics["macro_f1"],
    "test_log_loss": test_loss,
}

pd.DataFrame([run_info]).to_csv(
    RESULTS_DIR / "run_summary.csv",
    index=False,
)

print("\nAll outputs have been saved in the 'results' folder.")
print("Important output files:")
print("  results/experiment_results.csv")
print("  results/training_validation_accuracy.png")
print("  results/training_validation_loss.png")
print("  results/training_loss.png")
print("  results/experiment_comparison.png")
print("  results/test_confusion_matrix.png")
print("  results/test_confusion_matrix.csv")
print("  results/test_metrics.csv")
print("  results/classification_report.csv")
print("  results/test_predictions.csv")
print("  results/run_summary.csv")
