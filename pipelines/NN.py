import os
import sys
from ast import literal_eval
from random import seed as random_seed

import pandas as pd

from numpy import float32
from numpy.random import seed as np_random_seed

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, RobustScaler

from torch import nn, tensor, no_grad, sigmoid
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda import is_available as cuda_is_available
from torch.optim import Adam
from torch import manual_seed as torch_manual_seed
from torch import use_deterministic_algorithms
from torch import save as torch_save

from dvclive.live import Live

from wikilacra.scoring import scoring_functions
from wikilacra.training import train_val_test
from wikilacra.scaling import scaler


class EventsModel(nn.Module):
    """
    Neural net specified by input size (input_size), constant dropout across
    layers (dropout), number of hidden layers (N_hidden), constant hidden layer
    width (size_hidden), and the same activation function for all layers (set by
    the string activation).
    """

    def __init__(self, input_size, dropout, N_hidden, size_hidden, activation):
        super(EventsModel, self).__init__()

        if N_hidden < 1:
            raise Warning("N_hidden must be >= 1")

        match activation:
            case "ReLU":
                act_fn = nn.ReLU
            case _:
                raise Warning("Must select a valid activation function.")

        self.flatten = nn.Flatten()
        layers = [
            nn.Linear(input_size, size_hidden),
            act_fn(),
            nn.Dropout(dropout),
        ]
        for _ in range(N_hidden - 1):
            layers += [
                nn.Linear(size_hidden, size_hidden),
                act_fn(),
                nn.Dropout(dropout),
            ]
        layers += [nn.Linear(size_hidden, 1)]

        self.sequential = nn.Sequential(*layers)

    def forward(self, x):
        x = self.flatten(x)
        return self.sequential(x).squeeze(1)


if __name__ == "__main__":
    # Directory of the data
    engineered_dir = str(sys.argv[1])
    # Base name the engineered data
    engineered_basename = str(sys.argv[2])
    # Metric to optimize in cross-validation
    metric_name = str(sys.argv[3])  # precision, recall, fpr, tpr, f1
    # Random state to set in the appropriate sklearn functions
    random_state = int(sys.argv[4])
    # Number of concurrent jobs for cross-validation grid search
    n_jobs = int(sys.argv[5])
    # Proportion of test data to be held out for validation
    val_prop = float(sys.argv[6])
    # Proportion of test data to be held out for test
    test_prop = float(sys.argv[7])
    # Scaler type to apply after the custom scaling (see wikilacra.scaling)
    scale_type = str(sys.argv[8])
    # Learning rate
    lr = float(sys.argv[9])
    # Dropout rate after each layer
    dropout = float(sys.argv[10])
    # Number of hidden layers
    N_hidden = int(sys.argv[11])
    # Size of the hidden layers
    size_hidden = int(sys.argv[12])
    # Activation function for every layer
    activation = str(sys.argv[13])
    # Number of training epochs
    epochs = int(sys.argv[14])
    # Weight for positive classes
    pos_weight = float(sys.argv[15])

    torch_manual_seed(random_state)
    np_random_seed(random_state)
    random_seed(random_state)
    use_deterministic_algorithms(True)

    # Load the engineered and cleaned features
    engineered = pd.read_csv(
        os.path.join(engineered_dir, engineered_basename + ".csv"), index_col=0
    )
    # Read the endog/exog column lists
    with open(
        os.path.join(engineered_dir, engineered_basename + "_exog_cols.txt"), "r"
    ) as f:
        exog_cols = literal_eval(f.readline())
    with open(
        os.path.join(engineered_dir, engineered_basename + "_endog_cols.txt"), "r"
    ) as f:
        endog_cols = literal_eval(f.readline())

    # Normalization scaling from sklearn
    if scale_type == "standard":
        norm_scaler = StandardScaler()
    elif scale_type == "robust":
        norm_scaler = RobustScaler()
    else:
        raise Warning(f"scale-type options are [standard,robust], not {scale_type}")

    # X is the exog cols, y is the endog cols
    X = engineered[exog_cols]
    y = engineered[endog_cols].astype(int)

    X_scaled = scaler(X)
    X_scaled = norm_scaler.fit_transform(X_scaled)

    # Split the test set off, shuffle=False which means we're getting the last
    # entries in test
    X_scaled, X_scaled_val, X_scaled_test, y, y_val, y_test = train_val_test(
        X_scaled, y, val_prop, test_prop, False
    )

    X_train_t = tensor(X_scaled.astype(float32))
    Y_train_t = tensor(y.astype(float32).to_numpy())
    X_val_t = tensor(X_scaled_val.astype(float32))
    Y_val_t = tensor(y_val.astype(float32).to_numpy())
    X_test_t = tensor(X_scaled_test.astype(float32))
    Y_test_t = tensor(y_test.astype(float32).to_numpy())

    event_train = TensorDataset(X_train_t, Y_train_t)
    event_val = TensorDataset(X_val_t, Y_val_t)
    train_loader = DataLoader(event_train, batch_size=64)
    val_loader = DataLoader(event_val, batch_size=64)

    device = "cuda" if cuda_is_available() else "cpu"
    event_model = EventsModel(
        len(exog_cols), dropout, N_hidden, size_hidden, activation
    ).to(device)

    optim = Adam(event_model.parameters(), lr=lr)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=tensor([pos_weight], device=device))

with Live("dvclive/NN/") as live:
    for i in range(epochs):
        # --- train ---
        event_model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = event_model(xb)
            loss = loss_function(logits, yb.squeeze(1))
            train_loss += loss.item() * yb.size(0)
            optim.zero_grad()
            loss.backward()
            optim.step()

        # --- log training loss to DVCLive ---
        train_loss /= len(train_loader.dataset)
        live.log_metric("train/loss", train_loss)

        # --- eval: val error rate ---
        event_model.eval()
        val_loss = 0.0
        with no_grad():
            for xt, yt in val_loader:
                xt, yt = xt.to(device), yt.to(device)
                logits = event_model(xt)
                val_loss += loss_function(logits, yt.squeeze(1)).item() * yt.size(0)

        # --- log val loss to DVCLive ---
        val_loss /= len(val_loader.dataset)
        live.log_metric("val/loss", val_loss)
        live.next_step()  # step == epoch index

    # --- log val metrics ---
    event_model.eval()
    with no_grad():
        logits = event_model(X_val_t)
    val_preds = (sigmoid(logits) >= 0.5) * 1
    for _metric in scoring_functions.keys():
        score = scoring_functions[_metric](y_val, val_preds)
        live.log_metric(f"val/{_metric}", score, plot=False)

    # --- log test metrics ---
    event_model.eval()
    with no_grad():
        logits = event_model(X_test_t)
    test_preds = (sigmoid(logits) >= 0.5) * 1
    for _metric in scoring_functions.keys():
        score = scoring_functions[_metric](y_test, test_preds)
        live.log_metric(f"test/{_metric}", score, plot=False)

    torch_save(event_model, "outputs/models/NN-model.pt")
    torch_save(event_model.state_dict(), "outputs/models/NN-model-state.pt")
    live.log_artifact("outputs/models/NN-model.pt", name="NN-model")
    live.log_artifact("outputs/models/NN-model-state.pt", name="NN-model-state")

    # Confusion matrix on the test data
    fCMD = ConfusionMatrixDisplay.from_predictions(
        y_test, test_preds, display_labels=["NONE", "EVENT"]
    ).figure_

    # Log images and params into dvclive
    live.log_image("ConfusionMatrixDisplay.png", fCMD)
