"""
lstm_raw_ohlcv.py — Tests whether raw OHLCV price-action shape (not
hand-engineered indicators) carries predictive signal for the 5-bar
triple-barrier outcome. Uses a small LSTM over the 60 raw hourly bars
immediately preceding each signal (log returns + high/low/open relative
to close + log volume, all lookahead-safe by construction since the
window ends at the signal's own last-closed bar).

Chronological internal validation split within the training pool (85/15),
early stopping on validation AUC, final evaluation once on the untouched
2025-01 -> 2026-07 test set.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

torch.manual_seed(42)
np.random.seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class LSTMClassifier(nn.Module):
    def __init__(self, n_features=5, hidden=32, layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True,
                             dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden, 16), nn.ReLU(), nn.Dropout(dropout), nn.Linear(16, 1)
        )

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]  # (batch, hidden) -- final layer's final hidden state
        return self.head(last_hidden).squeeze(-1)


def train_and_eval(X_train, y_train, X_test, y_test, seed, epochs=60, patience=10):
    torch.manual_seed(seed)
    n = len(X_train)
    vs = int(n * 0.85)  # chronological internal val split, no shuffling
    X_tr, y_tr = X_train[:vs], y_train[:vs]
    X_vl, y_vl = X_train[vs:], y_train[vs:]

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    X_vl_t = torch.tensor(X_vl, dtype=torch.float32).to(DEVICE)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

    model = LSTMClassifier().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=64, shuffle=True)

    best_val_auc, best_state, no_improve = -1, None, 0
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_vl_t).cpu().numpy()
            val_auc = roc_auc_score(y_vl, val_logits) if len(set(y_vl)) > 1 else 0.5

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_t).cpu().numpy()
    test_probs = 1 / (1 + np.exp(-test_logits))
    test_auc = roc_auc_score(y_test, test_probs)
    return test_auc, test_probs, best_val_auc, epoch + 1


def main():
    X_train = np.load('/tmp/lstm_X_train.npy')
    y_train = np.load('/tmp/lstm_y_train.npy')
    X_test = np.load('/tmp/lstm_X_test.npy')
    y_test = np.load('/tmp/lstm_y_test.npy')

    print(f"Train: {X_train.shape}  Test: {X_test.shape}")
    print(f"Device: {DEVICE}")
    print()

    SEEDS = [42, 43, 44, 45, 46]
    test_aucs = []
    all_probs = []
    for seed in SEEDS:
        auc, probs, val_auc, n_epochs = train_and_eval(X_train, y_train, X_test, y_test, seed)
        test_aucs.append(auc)
        all_probs.append(probs)
        print(f"  seed {seed}: stopped at epoch {n_epochs}, best internal-val AUC={val_auc:.4f}  TEST AUC={auc:.4f}")

    print()
    print(f"LSTM (raw OHLCV sequences) MEAN TEST AUC: {np.mean(test_aucs):.4f} ± {np.std(test_aucs):.4f}")

    np.save('/tmp/lstm_test_probs.npy', np.mean(all_probs, axis=0))


if __name__ == "__main__":
    main()
