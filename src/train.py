import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve
import warnings

warnings.filterwarnings("ignore")
import wandb


def train(model, train_loader, val_loader, config, device, save, sweep=False):
    model = model.to(device)
    loss_function = torch.nn.CrossEntropyLoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), config.lr)

    best_metric = -1
    best_metric_epoch = -1
    best_val_loss = 1000
    epochs = 200
    print("-" * 30)
    print("Training ... ")
    early_stop = 50
    es_counter = 0

    for epoch in range(epochs):

        print("-" * 10)
        print(f"epoch {epoch + 1}/{epochs}")
        model.train()
        epoch_train_loss = 0

        for i, data in enumerate(train_loader):
            x = data.to(device)
            y = torch.tensor(data.y).type(torch.LongTensor).to(device)
            optimizer.zero_grad()
            out, _, _ = model(x)

            step_loss = loss_function(out, y)
            step_loss.backward(retain_graph=True)
            optimizer.step()
            epoch_train_loss += step_loss.item()

        epoch_train_loss = epoch_train_loss / (i + 1)

        val_loss, val_acc = validate_model(model, val_loader, device)
        print(f"epoch {epoch + 1} train loss: {epoch_train_loss:.4f}")

        if val_loss <= best_val_loss and epoch > 20:
            best_metric = val_acc
            best_val_loss = val_loss
            best_metric_epoch = epoch + 1
            with open(save, "wb") as f:
                torch.save(model, f)
            print("saved new best metric model")
            es_counter = 0
            if sweep:
                wandb_metric = {
                    "train_loss": epoch_train_loss,
                    "best_val_loss": best_val_loss,
                    "best_val_acc": best_metric,
                    "at epoch": best_metric_epoch,
                }
                wandb.log(wandb_metric)
        else:
            es_counter += 1

        if es_counter > early_stop:
            print("No loss improvment.")
            break
    print(
        f"train completed, best_val_acc: {best_metric:.4f}, best_val_loss: {best_val_loss:.4f} at epoch: {best_metric_epoch}"
    )

    return model


def validate_model(model, val_loader, device):
    model.eval()
    val_loss = 0
    loss_func = torch.nn.CrossEntropyLoss(reduction="sum")

    labels = []
    preds = []
    for i, data in enumerate(val_loader):
        data = data.to(device)
        label = torch.tensor(data.y).type(torch.LongTensor).to(device)
        out, _, _ = model(data)
        step_loss = loss_func(out, label)
        val_loss += step_loss.detach().item()
        preds.append(out.argmax(dim=1).detach().cpu().numpy())
        labels.append(label.cpu().numpy())
    preds = np.concatenate(preds).ravel()
    labels = np.concatenate(labels).ravel()
    acc = accuracy_score(preds, labels)
    loss = val_loss / (i + 1)
    return loss, acc


def test_model(test_loader, device, save, trained_model=None):
    if trained_model == None:
        with open(save, "rb") as f:
            model = torch.load(f)
    else:
        model = trained_model
    model.eval()
    labels = []
    all_probs = []
    preds = []

    for i, data in enumerate(test_loader):
        data = data.to(device)
        label = torch.tensor(data.y).type(torch.LongTensor).to(device)
        out, _, _ = model(data)

        probs = F.softmax(out, dim=1)
        all_probs.append(probs[:, 1].detach().cpu().numpy())
        preds.append(out.argmax(dim=1).detach().cpu().numpy())
        labels.append(label.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    preds = np.concatenate(preds).ravel()
    labels = np.concatenate(labels).ravel()

    acc = f1_score(labels, preds)
    auc = roc_auc_score(labels, all_probs)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    spec = tn / (tn + fp)
    sens = tp / (tp + fn)
    fpr, tpr, thresholds = roc_curve(labels, all_probs)
    roc_data = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}

    return acc, sens, spec, auc
