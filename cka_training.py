import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import optuna
from cka import linear_CKA
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def cka(X, Y):
    X = X.flatten(start_dim=1).numpy(force=True)
    Y = Y.flatten(start_dim=1).numpy(force=True)
    return linear_CKA(X, Y)


class LogCKACallback(pl.Callback):
    def __init__(self, data, out_file):
        self.data = data
        self.out_file = out_file
        self.cka_scores = []

    def on_train_epoch_end(self, trainer, module):
        cka_score = module.cka_score(self.data)
        self.cka_scores.append(cka_score)

    def on_train_end(self, trainer, module):
        val_metrics,  = trainer.validate(module)
        val_acc = val_metrics['val_accuracy']
        with open(self.out_file, 'a') as f:
            print(*self.cka_scores, sep=',', end=',', file=f)
            print(val_acc, file=f)


class SimpleCNN(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, batch_size=128, weight_decay=0.0, criterion=None):
        super(SimpleCNN, self).__init__()
        self.save_hyperparameters(ignore=['criterion'])
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)
        self.criterion = nn.CrossEntropyLoss() if criterion is None else criterion

    def _layers(self, x):
        out = []
        x = self.conv1(x)
        out.append(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.conv2(x)
        out.append(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        out.append(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        out.append(x)
        return out

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log('val_loss', loss, on_epoch=True, on_step=False, prog_bar=False)
        self.log('val_accuracy', acc, on_epoch=True, on_step=False, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        return optimizer

    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_dataset = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
        return DataLoader(train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=6)

    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        val_dataset = datasets.CIFAR10(root='data', train=False, download=True, transform=transform)
        return DataLoader(val_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=6)

    @torch.no_grad
    def cka_score(self, sample):
        outs = self._layers(sample)
        return sum(cka(i, outs[-1]) for i in outs[:-1])


def train_cnn(learning_rate=1e-3, batch_size=128, weight_decay=0.0, label_smoothing=0.0,
              val_every_n_epochs=1, max_epochs=10):
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    model = SimpleCNN(learning_rate=learning_rate, batch_size=batch_size, weight_decay=weight_decay,
                      criterion=criterion)
    logger = TensorBoardLogger('tb_logs', name='SimpleCNN')
    # checkpoint_callback = ModelCheckpoint(monitor='val_accuracy', mode='max', save_top_k=1, save_last=True)
    cka_data, _ = next(iter(model.val_dataloader()))
    cka_data = cka_data.clone()
    log_cka_callback = LogCKACallback(cka_data, 'cka.txt')
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        log_every_n_steps=100,
        check_val_every_n_epoch=100500,
        # callbacks=[checkpoint_callback]
        callbacks=[log_cka_callback]
    )

    trainer.fit(model)

    val_metrics, = trainer.validate(model)

    return val_metrics


def train_cnn_objective(trial):
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 0, 1e-3)
    n_epochs = trial.suggest_int('n_epochs', 5, 20)
    label_smoothing = trial.suggest_float('label_smoothing', 0, 1)
    val_metrics = train_cnn(learning_rate=lr,
                            batch_size=128,
                            weight_decay=weight_decay,
                            label_smoothing=label_smoothing,
                            val_every_n_epochs=n_epochs,
                            max_epochs=n_epochs)
    return val_metrics['val_accuracy']

if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')
    study.optimize(train_cnn_objective, n_trials=100)
    print(study.best_params)
