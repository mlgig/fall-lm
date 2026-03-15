import torch
import numpy as np
import torch.distributed as dist

from tqdm import tqdm
from copy import deepcopy
from itertools import chain
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from torch import nn
from datasets.arrow_dataset import Dataset

from sklearn.model_selection import train_test_split

from ..architecture import Mantis8M

from .trainer_utils.architecture import FineTuningNetwork
from .trainer_utils.dataset import LabeledDataset, UnlabeledDataset
from .trainer_utils.criterion import ContrastiveLoss
from .trainer_utils.augmentation import RandomCropResize
from .trainer_utils.scheduling import adjust_learning_rate


class MantisTrainer:

    def __init__(self, device, network=None):

        self.device = device

        if network is None:
            network = Mantis8M(
                seq_len=512,
                hidden_dim=256,
                num_patches=32,
                scalar_scales=None,
                hidden_dim_scalar_enc=32,
                epsilon_scalar_enc=1.1,
                transf_depth=6,
                transf_num_heads=8,
                transf_mlp_dim=512,
                transf_dim_head=128,
                transf_dropout=0.1,
                device=device,
                pre_training=False,
            )

        self.network = network.to(device)

    # ------------------------------------------------------------------
    # PRETRAIN (unchanged except float32 safety)
    # ------------------------------------------------------------------

    def pretrain(
        self,
        x,
        num_epochs=100,
        batch_size=512,
        base_learning_rate=2e-3,
        init_optimizer=None,
        criterion=None,
        augmentation_1=None,
        augmentation_2=None,
        data_parallel=True,
        learning_rate_adjusting=True,
        file_name=None,
    ):

        if data_parallel:
            self.network = nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
            gpu_index = self.device.index if self.device.type == "cuda" else None
            devices_ids = [gpu_index] if gpu_index is not None else None
            self.network = nn.parallel.DistributedDataParallel(
                self.network, device_ids=devices_ids, find_unused_parameters=False
            )

        if type(x) == Dataset:
            train_dataset = x
        else:
            train_dataset = UnlabeledDataset(x)

        sampler = DistributedSampler(train_dataset) if data_parallel else None
        data_loader = DataLoader(train_dataset, sampler=sampler, batch_size=batch_size)

        if criterion is None:
            criterion = ContrastiveLoss(temperature=0.1, device=self.device)

        if augmentation_1 is None:
            augmentation_1 = RandomCropResize(crop_rate_range=[0, 0.2], size=512)

        if augmentation_2 is None:
            augmentation_2 = RandomCropResize(crop_rate_range=[0, 0.2], size=512)

        if init_optimizer is None:
            optimizer = torch.optim.AdamW(
                self.network.parameters(),
                lr=base_learning_rate,
                betas=(0.9, 0.999),
                weight_decay=0.05,
            )
        else:
            optimizer = init_optimizer(self.network.parameters())

        rank = dist.get_rank() if (data_parallel and dist.is_initialized()) else 0
        self.network.train()
        progress_bar = tqdm(range(num_epochs))
        step = 0

        for epoch in progress_bar:
            loss_list = []
            for x_batch in data_loader:
                if learning_rate_adjusting:
                    adjust_learning_rate(
                        num_epochs, optimizer, data_loader, step, base_learning_rate
                    )
                x_batch = x_batch["data"].to(self.device)
                step += 1
                x_augmented_1 = augmentation_1(x_batch).to(self.device)
                x_augmented_2 = augmentation_2(x_batch).to(self.device)
                out_1 = self.network(x_augmented_1)
                out_2 = self.network(x_augmented_2)
                loss = criterion(out_1, out_2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())

            if rank == 0:
                avg_loss = np.mean(loss_list)
                progress_bar.set_description(
                    f"Epoch {epoch}: Train Loss {avg_loss:.4f}", refresh=True
                )
        if rank == 0 and file_name is not None:
            self.save(file_name, data_parallel=data_parallel)
        return self.network

    # ------------------------------------------------------------------
    # FINETUNE WITH EARLY STOPPING
    # ------------------------------------------------------------------

    def fit(
        self,
        x,
        y,
        fine_tuning_type="full",
        adapter=None,
        head=None,
        num_epochs=500,
        batch_size=256,
        base_learning_rate=2e-4,
        init_optimizer=None,
        criterion=None,
        learning_rate_adjusting=True,
        validation_split=0.2,
        early_stopping=True,
        patience=10,
    ):
        self.fine_tuning_type = fine_tuning_type
        if validation_split > 0:
            x_train, x_val, y_train, y_val = train_test_split(
                x,
                y,
                test_size=validation_split,
                stratify=y,
                random_state=42,
            )
        else:
            x_train, y_train = x, y
            x_val, y_val = None, None

        # -------------------------------------------------------
        # HEAD INITIALIZATION
        # -------------------------------------------------------

        if head is None:
            num_channels = x.shape[1] if adapter is None else adapter.new_num_channels
            head = nn.Sequential(
                nn.LayerNorm(self.network.hidden_dim * num_channels),
                nn.Linear(
                    self.network.hidden_dim * num_channels,
                    np.unique(y).shape[0],
                ),
            ).to(self.device)
        else:
            head = head.to(self.device)

        if adapter is not None:
            adapter = adapter.to(self.device)
        if fine_tuning_type == "head":
            self.fine_tuned_model = FineTuningNetwork(None, head, adapter).to(
                self.device
            )
        else:
            self.fine_tuned_model = FineTuningNetwork(
                self.network, head, adapter
            ).to(self.device)
        parameters = self._get_fine_tuning_params(fine_tuning_type)
        params_to_optimize = set([p for p in parameters])

        for _, param in self.fine_tuned_model.named_parameters():
            param.requires_grad = param in params_to_optimize

        self.fine_tuned_model.eval()
        self._set_train(fine_tuning_type)

        if criterion is None:
            criterion = nn.CrossEntropyLoss()

        if init_optimizer is None:
            optimizer = torch.optim.AdamW(
                parameters,
                lr=base_learning_rate,
                betas=(0.9, 0.999),
                weight_decay=0.05,
            )
        else:
            optimizer = init_optimizer(parameters)

        # -------------------------------------------------------
        # DATASETS
        # -------------------------------------------------------

        if fine_tuning_type == "head":
            train_dataset = LabeledDataset(
                self.transform(x_train, batch_size=batch_size, three_dim=False),
                y_train,
            )
            val_dataset = (
                LabeledDataset(
                    self.transform(x_val, batch_size=batch_size, three_dim=False),
                    y_val,
                )
                if x_val is not None
                else None
            )
        else:
            train_dataset = LabeledDataset(x_train, y_train)
            val_dataset = (
                LabeledDataset(x_val, y_val) if x_val is not None else None
            )
        data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_loader = (
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            if val_dataset is not None
            else None
        )

        # -------------------------------------------------------
        # EARLY STOPPING STATE
        # -------------------------------------------------------

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        progress_bar = tqdm(range(num_epochs))
        step = 1

        for epoch in progress_bar:
            self._set_train(fine_tuning_type)
            loss_list = []

            for (x_batch, y_batch) in data_loader:
                if learning_rate_adjusting:
                    adjust_learning_rate(
                        num_epochs, optimizer, data_loader, step, base_learning_rate
                    )
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                step += 1
                output = self.fine_tuned_model(x_batch)
                loss = criterion(output, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())

            train_loss = np.mean(loss_list)
            val_loss = None

            if val_loader is not None:
                self.fine_tuned_model.eval()
                val_losses = []

                with torch.no_grad():
                    for (x_batch, y_batch) in val_loader:
                        x_batch = x_batch.to(self.device)
                        y_batch = y_batch.to(self.device)
                        output = self.fine_tuned_model(x_batch)
                        loss = criterion(output, y_batch)
                        val_losses.append(loss.item())
                val_loss = np.mean(val_losses)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = deepcopy(self.fine_tuned_model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

                if early_stopping and patience_counter >= patience:
                    break

            if val_loss is not None:
                progress_bar.set_description(
                    f"Epoch {epoch}: Train {train_loss:.4f} | Val {val_loss:.4f}",
                    refresh=True,
                )
            else:
                progress_bar.set_description(
                    f"Epoch {epoch}: Train {train_loss:.4f}", refresh=True
                )
        if best_state is not None:
            self.fine_tuned_model.load_state_dict(best_state)
        return self.fine_tuned_model

    # ------------------------------------------------------------------
    # REMAINING METHODS (UNCHANGED)
    # ------------------------------------------------------------------

    def transform(self, x, batch_size=256, three_dim=False, to_numpy=True):
        concat = np.concatenate if to_numpy else torch.cat
        if three_dim:
            return concat(
                [
                    self._transform(
                        x[:, [i], :], batch_size=batch_size, to_numpy=to_numpy
                    )[:, None, :]
                    for i in range(x.shape[1])
                ],
                axis=1,
            )
        else:
            return concat(
                [
                    self._transform(
                        x[:, [i], :], batch_size=batch_size, to_numpy=to_numpy
                    )
                    for i in range(x.shape[1])
                ],
                axis=1,
            )

    def _transform(self, x, batch_size=256, to_numpy=True):
        self.network.eval()
        dataloader = self._prepare_dataloader_for_inference(x, batch_size)
        outs = []
        with torch.no_grad():
            for _, batch in enumerate(dataloader):
                x = batch[0].to(self.device)
                out = self.network(x)
                outs.append(out)
        outs = torch.cat(outs)
        return outs.cpu().numpy() if to_numpy else outs

    def predict_proba(self, x, batch_size=256, to_numpy=True):
        self.fine_tuned_model.eval()
        dataloader = self._prepare_dataloader_for_inference(x, batch_size)
        outs = []
        for _, batch in enumerate(dataloader):
            x = batch[0].to(self.device)
            with torch.no_grad():
                if self.fine_tuning_type == "head":
                    x = torch.cat(
                        [self.network(x[:, [i], :]) for i in range(x.shape[1])],
                        dim=-1,
                    )
                out = torch.softmax(self.fine_tuned_model(x), dim=-1)
            outs.append(out.cpu())
        outs = torch.cat(outs)
        return outs.numpy() if to_numpy else outs

    def predict(self, x, batch_size=256, to_numpy=True):
        probs = self.predict_proba(x, batch_size=batch_size, to_numpy=to_numpy)
        return probs.argmax(axis=1)

    def _prepare_dataloader_for_inference(self, x, batch_size):
        if isinstance(x, torch.Tensor):
            dataset = TensorDataset(x.type(torch.float32))
        else:
            dataset = TensorDataset(torch.tensor(x, dtype=torch.float32))
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
    

    def _get_fine_tuning_params(self, fine_tuning_type):
        tune_params_dict = {
            "full": [self.fine_tuned_model.parameters()],
            "scratch": [self.fine_tuned_model.parameters()],
            "head": [self.fine_tuned_model.head.parameters()],
            "adapter_head": [
                []
                if self.fine_tuned_model.adapter is None
                else self.fine_tuned_model.adapter.parameters(),
                self.fine_tuned_model.head.parameters(),
            ],
        }
        params_list = list(chain(*tune_params_dict[fine_tuning_type]))
        return params_list

    def _set_train(self, fine_tuning_type):
        if fine_tuning_type in ["full", "scratch"]:
            self.fine_tuned_model.train()

        elif fine_tuning_type == "head":
            self.fine_tuned_model.head.train()

        elif fine_tuning_type == "adapter_head":
            self.fine_tuned_model.adapter.train()
            self.fine_tuned_model.head.train()

        else:
            raise KeyError("Unknown fine_tuning_type")