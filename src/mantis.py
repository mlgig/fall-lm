from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from third_party.mantis_src.architecture import Mantis8M
# from mantis.trainer import MantisTrainer
from third_party.mantis_src.trainer import MantisTrainer
from third_party.mantis_src.adapters import MultichannelProjector, LinearChannelCombiner
import torch
import torch.nn.functional as F
import numpy as np
import random

class MantisClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):
    def __init__(self, device=None,
                 adapter_type=None, 
                 adapter_params=None, 
                 new_seq_len=320, 
                 fine_tuning_type='full', 
                 finetune_epochs=100,
                 pretrained_model="paris-noah/Mantis-8M",
                 random_state=None):
        
        # --- SEED INITIALIZATION ---
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)
            random.seed(random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random_state)
        
        # --- NEW DEVICE LOGIC ---
        if device is None:
            if torch.cuda.is_available(): self.device = 'cuda'
            elif torch.backends.mps.is_available(): self.device = 'mps'
            else: self.device = 'cpu'
        else:
            self.device = device
            
        self.adapter_type = adapter_type
        self.adapter_params = adapter_params
        self.new_seq_len = new_seq_len
        self.fine_tuning_type = fine_tuning_type
        self.finetune_epochs = finetune_epochs
        self.pretrained_model = pretrained_model
        self.random_state = random_state
        self.threshold_ = 0.5 
    
    def _resize_input(self, X):
        if X.ndim == 2:
            X = X.reshape((X.shape[0], -1, X.shape[1]))
        if self.new_seq_len % 32 != 0:
            raise ValueError("new_seq_len must be a multiple of 32")
        if X.shape[-1] == self.new_seq_len:
            return X
        X_scaled = F.interpolate(torch.tensor(X, dtype=torch.float),
                                 size=self.new_seq_len, mode='linear',
                                 align_corners=False)
        return X_scaled.numpy()

    def _build_adapter(self):
        if self.adapter_type == 'PCA':
            return MultichannelProjector(**self.adapter_params)
        elif self.adapter_type == 'linear':
            return LinearChannelCombiner(**self.adapter_params)
        return None

    def _init_optimizer(self, params):
        return torch.optim.AdamW(params, lr=2e-4, betas=(0.9, 0.999), weight_decay=0.05)

    def fit(self, X, y=None):
        self.network = Mantis8M(device=self.device).from_pretrained(self.pretrained_model)
        self.model = MantisTrainer(device=self.device, network=self.network)
        X = self._resize_input(X)
        if y is not None:
            # adapter = self._build_adapter()
            fit_kwargs = {
                'num_epochs': self.finetune_epochs,
                'validation_split': 0.2,
                'early_stopping': True,
                'patience': 10,
                'fine_tuning_type': self.fine_tuning_type,
                'init_optimizer': self._init_optimizer
            }
            self.model.fit(X, y, **fit_kwargs) 
        return self

    def predict_proba(self, X):
        X = self._resize_input(X)
        return self.model.predict_proba(X)
        
    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self.threshold_).astype(int)