from typing import List, Dict

import pytorch_lightning as pl
from torch.optim import AdamW
from torch.utils.data import Dataset as DatasetBase, DataLoader
from transformers import GPT2LMHeadModel, GPT2Config, get_linear_schedule_with_warmup, BertTokenizer


class Dataset(DatasetBase):
    def __init__(self, lines: List[str], tokenizer: BertTokenizer):
        self.data: List[str] = lines
        self.tokenizer: BertTokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        line = self.data[index]
        line = self.tokenizer.encode_plus(
            line,
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return line


class Net(pl.LightningModule):
    def __init__(
            self,
            train_dataset: List[str],
            batch_size,
            epochs,
            warm_up_steps: int = 0,
            lr: float = 1e-4,
            config_path: str = None,
            pretrained: str = None,
            vocab_path: str = None,
            valid_dataset: List[str] = None,
            additional_special_tokens: Dict[str, str] = None,
    ):
        super(Net, self).__init__()
        self.batch_size = batch_size
        self.epochs = epochs
        self.warm_up_steps = warm_up_steps
        self.lr = lr

        if pretrained:
            self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained(pretrained)
            self.model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained(pretrained)
        else:
            config: GPT2Config = GPT2Config.from_json_file(config_path)
            self.model = GPT2LMHeadModel(config=config)
            self.tokenizer = BertTokenizer(vocab_file=vocab_path, model_max_length=config.n_positions)

        if additional_special_tokens:
            self.tokenizer.add_special_tokens({'additional_special_tokens': list(additional_special_tokens.values())})
        self.t_total = len(train_dataset) * epochs
        self.dataset_train = Dataset(train_dataset, self.tokenizer)
        self.dataset_valid = Dataset(valid_dataset or [], self.tokenizer)

    def forward(self, input_ids, attention_mask):
        input_ids = input_ids
        attention_mask = attention_mask
        r = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
            return_dict=True,
        )
        return r["loss"]

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=8,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_valid,
            batch_size=self.batch_size,
            num_workers=8,
            shuffle=False,
            drop_last=True,
        )

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=0.001)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, self.warm_up_steps, self.t_total
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_nb):
        loss = self.forward(batch["input_ids"], batch["attention_mask"])

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_nb):
        loss = self.forward(batch["input_ids"], batch["attention_mask"])
        return loss
