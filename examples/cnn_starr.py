class CNNSTARR(L.LightningModule):
    def __init__(
        self,
        lr: float = 0.01,
        weight_decay: float = 0,
        revcomp: bool = True,
        log_vars_prefix: str = "NSC",
    ):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.revcomp = revcomp
        self.log_vars_prefix = log_vars_prefix
        self.loss_fn = nn.MSELoss()

        self.backbone = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=(1, 11), padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.2),
            nn.Conv2d(64, 128, kernel_size=(1, 9), padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.Dropout(0.2),
            nn.Conv2d(128, 256, kernel_size=(1, 7), padding="same"),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((1, 2), (1, 2)),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

    def forward_backbone(self, x):
        z = self.backbone(x)  # shape [batch, 512, 1, 1]
        z = z.view(z.size(0), -1)  # flatten to [batch, 512]
        return z

    def forward(self, x):
        embed_fwd = self.forward_backbone(x)
        if self.revcomp:
            embed_rc = self.forward_backbone(tensor_reverse_complement(x))
            embed_merged = (embed_fwd + embed_rc) / 2
        else:
            embed_merged = embed_fwd
        return self.head(embed_merged)

    def _step(self, batch, batch_idx, suffix: str):
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        out = self(inputs)
        loss = self.loss_fn(out, targets)

        # Log the training loss (this shows up in TensorBoard)
        self.log(
            # This var is used by the EarlyStopping
            f"{self.log_vars_prefix}_loss_{suffix}",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, suffix="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, suffix="val")

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
