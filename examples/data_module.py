def bins(s):
    # Can't do less bins AND have enough elements in each bin
    # There are several "outliers" in the NCREs activity,
    # take another log2 to compact the data further for assigning bins
    return pd.cut(np.log2(s + 1), bins=8, labels=False)


def make_tensor_dataset(df: pd.DataFrame, x_col: str, y_col: str):
    x = np.stack(df[x_col].values)
    # convert input: [batch, seq_len, 4] -> [batch, 4, 1, seq_len]
    tensor_x = torch.Tensor(x).permute(0, 2, 1).unsqueeze(2)
    # add one dimension in targets: [batch] -> [batch, 1]
    tensor_y = torch.Tensor(df[y_col].values).unsqueeze(1)
    return TensorDataset(tensor_x, tensor_y)


class DMSTARR(L.LightningDataModule):
    def __init__(
        self,
        fp: Path,
        y_col: str = "NSC_log2_enrichment",
        sample: Optional[int] = None,
        batch_size: int = 256,
        random_state: int = 913,
    ):
        super().__init__()
        self.fp = fp
        self.y_col = y_col
        self.sample = sample  # for quick code tests with small data sample
        self.df = self.df_train = self.df_val = self.df_test = None
        self.ds_train = self.ds_val = self.ds_test = None
        self.batch_size = batch_size
        self.random_state = random_state

    def prepare_data(self):
        usecols = [
            # "Chr",
            # "Start",
            # "End",
            # "NSC_log2_enrichment",
            # "ESC_log2_enrichment",
            self.y_col,
            "Seq",
            # "SeqRevComp",
        ]
        df = pd.read_csv(self.fp, usecols=usecols)
        df["SeqEnc"] = df.Seq.map(one_hot_encode).map(pad_arr)

        if self.sample:
            _, df = train_test_split(
                df,
                test_size=self.sample,
                random_state=self.random_state,
                stratify=bins(df[self.y_col]),
            )

        self.df = df

        df, self.df_test = train_test_split(
            df,
            test_size=0.10,
            random_state=self.random_state,
            stratify=bins(df[self.y_col]),
        )

        self.df_train, self.df_val = train_test_split(
            df,
            test_size=0.10,
            random_state=self.random_state,
            stratify=bins(df[self.y_col]),
        )

    def setup(self, stage: Optional[str] = None):
        func = partial(make_tensor_dataset, x_col="SeqEnc", y_col=self.y_col)
        if stage == "fit":
            self.ds_train = func(df=self.df_train)
            self.ds_val = func(df=self.df_val)
        elif stage == "test":
            self.ds_test = func(df=self.df_test)
        else:
            raise NotImplementedError(f"{stage = }")

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size, shuffle=False)
