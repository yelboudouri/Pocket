class PocketConfig:
    """
    Configuration class for Pocket Model.

    Args:
        vocab_size (int, optional): The size of the vocabulary. Default is 16384.
        n_positions (int, optional): The maximum number of positions for positional embeddings. Default is 4096.
        d_model (int, optional): Dimension of the model. Default is 512.
        nhead (int, optional): Number of attention heads for each attention layer. Default is 12.
        d_hid (int, optional): Dimension of the feedforward inner layer. Default is 128.
        nlayers (int, optional): Number of layers in the transformer encoder. Default is 8.
        embd_pdrop (float, optional): Dropout probability of embedding layers. Default is 0.1.
        resid_pdrop (float, optional): Dropout probability of the residual layers. Default is 0.1.
        pad_id (int, optional): Index of the padding token in the vocabulary. Default is 0.
        mask_id (int, optional): Index of the masking token in the vocabulary. Default is 1.
    """

    def __init__(
            self,
            vocab_size=16384,
            n_positions=512,
            d_model=256,
            nhead=8,
            d_hid=64,
            nlayers=4,
            embd_pdrop=0.1,
            resid_pdrop=0.1,
            pad_id=0,
            mask_id=1
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.d_model = d_model
        self.nhead = nhead
        self.d_hid = d_hid
        self.nlayers = nlayers
        self.embd_pdrop = embd_pdrop
        self.resid_pdrop = resid_pdrop
        self.pad_id = pad_id
        self.mask_id = mask_id
