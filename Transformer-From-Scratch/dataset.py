import torch
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, dataset, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()

        self.dataset = dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len
        
        # get sos, eos, pos token from tokenizer and stores as attribute
        self.sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype=torch.int64)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        """get input, prepare for transformer model

        Args:
            index (int): index of the data
        """
        # extract raw sentence pair from dataset
        src_target_pair =  self.dataset[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # tokenizes the sentence into list of token ids
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids  # source tokens' ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids  # target tokens' ids

        # pad the tokens
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2   # -2 : since sos and eos is considered
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1   # -1 : since only sos is considered in decoder

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")
        
        # Add sos and eos to source text
        # [SOS] token1 token2 ... tokenN [EOS] [PAD] [PAD] ...
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
            ]
        )

        # [SOS] token1 token2 ... tokenN [PAD] [PAD] ...
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        # token1 token2 ... tokenN [EOS] [PAD] [PAD] ...
        # We never want the model to learn to predict [SOS]
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input, # seq_len
            "decoder_input": decoder_input, # seq_len
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # 1, 1, seq_len
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),  # (1, seq_len) & (1, seq_len, seq_len) broadcasted
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text
        }
    
def causal_mask(size):
    ones = torch.ones(1, size, size)
    mask = torch.triu(ones, diagonal=1).type(torch.int) # triu - upper triangle. keep the upper triangle of diagonal as ones
    return mask == 0