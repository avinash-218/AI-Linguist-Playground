import warnings
from pathlib import Path
import torch
import torch.nn as nn
from tqdm import tqdm
from config import get_weights_file_path, get_config
from torch.utils.data import random_split, DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.tensorboard import SummaryWriter

from dataset import BilingualDataset, causal_mask
from model import build_transformer

def get_all_sentences(dataset, language):
    """Generator function that gives all sentences from the dataset

    Args:
        dataset (_type_): dataset
        language (_type_): language of the dataset
    """
    for item in dataset:
        yield item['translation'][language]

def get_or_build_tokenizer(config, dataset, language):
    """Build tokenizer from scratch specific to the dataset or load from existing file

    Args:
        config (_type_): config of tokenizer
        dataset (_type_): dataset
        language (_type_): language of dataset
    """
    tokenizer_path = Path(config['tokenizer_file'].format(language))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]')) # initialize word level tokenizer
        tokenizer.pre_tokenizer = Whitespace()  # split words by whitespace

        # tokenizer trainer - used to train tokenizer from dataset. includes special token
        # words are in vocabulary if freq is atleast 2
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(dataset, language), trainer=trainer)
        tokenizer.save(str(tokenizer_path))

    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    return tokenizer

def get_dataset(config):
    """load dataset

    Args:
        config (_type_): _description_
    """
    dataset_raw = load_dataset('opus_books', f"{config['lang_src']}-{config['lang_tgt']}", split='train')   #load only train data

    # build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, dataset_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, dataset_raw, config['lang_tgt'])

    # Train - Test Split - 90% - 10%
    train_data_size = int(0.9*len(dataset_raw))
    val_data_size = len(dataset_raw) - train_data_size

    train_dataset_raw, val_dataset_raw = random_split(dataset_raw, [train_data_size, val_data_size])

    train_data = BilingualDataset(train_dataset_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_data = BilingualDataset(val_dataset_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # Loop through the entire dataset to get the max number of tokens for source and target language sentences
    # shorter seq_len - cause data loss
    # longer seq_len - wasted compute - more sparse
    # so to find optimal seq_len, find max of token len of both languages
    max_len_src = 0
    max_len_tgt = 0
    for item in dataset_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids

        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max length of source sentence : {max_len_src}")
    print(f"Max length of target sentence : {max_len_tgt}")

    train_dataloader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model

def train_model(config):
    # Define the device
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    print(f'Using device : {device}')

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True) # create folder if not present

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_dataset(config)    # get the data loaders

    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], eps=1e-9)   # optimizer

    initial_epoch = 0
    global_step = 0

    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:
            model.train()
            encoder_input = batch['encoder_input'].to(device)   # batch, seq len
            decoder_input = batch['decoder_input'].to(device)   # batch, seq len
            encoder_mask = batch['encoder_mask'].to(device) # batch , 1, 1, seq_len
            decoder_mask = batch['decoder_mask'].to(device) # bath , 1, seq_len, seq_len

            # Run the tensors through transformer
            encoder_output = model.encode(encoder_input, encoder_mask)  # batch, seqlen, seqlen
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)    # batch, seqlen, d_model
            proj_output = model.project(decoder_output) # batch, seq len, tgt vocab size

            label = batch['label'].to(device)   # batch, seq len

            # batch, seq_len, tgt_vocab_size -> batch * seq_len, tgt_vocab_size
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))

            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

            # log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # backpropagate
            loss.backward()

            # update weights
            optimizer.step()
            optimizer.zero_grad()

            run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer, num_examples=2)

            global_step+=1

        # save model
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            "epoch":epoch,
            "model_state_dict":model.state_dict(),
            "optimzer_state_dict":optimizer.state_dict(),
            "global_step":global_step
        }, model_filename)

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # precompute encoder output and reuse it for every token we get from the decoder
    encoder_output = model.encode(source, source_mask)

    # initialize decode input with sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for the target (decoder input)
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output of decoder
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get the next token
        prob = model.project(out[:, -1])

        # select the token with max probability (since greedy search)
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)], dim=1)  # concat existin token with predicted token

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)

def run_validation(model, validation_data, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0

    # size of console width window
    console_width = 80

    with torch.no_grad():
        for batch in validation_data:
            count+=1
            
            encoder_input = batch['encoder_input'].to(device)   # batch, seq len
            encoder_mask = batch['encoder_mask'].to(device) # batch , 1, 1, seq_len

            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy()) # turns from token to human readable sentence

            # print
            print_msg('-'*console_width)
            print_msg(f'SOURCE: {source_text}')
            print_msg(f'TARGET: {target_text}')
            print_msg(f'PREDICTED: {model_out_text}')

            if count == num_examples:
                break

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)