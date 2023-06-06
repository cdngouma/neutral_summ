import pandas as pd
import time

#Personnalized libraries
from configs.config import DatasetConfig, HP
from data.DataLoader import build_dataloader
from utils.Errors import loss_estimation
from Procedures import Procedure
from model.lm import LanguageModel
from torch.utils.tensorboard import SummaryWriter
from configs.config import follow

import argparse


def build_dataloaders(ds_config, dataset="train"):
    """
    Instanciates dataloader for train, valid and test
    
    Input:
        ds_config: Dataset configurations
    Return:
        train_iter, valid_iter, test_iter: Dataloaders
        test_references: List of reference summaries
    """
    if dataset == "train":
        train_iter, vocab, _ = build_dataloader(
            file_path=ds_config.train_data, 
            vocab_size=ds_config.vocab_size,
            vocab_min_freq=ds_config.min_freq,
            vocab=None,
            is_train=True,
            shuffle_batch=False,
            max_num_reviews=ds_config.max_num_reviews,
            refs_path=None,
            max_len_rev=ds_config.max_len_rev,
            pin_memory=ds_config.pin_memory,
            num_workers=ds_config.workers,
            batch_size=ds_config.batch_size,
            device=ds_config.device
        )
        return train_iter, vocab

    elif dataset == "valid":
        valid_iter, _, valid_references = build_dataloader(
            file_path=ds_config.valid_data, 
            vocab_size=ds_config.vocab_size,
            vocab_min_freq=ds_config.min_freq,
            vocab=vocab,
            is_train=False,
            shuffle_batch=False,
            max_num_reviews=ds_config.max_num_reviews,
            refs_path=None,
            max_len_rev=ds_config.max_len_rev,
            pin_memory=ds_config.pin_memory,
            num_workers=ds_config.workers,
            batch_size=ds_config.batch_size,
            device=ds_config.device
        )
        return valid_iter

    elif dataset == "test":
        test_iter, _, test_references = build_dataloader(
            file_path=ds_config.test_data, 
            vocab_size=ds_config.vocab_size,
            vocab_min_freq=ds_config.min_freq,
            vocab=vocab,
            is_train=False,
            shuffle_batch=False,
            max_num_reviews=15,#ds_config.max_num_reviews,
            refs_path=None,
            max_len_rev=ds_config.max_len_rev,
            pin_memory=ds_config.pin_memory,
            num_workers=ds_config.workers,
            batch_size=ds_config.batch_size,
            device=ds_config.device
        )
        
        return test_iter, test_references
    else:
        raise Exception(f"Dataset '{dataset}' was not found.")


def load_parameters(agrs):
    """
    Load and update model and dataset parameters following specified parameters
    
    Input:
        args: user arguments
    Return:
        ds_config: Dataset configuration
        hp: Model hyperparameters
    """
    ds_config = DatasetConfig()
    hp = HP()
    
    # Update dataset parameters
    ds_config.batch_size = args.batch_size if args.batch_size else ds_config.batch_size
    
    # Update model parameters
    hp.use_rec = args.use_rec if args.use_rec else hp.use_rec
    hp.use_cls = args.use_cls if args.use_cls else hp.use_cls
    hp.use_grl = args.use_grl if args.use_grl else hp.use_grl
    hp.use_gat = args.use_gat if args.use_gat else hp.use_gat
    hp.use_proj = args.use_proj if args.use_proj else hp.use_proj
    hp.dec_hidden_type = args.dec_hidden_type if args.dec_hidden_type else hp.dec_hidden_type
    hp.mean_hidden_type = args.mean_hidden_type if args.mean_hidden_type else hp.mean_hidden_type
    hp.ref_hidden_type = args.ref_hidden_type if args.ref_hidden_type else hp.ref_hidden_type
    hp.gen_hidden_type = args.gen_hidden_type if args.gen_hidden_type else hp.gen_hidden_type
    hp.beam_decode = args.beam_decode if args.beam_decode else hp.beam_decode
    
    num_epochs = args.epochs
    if args.mode == "train":
        hp.lm_epochs = num_epochs if num_epochs else hp.lm_epochs
    elif args.mode == "finetune" or args.mode == "eval":
        hp.summarizer_epochs = num_epochs if num_epochs else hp.summarizer_epochs
    else:
        raise Exception(f"mode '{args.mode}' is not supported. Try 'train', 'finetune', or 'eval'.")
    
    return ds_config, hp
    

def run(args):
    # Load dataset config and model hypermarameters
    ds_config, hp = load_parameters(args)
    
    # Enables Tensorborad monitoring
    if follow["writer"]:
        comment = follow["Name"]
        writer = SummaryWriter(comment=comment)
    else:
        writer = None
    
    if args.mode == "lm" or args.mode == "summ":
        # Load datasets
        train_iter, vocab = build_dataloaders(ds_config, dataset="train")
        valid_iter = build_dataloaders(ds_config, dataset="valid")
        procedure = Procedure(hp, ds_config, vocab, writer=writer, train_ter=train_iter, valid_iter=valid_iter)
        
        if args.mode == "lm":
            # Train language model
            procedure.train_lm(path=args.lm_path, tolerance=args.tolerance, check_every=args.check_every)
        elif args.mode == "summ":
            # Train summarizer
            procedure.train_summarizer(path=args.summ_path, tolerance=args.tolerance, check_every=args.check_every)
    
    elif args.mode == "eval":
        # Load datasets
        _, vocab = build_dataloaders(ds_config, dataset="train")
        test_iter, references = build_dataloaders(ds_config, dataset="test")
        procedure = Procedure(hp, ds_config, vocab, writer=writer, train_ter=train_iter, valid_iter=valid_iter)
    
    else:
        raise Exception(f"mode '{args.mode}' is not supported. Try 'train', 'finetune', or 'eval'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Add arguments
    parser.add_argument("--mode", type=str, required=True, default="summ", help="train or test.")
    parser.add_argument("-b", "--batch_size", type=int, default=None, help="Number of reviews per batch.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument("--use_rec", type=bool, default=None, help="Enable or disable reviews reconstruction. If false, the language model only uses the classifier.")
    parser.add_argument("--use_cls", type=bool, default=None, help="Enable or disable classifier.")
    parser.add_argument("--use_grl", type=bool, default=None, help="Enable or disable GRL.")
    parser.add_argument("--use_gat", type=bool, default=None, help="Enable or disable GAT.")
    parser.add_argument("--use_proj", type=bool, default=None, help="Enable or disable projection mechanism.")
    parser.add_argument("--dec_hidden", type=int, dest="dec_hidden_type", default=None, help="Hidden representations used when decoding the input reviews.")
    parser.add_argument("--mean_hidden", type=int, dest="mean_hidden_type", default=None, help="Hidden representations used to create the mean representations of the input reviews.")
    parser.add_argument("--ref_hidden", type=int, dest="ref_hidden_type", default=None, help="Hidden representations used to compute the cosine similarity.")
    parser.add_argument("--gen_hidden", type=int, dest="gen_hidden_type", default=None, help="Hidden representations used to generate summaries at test time.")
    parser.add_argument("--beam_decode", type=bool, default=None, help="Enable or disable beam decoding.")
    parser.add_argument("--tolerance", type=int, default=None, help="Defines how many times to check the loss value before stopping the training early if the loss doesn't improve.")
    parser.add_argument("--check_every", type=int, default=None, help="Number of epochs after which check the loss value in order to stop early.")
    
    parser.add_argument("--lm_path", type=str, default=None, help="Path where to save the trained language model or path to a pretrained language model.")
    parser.add_argument("--summ_path", type=str, default=None, help="Path where to save the summarizer model.")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run program
    run(args)