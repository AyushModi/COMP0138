import argparse
import os
import random

import numpy as np
import torch
from tqdm import tqdm, trange
from transformers import BertTokenizer

from eval import eval_vae
from trainer import VAETrainer
from utils import batch_to_device, get_harv_data_loader, get_squad_data_loader


def main(args):
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    train_loader, _, _ = get_squad_data_loader(tokenizer, args.train_dir,
                                         shuffle=True, args=args)
    eval_data = get_squad_data_loader(tokenizer, args.dev_dir,
                                      shuffle=False, args=args)

    args.device = torch.cuda.current_device()

    trainer = VAETrainer(args)

    loss_log1 = tqdm(total=0, bar_format='{desc}')
    loss_log2 = tqdm(total=0, bar_format='{desc}')
    loss_log3 = tqdm(total=0, bar_format='{desc}')
    eval_log = tqdm(total=0, bar_format='{desc}')
    best_eval_log = tqdm(total=0, bar_format='{desc}')

    print("MODEL DIR: " + args.model_dir)

    best_bleu, best_bleu_d, best_em, best_f1 = 0.0, 0.0, 0.0, 0.0
    for epoch in trange(int(args.epochs), desc="Epoch", position=0):
        for batch in tqdm(train_loader, desc="Train iter", leave=False, position=1):
            c_ids, q_ids, a_ids, start_positions, end_positions,d1_ids,d2_ids,d3_ids \
            = batch_to_device(batch, args.device)
            ## There are now three d_ids, make the appropriate changes.
            trainer.train(c_ids, q_ids, a_ids, start_positions, end_positions,d1_ids,d2_ids,d3_ids)
            
            str1 = 'Q REC : {:06.4f} A REC : {:06.4f} D REC : {:06.4f}'
            str2 = 'ZQ KL : {:06.4f} ZA KL : {:06.4f} ZD KL : {:06.4f}'
            str3 = 'L_INFO : {:06.4f}'#  L_INFO_D_Q : {:06.4f}' # INFO_D_ANS : {:06.4f}'
            str1 = str1.format(float(trainer.loss_q_rec), float(trainer.loss_a_rec), float(trainer.loss_d_rec))
            str2 = str2.format(float(trainer.loss_zq_kl), float(trainer.loss_za_kl), float(trainer.loss_zd_kl))
            str3 = str3.format(float(trainer.loss_info))#, float(trainer.loss_info_d_q))#, float(trainer.loss_info_d_ans))
            loss_log1.set_description_str(str1)
            loss_log2.set_description_str(str2)
            loss_log3.set_description_str(str3)


        if epoch >= int(args.epochs)//2:
            metric_dict, bleu, _, bleu_d = eval_vae(epoch, args, trainer, eval_data)
            f1 = metric_dict["f1"]
            em = metric_dict["exact_match"]
            bleu = bleu * 100
            bleu_d = bleu_d * 100
            _str = '{}-th Epochs Q-BLEU : {:02.2f} D-BLEU : {:02.2f} EM : {:02.2f} F1 : {:02.2f}'
            _str = _str.format(epoch, bleu, bleu_d, em, f1)
            eval_log.set_description_str(_str)
            if em > best_em:
                best_em = em
            if f1 > best_f1:
                best_f1 = f1
                trainer.save(os.path.join(args.model_dir, "best_f1_model.pt"))
            if bleu > best_bleu:
                best_bleu = bleu
                trainer.save(os.path.join(args.model_dir, "best_q_bleu_model.pt"))
            if bleu_d > best_bleu_d:
                best_bleu_d = bleu_d
                trainer.save(os.path.join(args.model_dir, "best_d_bleu_model.pt"))

            _str = 'BEST Q-BLEU : {:02.2f} D-BLEU : {:02.2f} EM : {:02.2f} F1 : {:02.2f}'
            _str = _str.format(best_bleu, best_bleu_d, best_em, best_f1)
            best_eval_log.set_description_str(_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=1004, type=int)
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--train_dir', default='../data/sciq/squad_format/train.json')
    parser.add_argument('--dev_dir', default='../data/sciq/squad_format/valid.json')
    
    parser.add_argument("--max_c_len", default=384, type=int, help="max context length")
    parser.add_argument("--max_q_len", default=64, type=int, help="max query length")
    parser.add_argument("--max_d_len", default=15, type=int, help="max distractor length")

    parser.add_argument("--model_dir", default="../save/vae-checkpoint", type=str)
    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("--lr", default=1e-3, type=float, help="lr")
    parser.add_argument("--batch_size", default=64, type=int, help="batch_size")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="weight decay")
    parser.add_argument("--clip", default=5.0, type=float, help="max grad norm")

    parser.add_argument("--bert_model", default='bert-base-uncased', type=str)
    parser.add_argument('--enc_nhidden', type=int, default=300)
    parser.add_argument('--enc_nlayers', type=int, default=1)
    parser.add_argument('--enc_dropout', type=float, default=0.2)
    parser.add_argument('--dec_a_nhidden', type=int, default=300)
    parser.add_argument('--dec_a_nlayers', type=int, default=1)
    parser.add_argument('--dec_a_dropout', type=float, default=0.2)
    parser.add_argument('--dec_q_nhidden', type=int, default=900)
    parser.add_argument('--dec_q_nlayers', type=int, default=2)
    parser.add_argument('--dec_q_dropout', type=float, default=0.3)
    parser.add_argument('--dec_d_nhidden', type=int, default=900)
    parser.add_argument('--dec_d_nlayers', type=int, default=2)
    parser.add_argument('--dec_d_dropout', type=float, default=0.3)
    parser.add_argument('--nzqdim', type=int, default=50)
    parser.add_argument('--nzddim', type=int, default=50)
    parser.add_argument('--nza', type=int, default=20)
    parser.add_argument('--nzadim', type=int, default=10)
    parser.add_argument('--lambda_kl', type=float, default=0.1)
    parser.add_argument('--lambda_info', type=float, default=1.0)

    args = parser.parse_args()

    if args.debug:
        args.model_dir = "./dummy"
    # set model dir
    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)
    args.model_dir = os.path.abspath(model_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    main(args)
