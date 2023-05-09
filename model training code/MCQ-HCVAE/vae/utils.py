import random

import torch
from torch.utils.data import DataLoader, TensorDataset

from squad_utils import (convert_examples_to_features_answer_id,
                         convert_examples_to_harv_features,
                         read_squad_examples)


def get_squad_data_loader(tokenizer, file, shuffle, args):
    examples = read_squad_examples(file, is_training=True, debug=args.debug)
    features = convert_examples_to_features_answer_id(examples,
                                                      tokenizer=tokenizer,
                                                      max_seq_length=args.max_c_len,
                                                      max_query_length=args.max_q_len,
                                                      max_distractor_length=args.max_d_len,
                                                      max_ans_length=args.max_q_len,
                                                      doc_stride=128,
                                                      is_training=True)

    all_c_ids = torch.tensor([f.c_ids for f in features], dtype=torch.long)
    all_q_ids = torch.tensor([f.q_ids for f in features], dtype=torch.long)
    all_d1_ids = torch.tensor([f.d1_ids for f in features], dtype=torch.long)
    all_d2_ids = torch.tensor([f.d2_ids for f in features], dtype=torch.long)
    all_d3_ids = torch.tensor([f.d3_ids for f in features], dtype=torch.long)
    all_tag_ids = torch.tensor([f.tag_ids for f in features], dtype=torch.long)
    all_a_ids = (all_tag_ids != 0).long()
    all_start_positions = torch.tensor([f.noq_start_position for f in features], dtype=torch.long)
    all_end_positions = torch.tensor([f.noq_end_position for f in features], dtype=torch.long)

    all_data = TensorDataset(all_c_ids, all_q_ids, all_a_ids, all_start_positions, all_end_positions,all_d1_ids,all_d2_ids,all_d3_ids)
    data_loader = DataLoader(all_data, args.batch_size, shuffle=shuffle)

    return data_loader, examples, features

def get_harv_data_loader(tokenizer, file, shuffle, ratio, args):
    examples = read_squad_examples(file, is_training=True, debug=args.debug)
    random.shuffle(examples)
    num_ex = int(len(examples) * ratio)
    examples = examples[:num_ex]
    features = convert_examples_to_harv_features(examples,
                                                 tokenizer=tokenizer,
                                                 max_seq_length=args.max_c_len,
                                                 max_query_length=args.max_q_len,
                                                 doc_stride=128,
                                                 is_training=True)
    all_c_ids = torch.tensor([f.c_ids for f in features], dtype=torch.long)
    dataset = TensorDataset(all_c_ids)
    dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=args.batch_size)

    return features, dataloader

def batch_to_device(batch, device):
    batch = (b.to(device) for b in batch)
    c_ids, q_ids, a_ids, start_positions, end_positions, d1_ids, d2_ids, d3_ids = batch

    c_len = torch.sum(torch.sign(c_ids), 1)
    max_c_len = torch.max(c_len)
    c_ids = c_ids[:, :max_c_len]
    a_ids = a_ids[:, :max_c_len]

    q_len = torch.sum(torch.sign(q_ids), 1)
    max_q_len = torch.max(q_len)
    q_ids = q_ids[:, :max_q_len]
    
    d1_len = torch.sum(torch.sign(d1_ids), 1)
    max_d1_len = torch.max(d1_len)
    d1_ids = d1_ids[:, :max_d1_len]
    
    d2_len = torch.sum(torch.sign(d2_ids), 1)
    max_d2_len = torch.max(d2_len)
    d2_ids = d1_ids[:, :max_d2_len]
    
    d3_len = torch.sum(torch.sign(d3_ids), 1)
    max_d3_len = torch.max(d3_len)
    d3_ids = d1_ids[:, :max_d3_len]

    return c_ids, q_ids, a_ids, start_positions, end_positions, d1_ids, d2_ids, d3_ids
