# Copyright (c) 2019, Facebook, Inc. and its affiliates. All Rights Reserved
"""
Run BERT on several relation extraction benchmarks.
Adding some special tokens instead of doing span pair prediction in this version.
"""

import argparse
import logging
import os
import random
import time
import json

import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter

from torch.nn import CrossEntropyLoss

from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from bert_for_multilearning import BertForMultitaskLearning1, BertForMultitaskLearning2
from transformers import BertTokenizer
from optimization import BertAdam, warmup_linear
from tqdm import tqdm

import sys

CLS = "[CLS]"
SEP = "[SEP]"
EVAL_TAGS, EVAL_RELATIONS = [], []

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample1(object):
    """A single training/test example for span pair classification."""

    def __init__(self, guid, sentence, sent_type, tag, relation, sent_start, sent_end, subj_start, subj_end):
        self.guid = guid
        self.sentence = sentence
        self.sent_type = sent_type
        self.tag = tag
        self.relation = relation
        self.sent_start = sent_start
        self.sent_end = sent_end
        self.subj_start = subj_start
        self.subj_end = subj_end


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, sent_type_id, tag_id, relation_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.sent_type_id = sent_type_id
        self.tag_id = tag_id
        self.relation_id = relation_id


class DataProcessor(object):
    """Processor for the TACRED data set."""

    @classmethod
    def _read_json(cls, input_file):
        with open(input_file, "r", encoding='utf-8') as reader:
            data = json.load(reader)
        return data

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples1(
            self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples1(
            self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, test_file):
        """See base class."""
        return self._create_examples1(
            self._read_json(test_file), "test")

    def get_sent_type_labels(self, data_dir):
        """See base class."""
        dataset = self._read_json(os.path.join(data_dir, "train.json"))
        count = Counter()
        labels = []
        for example in dataset:
            count[example['sent_type']] += 1
        logger.info(f"sent_type: {len(count)} labels")
        for label, count in count.most_common():
            logger.info("%s: %.2f%%" % (label, count * 100.0 / len(dataset)))
            if label not in labels:
                labels.append(label)
        return labels

    def get_tag_labels(self, data_dir):
        """See base class."""
        dataset = self._read_json(os.path.join(data_dir, "train.json"))
        denominator = len([tag for example in dataset for tag in example['tag']])
        count = Counter()
        labels = []
        for example in dataset:
            for tag in example['tag']:
                count[tag] += 1
        logger.info(f"tag: {len(count)} labels")
        for label, count in count.most_common():
            logger.info("%s: %.2f%%" % (label, count * 100.0 / denominator))
            if label not in labels:
                labels.append(label)
        return labels

    def get_relation_labels1(self, data_dir):
        """See base class."""
        dataset = self._read_json(os.path.join(data_dir, "train.json"))
        denominator = len([rel for example in dataset for rel in example['relation']])
        count = Counter()
        labels = []
        for example in dataset:
            for relation in example['relation']:
                count[relation] += 1
        logger.info(f"relation: {len(count)} labels")
        for label, count in count.most_common():
            logger.info("%s: %.2f%%" % (label, count * 100.0 / denominator))
            if label not in labels:
                labels.append(label)
        return labels

    def _create_examples1(self, dataset, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for example in dataset:
            sentence = [convert_token(token) for token in example['token']]
            assert example['subj_start'] <= example['subj_end'] < len(sentence)
            examples.append(InputExample1(guid=f'{set_type}-{example["id"]}',
                                          sentence=sentence,
                                          sent_type=example['sent_type'],
                                          tag=example['tag'],
                                          relation=example['relation'],
                                          sent_start=example['sent_start'],
                                          sent_end=example['sent_end'],
                                          subj_start=example['subj_start'],
                                          subj_end=example['subj_end']))
        return examples


def convert_examples_to_features1(examples, label2id, max_seq_length, tokenizer, special_tokens):
    """Loads a data file into a list of `InputBatch`s."""

    def get_special_token(w):
        if w not in special_tokens:
            special_tokens[w] = "[unused%d]" % (len(special_tokens) + 1)
        return special_tokens[w]

    num_tokens = 0
    num_fit_examples = 0
    num_shown_examples = 0
    features = []
    neg_tag_label = 'O'
    neg_rel_label = '0'
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens = [CLS]
        tag_labels = [neg_tag_label]
        relation_labels = [neg_rel_label]
        attention_mask = [1]
        SUBJECT_START = get_special_token("SUBJ_START")
        SUBJECT_END = get_special_token("SUBJ_END")
        SENTENCE_START = get_special_token("SENT_START")
        SENTENCE_END = get_special_token("SENT_END")

        for i, (token, tag, relation) in enumerate(zip(example.sentence, example.tag, example.relation)):
            if i == example.sent_start:
                tokens.append(SENTENCE_START)
                tag_labels.append(neg_tag_label)
                relation_labels.append(neg_rel_label)
                attention_mask.append(1)
            if i == example.subj_start:
                tokens.append(SUBJECT_START)
                tag_labels.append(neg_tag_label)
                relation_labels.append(neg_rel_label)
                attention_mask.append(1)
            for sub_token_id, sub_token in enumerate(tokenizer.tokenize(token)):
                tokens.append(sub_token)
                tag_labels.append(tag)
                relation_labels.append(relation)
                attention_mask.append(1)
            if i == example.subj_end:
                tokens.append(SUBJECT_END)
                tag_labels.append(neg_tag_label)
                relation_labels.append(neg_rel_label)
                attention_mask.append(1)
            if i == example.sent_end:
                tokens.append(SENTENCE_END)
                tag_labels.append(neg_tag_label)
                relation_labels.append(neg_rel_label)
                attention_mask.append(1)
        tokens.append(SEP)
        tag_labels.append(neg_tag_label)
        relation_labels.append(neg_rel_label)
        attention_mask.append(1)
        num_tokens += len(tokens)

        if len(tokens) > max_seq_length:
            tokens = tokens[:max_seq_length - 1] + [SEP]
            tag_labels = tag_labels[:max_seq_length - 1] + [neg_tag_label]
            relation_labels = relation_labels[:max_seq_length - 1] + [neg_rel_label]
            attention_mask = attention_mask[:max_seq_length - 1] + [1]
        else:
            num_fit_examples += 1

        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = attention_mask
        padding = [0] * (max_seq_length - len(input_ids))
        tag_labels += [neg_tag_label] * (max_seq_length - len(input_ids))
        relation_labels += [neg_rel_label] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        try:
            sent_type_id = label2id['sent_type'][example.sent_type]
            tag_id = [label2id['tag'][tag] if tag in label2id['tag'] else label2id['tag']['O'] for tag in tag_labels]
            relation_id = [label2id['relation'][relation] for relation in relation_labels]
        except KeyError:
            print(label2id['tag'], tag_labels)
            raise KeyError

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(tag_id) == max_seq_length
        assert len(relation_id) == max_seq_length

        if num_shown_examples < 20:
            if (ex_index < 5) or (any([x != 0 for x in relation_id])):
                num_shown_examples += 1
                logger.info("*** Example ***")
                logger.info("guid: %s" % example.guid)
                logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("relation_id: %s" % " ".join([str(x) for x in relation_id]))
                logger.info("tag_id: %s" % " ".join([str(x) for x in tag_id]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logger.info("sent_type: %s (id = %d)" % (example.sent_type, sent_type_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          sent_type_id=sent_type_id,
                          tag_id=tag_id,
                          relation_id=relation_id))
    logger.info("Average #tokens: %.2f" % (num_tokens * 1.0 / len(examples)))
    logger.info("%d (%.2f %%) examples can fit max_seq_length = %d" % (num_fit_examples,
                                                                       num_fit_examples * 100.0 / len(examples),
                                                                       max_seq_length))
    return features


def convert_token(token):
    """ Convert PTB tokens to normal tokens """
    if (token.lower() == '-lrb-'):
        return '('
    elif (token.lower() == '-rrb-'):
        return ')'
    elif (token.lower() == '-lsb-'):
        return '['
    elif (token.lower() == '-rsb-'):
        return ']'
    elif (token.lower() == '-lcb-'):
        return '{'
    elif (token.lower() == '-rcb-'):
        return '}'
    return token


def simple_accuracy(sent_type_preds, sent_type_labels, tag_preds, tag_labels, relation_preds, relation_labels):
    return {
        'sent_type_acc': (sent_type_preds == sent_type_labels).mean(),
        'tag_acc': (tag_preds == tag_labels).mean(),
        'relation_acc': (relation_preds == relation_labels).mean()
    }


from sklearn.metrics import classification_report
from collections import defaultdict


def compute_f1(sent_type_preds, sent_type_labels, tag_preds, tag_labels, relation_preds, relation_labels, label2id):
    return {
        'sent_type_f1': classification_report(sent_type_labels, sent_type_preds, output_dict=True)['1']['f1-score'],
        'tag_f1': classification_report(tag_labels, tag_preds, labels=[label2id['tag'][x] for x in EVAL_TAGS],
                                        output_dict=True)['micro avg']['f1-score'],
        'relation_f1': classification_report(relation_labels, relation_preds,
                                             labels=[label2id['relation'][x] for x in EVAL_RELATIONS],
                                             output_dict=True)['micro avg']['f1-score']
    }


def evaluate(model, device, eval_dataloader, eval_sent_type_ids,
             eval_tag_ids, eval_relation_ids, num_sent_type_labels,
             num_tag_labels, num_relation_labels, label2id, compute_scores=True, verbose=True):
    print(compute_scores)
    model.eval()
    if hasattr(model, 'sent_type_loss_weight'):
        sent_type_loss_weight = model.sent_type_loss_weight
        tag_loss_weight = model.tag_loss_weight
        relation_loss_weight = model.relation_loss_weight
    else:
        sent_type_loss_weight = model.module.sent_type_loss_weight
        tag_loss_weight = model.module.tag_loss_weight
        relation_loss_weight = model.module.relation_loss_weight

    eval_loss = defaultdict(float)
    nb_eval_steps = 0
    preds = defaultdict(list)
    for input_ids, input_mask, segment_ids, \
        sent_type_label_ids, tag_label_ids, \
        relation_label_ids in tqdm(eval_dataloader, total=len(eval_dataloader),
                                   desc='validation'):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        sent_type_label_ids = sent_type_label_ids.to(device)
        tag_label_ids = tag_label_ids.to(device)
        relation_label_ids = relation_label_ids.to(device)

        with torch.no_grad():
            output = model(input_ids=input_ids,
                           token_type_ids=segment_ids,
                           attention_mask=input_mask,
                           sent_type_labels=None,
                           tag_labels=None, relation_labels=None)
        sent_type_logits, tag_logits, relation_logits = output[:3]

        loss_fct = CrossEntropyLoss()

        tmp_sent_type_eval_loss = loss_fct(sent_type_logits.view(-1, num_sent_type_labels),
                                           sent_type_label_ids.view(-1))
        eval_loss['sent_type'] += tmp_sent_type_eval_loss.mean().item()

        active_loss = input_mask.view(-1) == 1

        active_labels = tag_label_ids.view(-1)[active_loss]
        active_logits = tag_logits.view(-1, num_tag_labels)[active_loss]
        tmp_tag_eval_loss = loss_fct(active_logits, active_labels)
        eval_loss['tag'] += tmp_tag_eval_loss.mean().item()

        active_labels = relation_label_ids.view(-1)[active_loss]
        active_logits = relation_logits.view(-1, num_relation_labels)[active_loss]
        tmp_relation_eval_loss = loss_fct(active_logits, active_labels)
        eval_loss['relation'] += tmp_relation_eval_loss.mean().item()

        eval_loss['weighted_loss'] = sent_type_loss_weight * eval_loss['sent_type'] + \
                                     tag_loss_weight * eval_loss['tag'] + \
                                     relation_loss_weight * eval_loss['relation']

        nb_eval_steps += 1
        if len(preds['sent_type']) == 0:
            preds['sent_type'].append(sent_type_logits.detach().cpu().numpy())
            preds['tag'].append(tag_logits.detach().cpu().numpy())
            preds['relation'].append(relation_logits.detach().cpu().numpy())
        else:
            preds['sent_type'][0] = np.append(
                preds['sent_type'][0], sent_type_logits.detach().cpu().numpy(), axis=0)
            preds['tag'][0] = np.append(
                preds['tag'][0], tag_logits.detach().cpu().numpy(), axis=0)
            preds['relation'][0] = np.append(
                preds['relation'][0], relation_logits.detach().cpu().numpy(), axis=0)

    scores = {}
    for key in eval_loss:
        eval_loss[key] = eval_loss[key] / nb_eval_steps

    for key in preds:
        scores[key] = np.max(softmax(preds[key][0], axis=-1), axis=-1)
        preds[key] = np.argmax(preds[key][0], axis=-1)

    if compute_scores:
        result = compute_f1(preds['sent_type'], eval_sent_type_ids.numpy(),
                            np.array([x for y in preds['tag'] for x in y]),
                            np.array([x for y in eval_tag_ids.numpy() for x in y]),
                            np.array([x for y in preds['relation'] for x in y]),
                            np.array([x for y in eval_relation_ids.numpy() for
                                      x in y]), label2id)
        result.update(simple_accuracy(preds['sent_type'], eval_sent_type_ids.numpy(),
                                      np.array([x for y in preds['tag'] for x in y]),
                                      np.array([x for y in eval_tag_ids.numpy() for x in y]),
                                      np.array([x for y in preds['relation'] for x in y]),
                                      np.array([x for y in eval_relation_ids.numpy() for
                                                x in y])))
    else:
        result = {}

    for key in eval_loss:
        result[f'eval_loss_{key}'] = eval_loss[key]
    if verbose:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
    return preds, result, scores


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    global EVAL_RELATIONS, EVAL_TAGS

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.do_train:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "train.log"), 'w'))
    else:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "eval.log"), 'w'))
    logger.info(args)
    logger.info("device: {}, n_gpu: {}, 16-bits training: {}".format(
        device, n_gpu, args.fp16))

    processor = DataProcessor()
    # sent_type_label_list = processor.get_sent_type_labels(args.data_dir)
    sent_type_label_list = [0, 1]
    tag_label_list = processor.get_tag_labels(args.data_dir)
    relation_label_list = processor.get_relation_labels1(args.data_dir) if args.mode == 1 \
        else processor.get_relation_labels2(args.data_dir)

    EVAL_TAGS = [tag for tag in tag_label_list if tag != 'O']
    EVAL_RELATIONS = [relation for relation in relation_label_list if relation != '0']

    logger.info(EVAL_TAGS)
    logger.info(EVAL_RELATIONS)

    label2id = {}
    id2label = {}

    label2id['sent_type'] = {label: i for i, label in enumerate(sent_type_label_list)}
    id2label['sent_type'] = {i: label for i, label in enumerate(sent_type_label_list)}
    label2id['tag'] = {label: i for i, label in enumerate(tag_label_list)}
    id2label['tag'] = {i: label for i, label in enumerate(tag_label_list)}
    label2id['relation'] = {label: i for i, label in enumerate(relation_label_list)}
    id2label['relation'] = {i: label for i, label in enumerate(relation_label_list)}

    num_sent_type_labels = 2
    num_tag_labels = len(tag_label_list)
    num_relation_labels = len(relation_label_list)

    do_lower_case = 'uncased' in args.model
    tokenizer = BertTokenizer.from_pretrained(args.model, do_lower_case=do_lower_case)

    special_tokens = {}
    if os.path.exists(os.path.join(args.output_dir, 'special_tokens.json')):
        special_tokens = json.load(open(os.path.join(args.output_dir, 'special_tokens.json')))
    if args.do_validate:
        eval_examples = processor.get_dev_examples(args.data_dir)
        if args.mode == 1:
            eval_features = convert_examples_to_features1(
                eval_examples, label2id, args.max_seq_length, tokenizer, special_tokens)
        # else:
        #     eval_features = convert_examples_to_features2(
        #         eval_examples, label2id, args.max_seq_length, tokenizer, special_tokens)
        logger.info("***** Dev *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_sent_type_label_ids = torch.tensor([f.sent_type_id for f in eval_features], dtype=torch.long)
        all_tag_label_ids = torch.tensor([f.tag_id for f in eval_features], dtype=torch.long)
        all_relation_label_ids = torch.tensor([f.relation_id for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                  all_sent_type_label_ids, all_tag_label_ids, all_relation_label_ids)
        eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size)
        eval_sent_type_ids = all_sent_type_label_ids
        eval_tag_ids = all_tag_label_ids
        eval_relation_ids = all_relation_label_ids

    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        if args.mode == 1:
            train_features = convert_examples_to_features1(
                train_examples, label2id, args.max_seq_length, tokenizer, special_tokens)
        # else:
        #     train_features = convert_examples_to_features2(
        #         train_examples, label2id, args.max_seq_length, tokenizer, special_tokens)
        json.dump(special_tokens, open(os.path.join(args.output_dir, 'special_tokens.json'), 'w'))
        if args.train_mode == 'sorted' or args.train_mode == 'random_sorted':
            train_features = sorted(train_features, key=lambda f: np.sum(f.input_mask))
        else:
            random.shuffle(train_features)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_sent_type_label_ids = torch.tensor([f.sent_type_id for f in train_features], dtype=torch.long)
        all_tag_label_ids = torch.tensor([f.tag_id for f in train_features], dtype=torch.long)
        all_relation_label_ids = torch.tensor([f.relation_id for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                   all_sent_type_label_ids, all_tag_label_ids, all_relation_label_ids)
        train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size)
        train_batches = [batch for batch in train_dataloader]

        num_train_optimization_steps = \
            len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        logger.info("***** Training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        best_result = None
        eval_step = max(1, len(train_batches) // args.eval_per_epoch)
        lrs = [args.learning_rate] if args.learning_rate else \
            [1e-6, 2e-6, 3e-6, 5e-6, 1e-5, 2e-5, 3e-5, 5e-5]
        for lr in lrs:
            if not args.model.startswith('../models/'):
                if args.mode == 1:
                    model = BertForMultitaskLearning1.from_pretrained(
                        args.model, cache_dir=str(PYTORCH_PRETRAINED_BERT_CACHE), num_tag_labels=num_tag_labels,
                        num_relation_labels=num_relation_labels, sent_type_loss_weight=args.sent_type_weight,
                        tag_loss_weight=args.tag_weight, relation_loss_weight=args.relation_weight)
                else:
                    model = BertForMultitaskLearning2.from_pretrained(
                        args.model, cache_dir=str(PYTORCH_PRETRAINED_BERT_CACHE), num_tag_labels=num_tag_labels,
                        num_relation_labels=num_relation_labels, sent_type_loss_weight=args.sent_type_weight,
                        tag_loss_weight=args.tag_weight, relation_loss_weight=args.relation_weight)
            else:
                if args.mode == 1:
                    model = BertForMultitaskLearning1.from_pretrained(args.model, num_tag_labels=num_tag_labels,
                                                                      num_relation_labels=num_relation_labels,
                                                                      from_tf=True,
                                                                      sent_type_loss_weight=args.sent_type_weight,
                                                                      tag_loss_weight=args.tag_weight,
                                                                      relation_loss_weight=args.relation_weight)
                else:
                    model = BertForMultitaskLearning2.from_pretrained(args.model, num_tag_labels=num_tag_labels,
                                                                      num_relation_labels=num_relation_labels,
                                                                      from_tf=True,
                                                                      sent_type_loss_weight=args.sent_type_weight,
                                                                      tag_loss_weight=args.tag_weight,
                                                                      relation_loss_weight=args.relation_weight)
            if args.fp16:
                model.half()
            model.to(device)
            if n_gpu > 1:
                model = torch.nn.DataParallel(model)

            param_optimizer = list(model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer
                            if not any(nd in n for nd in no_decay)], 'weight_decay': float(args.weight_decay)},
                {'params': [p for n, p in param_optimizer
                            if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            if args.fp16:
                try:
                    from apex.optimizers import FP16_Optimizer
                    from apex.optimizers import FusedAdam
                except ImportError:
                    raise ImportError("Please install apex from https://www.github.com/nvidia/apex"
                                      "to use distributed and fp16 training.")

                optimizer = FusedAdam(optimizer_grouped_parameters,
                                      lr=lr,
                                      bias_correction=False,
                                      max_grad_norm=1.0)
                if args.loss_scale == 0:
                    optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
                else:
                    optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

            else:
                optimizer = BertAdam(optimizer_grouped_parameters,
                                     lr=lr,
                                     warmup=args.warmup_proportion,
                                     t_total=num_train_optimization_steps)

            start_time = time.time()
            global_step = 0
            tr_loss = 0
            nb_tr_examples = 0
            nb_tr_steps = 0
            for epoch in range(int(args.num_train_epochs)):
                model.train()
                logger.info("Start epoch #{} (lr = {})...".format(epoch, lr))
                if args.train_mode == 'random' or args.train_mode == 'random_sorted':
                    random.shuffle(train_batches)
                for step, batch in enumerate(tqdm(train_batches, total=len(train_batches), desc='fitting ... ')):
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, input_mask, segment_ids, sent_type_label_ids, tag_label_ids, relation_label_ids = batch
                    loss = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                                 sent_type_labels=sent_type_label_ids,
                                 tag_labels=tag_label_ids,
                                 relation_labels=relation_label_ids)
                    if n_gpu > 1:
                        loss = loss.mean()
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    if args.fp16:
                        optimizer.backward(loss)
                    else:
                        loss.backward()

                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1

                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        if args.fp16:
                            lr_this_step = lr * \
                                           warmup_linear(global_step / num_train_optimization_steps,
                                                         args.warmup_proportion)
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = lr_this_step
                        optimizer.step()
                        optimizer.zero_grad()
                        global_step += 1

                    if args.do_validate and (step + 1) % eval_step == 0:
                        logger.info('Epoch: {}, Step: {} / {}, used_time = {:.2f}s, loss = {:.6f}'.format(
                            epoch, step + 1, len(train_batches),
                                   time.time() - start_time, tr_loss / nb_tr_steps))
                        save_model = False

                        preds, result, scores = evaluate(model, device, eval_dataloader, eval_sent_type_ids,
                                                         eval_tag_ids, eval_relation_ids,
                                                         num_sent_type_labels, num_tag_labels, num_relation_labels,
                                                         label2id)
                        model.train()
                        result['global_step'] = global_step
                        result['epoch'] = epoch
                        result['learning_rate'] = lr
                        result['batch_size'] = args.train_batch_size
                        logger.info("First 20 predictions:")
                        for sent_type_pred, sent_type_label in zip(preds['sent_type'][:20],
                                                                   eval_sent_type_ids.numpy()[:20]):
                            sign = u'\u2713' if sent_type_pred == sent_type_label else u'\u2718'
                            logger.info("pred = %s, label = %s %s" % (id2label['sent_type'][sent_type_pred],
                                                                      id2label['sent_type'][sent_type_label], sign))

                        if (best_result is None) or (result[args.eval_metric] > best_result[args.eval_metric]):
                            best_result = result
                            save_model = True
                            logger.info("!!! Best dev %s (lr=%s, epoch=%d): %.2f" %
                                        (args.eval_metric, str(lr), epoch, result[args.eval_metric] * 100.0))

                        if save_model:
                            model_to_save = model.module if hasattr(model, 'module') else model
                            output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
                            output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
                            torch.save(model_to_save.state_dict(), output_model_file)
                            model_to_save.config.to_json_file(output_config_file)
                            tokenizer.save_vocabulary(args.output_dir)
                            if best_result:
                                output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
                                with open(output_eval_file, "w") as writer:
                                    for key in sorted(result.keys()):
                                        writer.write("%s = %s\n" % (key, str(result[key])))
                                # with open(os.path.join(args.output_dir, "eval_predictions.txt"), "w") as f:
                                #     f.write("id\tpred\tlabel\tsoftmax_score\n")
                                #     for ex, pred, gold, score in zip(eval_examples, preds, eval_sent_type_ids, scores):
                                #         f.write("%s\t%s\t%s\t%s\n" % (
                                #             ex.guid, id2label[pred], id2label[gold.item()], score))
    # json.dump(special_tokens, open(os.path.join(args.output_dir, 'special_tokens.json'), 'w'))
    if args.do_eval:
        # if args.eval_test:
        test_file = os.path.join(args.data_dir, 'test.json') if args.test_file == '' else args.test_file
        eval_examples = processor.get_test_examples(test_file)
        eval_features = convert_examples_to_features1(
            eval_examples, label2id, args.max_seq_length, tokenizer, special_tokens)
        logger.info("***** Test *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_sent_type_label_ids = torch.tensor([f.sent_type_id for f in eval_features], dtype=torch.long)
        all_tag_label_ids = torch.tensor([f.tag_id for f in eval_features], dtype=torch.long)
        all_relation_label_ids = torch.tensor([f.relation_id for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                  all_sent_type_label_ids, all_tag_label_ids, all_relation_label_ids)
        eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size)
        eval_sent_type_ids = all_sent_type_label_ids
        eval_tag_ids = all_tag_label_ids
        eval_relation_ids = all_relation_label_ids

        model = BertForMultitaskLearning1.from_pretrained(
            args.output_dir, num_tag_labels=num_tag_labels,
            num_relation_labels=num_relation_labels, sent_type_loss_weight=args.sent_type_weight,
            tag_loss_weight=args.tag_weight, relation_loss_weight=args.relation_weight)
        if args.fp16:
            model.half()
        model.to(device)
        preds, result, scores = evaluate(model, device, eval_dataloader, eval_sent_type_ids,
                                         eval_tag_ids, eval_relation_ids,
                                         num_sent_type_labels, num_tag_labels, num_relation_labels, label2id,
                                         compute_scores=False)

        prediction_results = {'id': [ex.guid for ex in eval_examples],
                              'TOKEN': [' '.join(ex.sentence) for ex in eval_examples],
                              'sent_start': [ex.sent_start for ex in eval_examples],
                              'sent_end': [ex.sent_end for ex in eval_examples],
                              'subj_start': [ex.subj_start for ex in eval_examples],
                              'subj_end': [ex.subj_end for ex in eval_examples],
                              'sent_type_label': [ex.sent_type for ex in eval_examples],
                              'sent_type_pred': [id2label['sent_type'][x] for x in preds['sent_type']],
                              'sent_type_scores': [score for score in scores['sent_type']],
                              'tag_label': [' '.join(ex.tag) for ex in eval_examples],
                              'tag_pred': [' '.join([id2label['tag'][x] for x in sent]) for
                                           ex, sent in
                                           zip(eval_examples, preds['tag'])],
                              'tag_scores': [' '.join([str(score) for i, score in enumerate(sent)]) for
                                             ex, sent in
                                             zip(eval_examples, scores['tag'])],
                              'relation_label': [' '.join(ex.relation) for ex in eval_examples],
                              'relation_pred': [' '.join([id2label['relation'][x] for x in sent]) for
                                                ex, sent in
                                                zip(eval_examples, preds['relation'])],
                              'relation_scores': [' '.join([str(score) for i, score in enumerate(sent)]) for
                                                  ex, sent in
                                                  zip(eval_examples, scores['relation'])]
                              }

        prediction_results = pd.DataFrame(prediction_results)
        prediction_results.to_csv(os.path.join(args.output_dir, f"{args.test_file.split('/')[-1]}_predictions.tsv"),
                                  sep='\t', index=False)
        with open(os.path.join(args.output_dir, f"{args.test_file.split('/')[-1]}_eval_results.txt"), "w") as f:
            for key in sorted(result.keys()):
                f.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", default='', type=str, required=False)
    parser.add_argument("--model", default=None, type=str, required=True)
    parser.add_argument("--from_tf", default=False, type=bool, required=True)
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_per_epoch", default=3, type=int,
                        help="How many times it evaluates on dev set per epoch")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    # parser.add_argument("--negative_label", default="no_relation", type=str)
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_validate", action='store_true', help="Whether to run validation on dev set.")
    parser.add_argument("--train_mode", type=str, default='random_sorted',
                        choices=['random', 'sorted', 'random_sorted'])
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the test set.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    # parser.add_argument("--eval_test", action="store_true", help="Whether to evaluate on final test set.")
    parser.add_argument("--mode", type=int, default=1, choices=[1, 2])
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_metric", default="sent_type_f1", type=str)
    parser.add_argument("--learning_rate", default=None, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=5.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--sent_type_weight", default=0.0, type=float,
                        help="")
    parser.add_argument("--tag_weight", default=0.1, type=float,
                        help="")
    parser.add_argument("--relation_weight", default=1.0, type=float,
                        help="")
    parser.add_argument("--weight_decay", default=0.1, type=float,
                        help="weight_decay coefficient for regularization")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    args = parser.parse_args()
    main(args)
