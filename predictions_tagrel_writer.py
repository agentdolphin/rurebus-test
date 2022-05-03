from collections import namedtuple
import json
import pandas as pd
import os
import numpy as np
from tqdm import tqdm

SpanText = namedtuple('SpanText', ['start', 'stop', 'text'])


def write_tag_and_relation_predictions_to_folder(output_dir: str, eval_examples_path, eval_predictions_path,
                                                 force_write=False, postprocess=True, prefix_part='test_part',
                                                 bert_tokenizer='bert-base-multilingual-uncased', offset_in_id=2):
    """
    Main function. Should write annotated test examples.

    @param output_dir: where to write ('output')
    @param eval_examples_path: test.json examples path? ('examples/test.json')
    @param eval_predictions_path: prediction path from run_multilearning ('out/_predictions.tsv')
    @param force_write: whether to overwrite output of this function
    @param postprocess: generates csv in predictions folder?? Unresolved dependencies here if True. Mutates df if False.
    @param prefix_part: 'test_full' should be used, corresponding examples not found otherwise, used for naming for output files
    @param bert_tokenizer: used only if postprocess == True
    @param offset_in_id: default 2 is OK, used only for column-ids in df.
    """
    check_output_dir(os.path.join(output_dir, 'relations'), force_write)
    check_output_dir(os.path.join(output_dir, 'set_1'), force_write)

    eval_examples = read_examples(eval_examples_path)
    df = pd.read_csv(eval_predictions_path, sep='\t')

    columns_to_transform = [
        'TOKEN', 'tag_label', 'tag_pred', 'tag_scores', 'relation_label', 'relation_pred', 'relation_scores'
    ]
    if postprocess:
        # bert_tokenizer = BertTokenizer.from_pretrained(bert_tokenizer)
        print('There should be postprocess but dependency not found')
        # df = multi_predictions_postprocessing(df, bert_tokenizer)
        df_copy = df.copy()
        for column in columns_to_transform:
            if 'scores' in column:
                df_copy.loc[:, column] = df_copy[column].apply(lambda x: ' '.join([str(y) for y in x]))
            else:
                df_copy.loc[:, column] = df_copy[column].apply(lambda x: ' '.join(x))
        df_copy.to_csv(eval_predictions_path + '_postprocessed.csv', sep='\t', index=False)
    else:
        for column in columns_to_transform:
            if 'scores' in column:
                df.loc[:, column] = df[column].apply(lambda x: [float(y) for y in x.split(' ')])
            else:
                df.loc[:, column] = df[column].apply(lambda x: x.split(' '))

    df.loc[:, 'source_file_name'] = df.id.apply(lambda x: x.split('-')[offset_in_id])
    df.loc[:, 'sent_id_in_source_file'] = df.id.apply(lambda x: x.split('-')[offset_in_id + 1])
    df.loc[:, 'subj_id_in_sentence'] = df.id.apply(lambda x: x.split('-')[offset_in_id + 2])
    unique_source_file_names = df.source_file_name.unique()

    for source_file_name in tqdm(unique_source_file_names, total=len(unique_source_file_names)):
        predicted_relations_and_examples_tuples = []
        predicted_tags_and_examples_tuples = []
        tmp_df = df[df.source_file_name == source_file_name]
        tmp_examples = [example for example in eval_examples if
                        example['id'].split('-')[1] == source_file_name]
        assert len(tmp_df.source_file_name.unique()) == 1
        source_file_name = tmp_df.source_file_name.unique()[0]

        unique_sent_ids_in_source_file = tmp_df.sent_id_in_source_file.unique()
        for sent_id_in_source_file in unique_sent_ids_in_source_file:
            sent_df = tmp_df[tmp_df.sent_id_in_source_file == sent_id_in_source_file]

            aggregated_tags = aggregate_tags_by_max_score(sent_df)
            corresponding_examples = [example for example in tmp_examples if
                                      example['id'].startswith(
                                          f'{prefix_part}-{source_file_name}-{sent_id_in_source_file}')]
            if not corresponding_examples:
                print(corresponding_examples, sent_df)
            predicted_tags_and_examples_tuples.append((aggregated_tags, corresponding_examples[0]))
            for row in sent_df.itertuples():
                subj_id = row.subj_id_in_sentence
                # print(subj_id, [example['id'] for example in corresponding_examples])
                corresponding_example = [example for example in corresponding_examples if
                                         example['id'].startswith(
                                             f'{prefix_part}-{source_file_name}-{sent_id_in_source_file}-{subj_id}')][0]
                relations = row.relation_pred
                if set(relations) == {'0'}:
                    continue
                else:
                    predicted_relations_and_examples_tuples.append(
                        (relations, corresponding_example, corresponding_examples))

        write_tag_ann_to_file(os.path.join(output_dir, 'set_1', f"{source_file_name}.ann"),
                              predicted_tags_and_examples_tuples)
        write_relation_ann_to_file(os.path.join(output_dir, 'relations', f"{source_file_name}.ann"),
                                   predicted_relations_and_examples_tuples)


def write_tag_ann_to_file(filename, predicted_tags_and_examples_tuples):
    encountered_tags = {}
    result = ''
    for example_id, (tags, example) in enumerate(predicted_tags_and_examples_tuples):
        sentence = example['sentence']
        offset = sentence.start
        tags += ['O'] * (len(example['token']) - len(tags))
        spans = get_bio_spans(tags)
        tokens = example['tokenized_sentence']
        for span_id, (token_start, token_stop) in enumerate(spans):
            tag = tags[token_start][2:]
            key = f'{example_id}-{span_id}'
            encountered_tags[key] = f'{len(encountered_tags) + 1}'
            # Error with predicted tags out of range here?? fixed with if
            if token_start < len(tokens) and token_stop < len(tokens):
                start_token = tokens[token_start]
                stop_token = tokens[token_stop]
                entity_text = sentence.text[start_token.start:stop_token.stop]
                start = start_token.start + offset
                stop = stop_token.stop + offset
                result += f'T{encountered_tags[key]}\t{tag} {start} {stop}\t{entity_text}\n'

    print(result, file=open(filename, 'w', encoding='utf-8'), end='')


def get_bio_spans(tags):
    """
    Gets spans tuples (from_span_index, to_span_index) from tags list

    @param tags: list of tags (512) (example: ['O', 'O', 'B-CMP', 'B-CMP', 'O', 'O', 'I-MET', 'I-MET', 'I-MET', 'I-MET', 'I-MET', 'I-MET', 'I-MET', 'I-MET', 'I-MET', 'I-MET', 'O', ..., 'O')
    @return: list of tuples, one element per tag (example: [(2, 2), (3, 3), (6, 15)])
    """
    start_ids = [i for i, tag in enumerate(tags) if tag.startswith('B-')]
    additional_start_ids = []
    prev_tag = 'O'
    for i, tag in enumerate(tags):
        if tag.startswith('I-') and prev_tag == 'O':
            additional_start_ids.append(i)
        prev_tag = tag
    start_ids += additional_start_ids

    stop_ids = []
    for start_id in start_ids:
        stop_id = start_id + 1
        while stop_id < len(tags) and tags[stop_id].startswith('I-'):
            stop_id += 1
        stop_ids.append(stop_id - 1)
    spans = [(start_id, stop_id) for start_id, stop_id in zip(start_ids, stop_ids)]
    return spans


def write_relation_ann_to_file(filename, predicted_relations_and_examples_tuples):
    result = ''
    relation_id = 1
    for _, (relations, example, examples) in enumerate(predicted_relations_and_examples_tuples):
        formatted_relations = get_formatted_relations(relations, example, examples)
        for _, (relation, arg1, arg2) in enumerate(formatted_relations):
            result += f'R{relation_id}\t{relation} {arg1} {arg2}\n'
            relation_id += 1

    print(result, file=open(filename, 'w', encoding='utf-8'), end='')


def get_formatted_relations(relations, example, corresponding_examples):
    found_relations = []
    arg2 = {}
    arg1 = 'Arg1:' + example['id'].split('-')[-1]

    for ex in corresponding_examples:
        if ex['id'] == example['id']:
            continue
        key = ex['id'].split('-')[-1]
        for i, relation in enumerate(relations):
            if relation not in ['0', 'O'] and ex['subj_start'] <= i <= ex['subj_end']:
                if key not in arg2:
                    arg2[key] = set()
                arg2[key].add(relation)
    for key in arg2:
        for relation in arg2[key]:
            found_relations.append((relation, arg1, f'Arg2:{key}'))

    return found_relations


def aggregate_tags_by_max_score(mini_dataset):
    """
    Gets array of predicted tags with max score from all sentence examples

    @param mini_dataset: sentence whole-df subset as df of examples
    @return: tags list of 512 elements (example: ['B-CMP', 'B-MET', 'I-MET', 'I-MET', 'I-MET', 'O', 'O', ..., 'O'])
    """
    df = mini_dataset.copy().reset_index(drop=True)
    scores = np.array([x for x in df.tag_scores.values])
    tags = np.array([x for x in df.tag_pred.values])
    max_ids = np.argsort(scores, axis=0)[-1]
    aggregated_tags = []
    for column, row in enumerate(max_ids):
        aggregated_tags.append(tags[row, column])
    return aggregated_tags


def read_examples(examples_path):
    """
    Reads sentences from test.json.

    @param examples_path: path to test.json
    @return: examples list
    """
    examples = json.load(open(examples_path))
    for i in range(len(examples)):
        examples[i]['sentence'] = SpanText(start=examples[i]['sentence'][0],
                                           stop=examples[i]['sentence'][1],
                                           text=examples[i]['sentence'][2])
        examples[i]['tokenized_sentence'] = [
            SpanText(start=token[0], stop=token[1], text=token[2]) for token in examples[i]['tokenized_sentence']
        ]
    return examples


def check_output_dir(output_dir: str, force_write: bool):
    """
    (force) makes dirs, connected with force_write flag from args

    @param output_dir: output dir to (re)write
    @param force_write: whether force_write mode is used
    """
    if os.path.exists(output_dir):
        if force_write:
            print('overwriting existing files')
            # os.system(f'rm {os.path.join(output_dir, "*")}')
        else:
            raise RuntimeError
    else:
        os.makedirs(output_dir, exist_ok=True)


write_tag_and_relation_predictions_to_folder(
    output_dir='out',
    eval_examples_path='examples/test.json',
    eval_predictions_path='output/_predictions.tsv',
    prefix_part='test_full',
    force_write=True,
    postprocess=False
)
