import os
# UNUSED?
# sys.path.append('../task6_baseline')
# from data_utils import multi_predictions_postprocessing

# 1 TASK?
# def get_tag_dataset(rurebus_data_dir: str, part: str = 'train_part_3'):
#     """
#     Probably used for NER task (1), and do the same as get_tag_and_relation_dataset does.
#
#     @param rurebus_data_dir:
#     @param part:
#     @return:
#     """
#     rurebus_dir = os.path.join(rurebus_data_dir, part)
#     annotatated_dataset = get_annotated_dataset(rurebus_dir=rurebus_dir)
#     tokenized_dataset = get_tokenized_dataset(rurebus_dir=rurebus_dir)
#
#     dataset = tokenized_dataset if len(tokenized_dataset) < 0 else annotatated_dataset
#     examples = []
#     for annotated_file in dataset:
#         sentences, tokenized_sentences = tokenized_dataset[annotated_file]['file_sentences']
#         for sent_id, (sentence, tokenized_sentence) in enumerate(zip(sentences, tokenized_sentences)):
#             entities_from_span = filter_entities_dict_by_text_span(
#                 annotatated_dataset[annotated_file],
#                 sentence.start, sentence.stop
#             )
#             example_tags = []
#             prev_tag = 'O'
#             for token in tokenized_sentence:
#                 tag, entity_tag = get_entity_tag_from_token_span(entities_from_span, sentence.start + token.start,
#                                                                  sentence.start + token.stop)
#                 if prev_tag == tag:
#                     entity_tag = 'I-' + entity_tag if entity_tag != 'O' else 'O'
#                 else:
#                     prev_tag = tag
#                     entity_tag = 'B-' + entity_tag if entity_tag != 'O' else 'O'
#
#                 example_tags.append(
#                     entity_tag
#                 )
#
#             token = [token.text for token in tokenized_sentence]
#
#             example = {
#                 'token': token,
#                 'tag': example_tags,
#                 'id': f'{part}-{annotated_file}-{sent_id}',
#                 'sentence': sentence,
#                 'tokenized_sentence': tokenized_sentence
#             }
#             examples.append(example)
#
#     return examples

import examples_generator


# 1 TASK? 3 TASK TEST?
def get_tag_test_dataset(rurebus_data_dir: str):
    tokenized_dataset = examples_generator.get_tokenized_dataset(rurebus_dir=rurebus_data_dir)
    examples = []
    for annotated_file in tokenized_dataset:
        sentences, tokenized_sentences = tokenized_dataset[annotated_file]['file_sentences']
        for sent_id, (sentence, tokenized_sentence) in enumerate(zip(sentences, tokenized_sentences)):
            example_tags = ['O'] * len(tokenized_sentence)
            token = [token.text for token in tokenized_sentence]
            example = {
                'token': token,
                'tag': example_tags,
                'id': f'test_part-{annotated_file}-{sent_id}',
                'sentence': sentence,
                'tokenized_sentence': tokenized_sentence
            }
            examples.append(example)
    return examples


# 1 TASK?
# def write_tag_predictions_to_folder(output_dir: str, eval_examples_path, eval_predictions_path,
#                                     force_write=False):
#     """
#     Probably the same as write_tag_and_relation_predictions_to_folder, but for NER task (1)?
#
#     @param output_dir:
#     @param eval_examples_path:
#     @param eval_predictions_path:
#     @param force_write:
#     """
#     check_output_dir(output_dir, force_write)
#
#     eval_examples = read_examples(eval_examples_path)
#     df = pd.read_csv(eval_predictions_path, sep='\t')
#
#     columns_to_split = ['token', 'tag_label', 'tag_pred', 'scores']
#     for column in columns_to_split:
#         if 'scores' in column:
#             df.loc[:, column] = df[column].apply(lambda x: [float(y) for y in x.split(' ')])
#         else:
#             df.loc[:, column] = df[column].apply(lambda x: x.split(' '))
#
#     df.loc[:, 'source_file_id'] = df.id.apply(lambda x: '-'.join(x.split('-')[2:-1]))  # 1:-1 - unique .ann path
#     df.loc[:, 'source_file_name'] = df.id.apply(lambda x: '-'.join(x.split('-')[2:-1]))
#     df.loc[:, 'sent_id_in_source_file'] = df.id.apply(lambda x: '-'.join(x.split('-')[2:]))
#     unique_source_file_ids = df.source_file_id.unique()
#
#     for source_file_id in unique_source_file_ids:
#         predicted_tags_and_examples_tuples = []
#         tmp_df = df[df.source_file_id == source_file_id]
#         tmp_examples = [example for example in eval_examples if
#                         '-'.join(example['id'].split('-')[1:-1]) == source_file_id]
#         assert len(tmp_df.source_file_name.unique()) == 1
#         source_file_name = tmp_df.source_file_name.unique()[0]
#         for row in tmp_df.itertuples():
#             sent_id = row.sent_id_in_source_file
#
#             corresponding_example = [example for example in tmp_examples if example['id'] == f'test_part-{sent_id}'][0]
#             predicted_tags_and_examples_tuples.append((row.tag_pred, corresponding_example))
#
#         write_tag_ann_to_file(os.path.join(output_dir, f"{source_file_name}.ann"), predicted_tags_and_examples_tuples)



