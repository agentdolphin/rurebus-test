import os
from collections import namedtuple
from razdel import sentenize, tokenize

SpanText = namedtuple('SpanText', ['start', 'stop', 'text'])


def sentenize_wrap(text):
    """
        Returns Spantexts for sentences per text.

        @param text:
        @return: list of SpanTexts, one per sentence
    """
    return [
        SpanText(start=sentence.start, stop=sentence.stop, text=sentence.text) for sentence in sentenize(text)
    ]


def tokenize_wrap(sentence):
    """
        Returns Spantexts for words per sentence.

        @param sentence:
        @return: list of SpanTexts, one per word
        """
    return [
        SpanText(start=token.start, stop=token.stop, text=token.text) for token in tokenize(sentence)
    ]


def get_tokenized_dataset(rurebus_dir: str):
    """
        Parses .txt files and extracting sentences with razdel.

        @param rurebus_dir: path to train files
        @return: list of 'filename' => (file_sentences, file_tokenized_sentences)
        file_sentences: list of Spantext, one per sentence.
        file_tokenized_sentences: list of lists of Spantext, one Spantext per word, one list of Spantext per sentence.
    """
    annotations = [file.rstrip('tx') for file in os.listdir(rurebus_dir) if file.endswith('.txt')]
    result = {}
    for annotation_file in annotations:
        file_txt = open(os.path.join(rurebus_dir, annotation_file) + 'txt', 'r', encoding='utf-8').read()
        file_sentences = [sentence for sentence in sentenize_wrap(file_txt)]
        file_tokenized_sentences = [[tok for tok in tokenize_wrap(sentence.text)] for sentence in file_sentences]

        result[annotation_file.rstrip('.')] = {
            'file_sentences': (file_sentences, file_tokenized_sentences)
        }
    return result


def get_annotated_dataset(rurebus_dir: str):
    """
        Parses .ann files and extracting annotated info.

        @param rurebus_dir: path to train files
        @return: list of 'filename' => {entities_map, relations_map, subj_to_obj_map}, one per file.
        entities_map: 'T2' => ('MET', (146, 180), 'эфффективности бюджетных инвестиций')
        relations_map: 'R1' => ('FPS', 'T3', 'T2')
        subj_to_obj_map: 'T3' => set{('T2', 'FPS')}, one element per entity. This is set of arg2's per every arg1.
        """
    annotations = [file.rstrip('an') for file in os.listdir(rurebus_dir) if file.endswith('.ann')]
    result = {}
    for annotated_file in annotations:
        annotated_files_lines_list = [line.strip() for line in open(os.path.join(rurebus_dir, annotated_file) + 'ann', 'r', encoding='utf-8').readlines()]
        annotated_files_lines_list = [tuple(line.split('\t')) for line in annotated_files_lines_list]
        entities = [line for line in annotated_files_lines_list if line[0].startswith('T')]
        unparsed_entities = set([line[0] for line in entities if len(line[1].split()) != 3])
        entities = [(line[0], line[1], '\t'.join(line[2:])) for line in entities if line[0] not in unparsed_entities]
        try:
            entities_map = dict(
                [(tag, (attrs.split(' ')[0], tuple([int(position) for position in attrs.split(' ')[1:]]), tag_text)) for
                 (tag, attrs, tag_text) in entities])
        except ValueError:
            print(annotated_file)
            entities_map = dict(
                [(tag, (attrs.split(' ')[0], tuple(
                    [int(position) if ';' not in position else int(position.split(';')[0]) for position in
                     attrs.split(' ')[1:]]), tag_text)) for
                 (tag, attrs, tag_text) in entities])
            # raise ValueError

        relations = [line for line in annotated_files_lines_list if line[0].startswith('R')]
        relations_arg1 = [line[1].split(' ')[1][5:] for line in relations]
        relations_arg2 = [line[1].split(' ')[2][5:] for line in relations]
        relations = [line for l_arg, r_arg, line in zip(relations_arg1, relations_arg2, relations) if
                     r_arg not in unparsed_entities or r_arg not in unparsed_entities]
        relations_map = dict([
            (
                tag,
                tuple([attr if attr_id == 0 else attr.split(':')[-1] for attr_id, attr in enumerate(attrs.split(' '))]))
            for (tag, attrs) in relations
        ])

        new_relation_map = {}
        for tag, (relation_type, subj, obj) in relations_map.items():
            if subj not in new_relation_map:
                new_relation_map[subj] = set()
            new_relation_map[subj].add((obj, relation_type))

        result[annotated_file.rstrip('.')] = {
            'entities_map': entities_map,
            'relations_map': relations_map,
            'subj_to_obj_map': new_relation_map
        }

    return result


def filter_entities_dict_by_text_span(entities_dict, span_start, span_end):
    """
        Get entities in sentence by (sentence_start_index, sentence_stop_index) and dict from annotated dataset

        @param entities_dict: dict from get_annotated_dataset
        @param span_start: index of sentence start
        @param span_end: index of sentence end
        @return: entities_dict[entities_map], but filtered by sentence
    """
    entities_from_span = [
        tag for tag, (_, (start, end), _) in entities_dict['entities_map'].items() if
        span_start <= start <= end <= span_end
    ]
    result = {
        'entities_map': dict([(tag, entities_dict['entities_map'][tag]) for tag in entities_from_span])
    }
    return result


def get_entity_tag_from_token_span(entities_dict, token_start, token_end):
    """
        Get entity from token by (token_start_index, token_end_index) and dict from annotated sentence entities from filter_entities_dict_by_text_span

        @param entities_dict: dict from filter_entities_dict_by_text_span
        @param token_start: index of token start
        @param token_end: index of token end
        @return: (tag, entity_tag)
            example: ('T9', 'B-ACT')
    """
    result = [
        (tag, entity_tag) for tag, (entity_tag, (start, end), _) in entities_dict['entities_map'].items() if
        start <= token_start <= token_end <= end
    ]
    if len(result) > 0:
        result = result[0]
    else:
        result = ('O', 'O')
    return result


def get_tag_and_relation_dataset(rurebus_data_dir: str, part: str = 'train_part_3'):
    """
        Processes rurebus_data_dir/train_part_3 directory and get all the examples, probably to send it for
        BERT-multilearning further.

        One example per finded entity!!!
        example = {
                    'sent_type': 1, const
                    'sent_start': -1, const
                    'sent_end': -1, const
                    'token': token, ['Модель', 'управления', 'государственными', 'инвестициями', ...] -- all sentence
                    'tag': example_tags, ['B-ACT', 'I-ACT', 'B-ECO', 'I-ECO', ...] -- all sentence
                    'id': f'{part}-{annotated_file}-{sent_id}-{subj}', train_part_3-0-T3
                    'sentence': sentence, SpanText(sentence)
                    'tokenized_sentence': tokenized_sentence, [SpanText(token), ...] -- all sentence
                    'subj_start': subj_start, 17 -- index in sentence, inclusive
                    'subj_end': subj_end, 17 -- index in sentence, inclusive
                    'relation': relations, ['0', '0', ..., '0', '0', '0', 'FPS', 'FPS', 'FPS', '0'] -- all sentence
                }


        @param rurebus_data_dir: rurebus directory path
        @param part: train_part folder name inside rurebus directory path
        @return:
    """
    rurebus_dir = os.path.join(rurebus_data_dir, part)
    annotated_dataset = get_annotated_dataset(rurebus_dir=rurebus_dir)
    tokenized_dataset = get_tokenized_dataset(rurebus_dir=rurebus_dir)

    examples = []
    for annotated_file in tokenized_dataset:
        sentences, tokenized_sentences = tokenized_dataset[annotated_file]['file_sentences']
        for sent_id, (sentence, tokenized_sentence) in enumerate(zip(sentences, tokenized_sentences)):
            entities_from_span = filter_entities_dict_by_text_span(
                annotated_dataset[annotated_file],
                sentence.start, sentence.stop
            )
            example_tags, tag_marks = [], []
            finded_entities = set()
            prev_tag = 'O'
            for token in tokenized_sentence:
                tag, entity_tag = get_entity_tag_from_token_span(entities_from_span, sentence.start + token.start,
                                                                 sentence.start + token.stop)
                if prev_tag == tag:
                    entity_tag = 'I-' + entity_tag if entity_tag != 'O' else 'O'
                else:
                    prev_tag = tag
                    entity_tag = 'B-' + entity_tag if entity_tag != 'O' else 'O'

                example_tags.append(
                    entity_tag
                )
                tag_marks.append(tag)
                if tag.startswith('T'):
                    finded_entities.add(tag)

            token = [token.text for token in tokenized_sentence]
            subj_to_obj_rel = annotated_dataset[annotated_file]['subj_to_obj_map']
            for relation_id, subj in enumerate(finded_entities):
                subj_ids = [i for i, tag in enumerate(tag_marks) if tag == subj]
                if subj in subj_to_obj_rel:
                    objs = subj_to_obj_rel[subj]
                else:
                    objs = set()
                relations = ['0'] * len(token)
                for (obj, relation) in objs:
                    obj_ids = [i for i, tag in enumerate(tag_marks) if tag == obj]
                    for i in obj_ids:
                        relations[i] = relation

                subj_start, subj_end = subj_ids[0], subj_ids[-1]
                example = {
                    'sent_type': 1,
                    'sent_start': -1,
                    'sent_end': -1,
                    'token': token,
                    'tag': example_tags,
                    'id': f'{part}-{annotated_file}-{sent_id}-{subj}',
                    # 'id': f'{part}-{annotated_file}-{sent_id}-{"_".join([str(subj_start), str(subj_end)])}',
                    'sentence': sentence,
                    'tokenized_sentence': tokenized_sentence,
                    'subj_start': subj_start,
                    'subj_end': subj_end,
                    'relation': relations
                }
                examples.append(example)
    return examples
