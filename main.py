import json
import os.path

import examples_generator

RUREBUS_DATA_DIR = 'train_data'
RUREBUS_DATA_PARTS = ['train_part_1', 'train_part_2', 'train_part_3']
EXAMPLES_PATH = 'examples'
EXAMPLES_FILE = 'train.json'
EXAMPLES_DEV_FILE = 'dev.json'


examples = []
with open(os.path.join(EXAMPLES_PATH, EXAMPLES_FILE), 'w', encoding='utf-8') as writer, open(os.path.join(EXAMPLES_PATH, EXAMPLES_DEV_FILE), 'w', encoding='utf-8') as dev_writer:
    for part in RUREBUS_DATA_PARTS:
        part_examples = examples_generator.get_tag_and_relation_dataset(RUREBUS_DATA_DIR, part)
        examples.extend(part_examples)
    dev_examples = [example for example in examples if example % 10 == 0]
    examples = [example for example in examples if example % 10 != 0]
    writer.write(json.dumps(examples))
    dev_writer.write(json.dumps(dev_examples))

RUREBUS_TEST_DATA_DIR = 'test_data'
RUREBUS_TEST_DATA_PARTS = ['test_full']
EXAMPLES_TEST_FILE = 'test.json'

examples = []
with open(os.path.join(EXAMPLES_PATH, EXAMPLES_TEST_FILE), 'w', encoding='utf-8') as writer:
    for part in RUREBUS_TEST_DATA_PARTS:
        part_examples = examples_generator.get_tag_and_relation_dataset(RUREBUS_TEST_DATA_DIR, part)
        examples.extend(part_examples)
    writer.write(json.dumps(examples))
