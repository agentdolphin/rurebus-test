import json
import os.path

import examples_generator

RUREBUS_DATA_DIR = 'train_data'
RUREBUS_DATA_PARTS = ['train_part_1', 'train_part_2', 'train_part_3']
EXAMPLES_PATH = 'examples'
EXAMPLES_FILE = 'train.json'

examples = []
with open(os.path.join(EXAMPLES_PATH, EXAMPLES_FILE), 'w', encoding='utf-8') as writer:
    for part in RUREBUS_DATA_PARTS:
        part_examples = examples_generator.get_tag_and_relation_dataset(RUREBUS_DATA_DIR, part)
        examples.extend(part_examples)
    writer.write(json.dumps(examples))
