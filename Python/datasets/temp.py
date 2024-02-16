import json
import argparse

# Global variables
parser = argparse.ArgumentParser(description='')
parser.add_argument('--hand', help='If you collect the left hand data, set "left", else "right".', choices=['left', 'right'], required=True)
hand = parser.parse_args().hand

if __name__ == '__main__':
    size_list = []
    with open(f'./dataset_{hand}.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    new_dataset = {}
    for data in dataset['data']:
        size = len(data['acc'])
        if not size in size_list:
            size_list.append(size)
            new_dataset[size] = {'data':[]}
        new_dataset[size]['data'].append(data)
        
    for size in size_list:
        with open(f'./{hand}/dataset_{hand}_{size}.json', 'w', encoding='utf-8') as f:
            json.dump(new_dataset[size], f, indent=4)