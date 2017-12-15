#!/usr/bin/env python3
import json
from os.path import isfile

if isfile('data/persona.json'):
    with open('data/persona.json') as f:
        data = json.load(f)
else:
    data = []

intent = input('Name of intent: ')
dialog = input('Name of dialog: ')

print()

try:
    while True:
        question = input('Question: ')
        response = input('Response: ')
        print()

        data.append({
            'input': {
                'sentence': question,
                'dialog': dialog
            },
            'output': {
                'sentence': response,
                'intent': intent
            }
        })
except (KeyboardInterrupt, SystemExit):
    print()

with open('data/persona.json', 'w') as f:
    json.dump(data, f, indent=4)
