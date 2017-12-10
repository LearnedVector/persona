import json

with open('./data/persona.dialog.json', 'r') as f:
    data = json.load(f)

new_list = []
addded_list = []
for pair in data:
    if pair["input"] not in addded_list:
        new_list.append(pair)
        addded_list.append(pair["input"])

with open('./data/__persona.dialog.json', 'w') as f:
    json.dump(new_list, f)
    