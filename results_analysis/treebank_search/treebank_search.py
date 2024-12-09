import conllu
from collections import Counter

aspect_counter, aspect_counter_SOV = Counter(), Counter()

with open('hi_pud-ud-test.conllu', 'r', encoding='utf-8') as f:
    data = conllu.parse(f.read())

for sentence in data:
    for node in sentence:
        if node['upos'] == 'VERB':
            verb = node
            if verb['feats'] is not None and 'Aspect' in verb['feats']:
                aspect_counter[verb['feats']['Aspect']] += 1
            verb_id = node['id']
            subjects = [n for n in sentence if n['head'] == verb_id and n['deprel'] == 'nsubj']
            objects = [n for n in sentence if n['head'] == verb_id and n['deprel'] == 'obj']
            if subjects and objects:
                subject = subjects[0]
                object = objects[0]
                if subject['id'] < object['id'] < verb_id:
                    if verb['feats'] is not None and 'Aspect' in verb['feats']:
                        aspect_counter_SOV[verb['feats']['Aspect']] += 1

print("Aspect counts among all verbs:")
for aspect, count in aspect_counter.items():
    print(f"  {aspect}: {count}")

print("Aspect counts among verbs in SOV clauses:")
for aspect, count in aspect_counter_SOV.items():
    print(f"  {aspect}: {count}")

