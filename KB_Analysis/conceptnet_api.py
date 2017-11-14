import requests
from pprint import pprint

entities = ['shovel', 'dig']
for entity in entities:
    obj = requests.get('http://api.conceptnet.io/c/en/' + entity + '?offset=0&limit=1000').json()
    pprint(len(obj['edges']))
    for x in obj['edges']:
        if x['end']['label'] == entity:
            continue
        print(entity, '-', x['rel']['label'], '-', x['end']['label'])
    print("\n")
