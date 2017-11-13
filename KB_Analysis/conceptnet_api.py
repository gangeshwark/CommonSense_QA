import requests
from pprint import pprint

entity = 'hand'
obj = requests.get('http://api.conceptnet.io/c/en/' + entity).json()
# pprint(obj)

for x in obj['edges']:
    print(entity, '-', x['rel']['label'], '-', x['end']['label'])
