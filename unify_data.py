from pprint import pprint

import pandas as pd
import os
import xml.etree.ElementTree
import io, json

base_data_path = 'KB/'
all_scripts = {}


def get_omcs_data():
    for subdir, dirs, files in os.walk(base_data_path + 'OMCS/'):
        for file in sorted(files):
            scripts = []
            file_path = subdir + os.path.sep + file
            e = xml.etree.ElementTree.parse(file_path).getroot()
            for atype in e.findall('script'):
                script = {'id': atype.get('id')}
                items = []
                for i in atype.findall('item'):
                    item_text = i.get('text')
                    if not '.' in item_text[-2:]:
                        items.append(item_text + ' .')
                    else:
                        items.append(item_text)
                script['items'] = items
                script['text'] = ' '.join(items)
                scripts.append(script)
            all_scripts[file] = scripts


def get_descript_data():
    for subdir, dirs, files in os.walk(base_data_path + 'DS/'):
        for file in sorted(files):
            scripts = []
            file_path = subdir + os.path.sep + file
            print(file_path)
            e = xml.etree.ElementTree.parse(file_path).getroot()
            for atype in e.findall('script'):
                script = {'id': atype.get('id')}
                items = []
                for i in atype.findall('item'):
                    item_text = i.get('original')
                    if not '.' in item_text[-2:]:
                        items.append(item_text + '.')
                    else:
                        items.append(item_text)
                script['items'] = items
                script['text'] = ' '.join(items)
                scripts.append(script)
            all_scripts[file] = scripts


def get_rkp_data():
    for subdir, dirs, files in os.walk(base_data_path + 'RKP/'):
        for file in sorted(files):
            scripts = []
            file_path = subdir + os.path.sep + file
            print(file_path)
            e = xml.etree.ElementTree.parse(file_path).getroot()
            for atype in e.findall('script'):
                script = {'id': atype.get('id')}
                items = []
                for i in atype.findall('item'):
                    item_text = i.get('text')
                    if not '.' in item_text[-2:]:
                        items.append(item_text + '.')
                    else:
                        items.append(item_text)
                script['items'] = items
                script['text'] = ' '.join(items)
                scripts.append(script)
            all_scripts[file] = scripts


if __name__ == '__main__':
    get_omcs_data()
    get_descript_data()
    get_rkp_data()
    with io.open('all_scripts.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(all_scripts, ensure_ascii=False, indent=4))
    pprint(all_scripts)
