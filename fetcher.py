# this script downloads plain .txt texts of specified articles to corresponding directories

import subprocess
import os

# ./articles/<folder>/<train/test>: [<article ids>]
article_ids = {
    'diseases': ['PMC5363789', 'PMC6632217', 'PMC7114954', 'PMC5477351', 'PMC5530939', # "training" articles
                 'PMC8219168', 'PMC6013753', 'PMC7820105'], # testing articles
    'food': ['PMC6137369', 'PMC5561571', 'PMC6131755', 'PMC6542079', 'PMC6356196', # "training" articles
             'PMC6627945', 'PMC7442364', 'PMC6675761'] # testing articles
    }

# dictionary to a list of pairs e.g. ('food', 'PMC6137369'), ('food', 'PMC5561571') etc.
article_ids = [(key, val) for key, values in article_ids.items() for val in values]

for dir, article_id in article_ids:
    if os.path.exists(f'./articles/{dir}/{article_id}.txt'):
        print(f'./articles/{dir}/{article_id}.txt already fetched...')
        continue

    cmd = ['aws', 's3', 'cp', f's3://pmc-oa-opendata/oa_comm/txt/all/{article_id}.txt',
                    f'./articles/{dir}/{article_id}.txt', '--no-sign-request']

    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError:
        try:
            print('Trying oa_noncomm directory instead...')
            cmd[3] = f's3://pmc-oa-opendata/oa_noncomm/txt/all/{article_id}.txt'
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError:
            print(f'{article_id} was not found, skipping to the next one...')
