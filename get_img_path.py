import pandas as pd
import sys
import os
from os.path import join

def main(samples_frame_json):
    samples_frame = pd.read_json(samples_frame_json, lines =True)

    samples_frame['img_path'] = '../hateful_memes/' + samples_frame['img']

    out = samples_frame['img_path']

    out_path = join("./for_fairface", samples_frame_json)
    out.to_csv(out_path, encoding = "utf-8", index = False)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage:')
        print('  python {} <memes json file>'.format(sys.argv[0]))
    
    main(sys.argv[1])