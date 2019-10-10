import yaml

load_files = ['data/MVAD/corpus_M-VAD_train.txt',
              'data/MVAD/corpus_M-VAD_val.txt', 'data/MVAD/corpus_M-VAD_test.txt']
save_files = ['data/MVAD/corpus_M-VAD_train.yaml',
              'data/MVAD/corpus_M-VAD_val.yaml', 'data/MVAD/corpus_M-VAD_test.yaml']
for load_file, save_file in zip(load_files, save_files):
    corpus_dict = {}
    with open(load_file, 'r') as lf:
        txt_lines = lf.readlines()
        for txt_line in txt_lines:
            file_name, caption = list(map(str.strip, txt_line.split('\t')))
            corpus_dict.update({file_name: caption})
        with open(save_file, 'w') as sf:
            yaml.dump(corpus_dict, sf)

# with open(save_files[1]) as sf:
#     d1 = yaml.load(sf, Loader=yaml.Loader)
#     print(len(d1))
