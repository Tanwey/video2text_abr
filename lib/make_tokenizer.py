import sentencepiece as spm

spm.SentencePieceTrainer.Train('--input=../MVAD/TrainCorpus.txt --model_prefix=tokenizer --vocab_size=5000 --pad_id=0 --unk_id=3 --bos_id=1, --eos_id=2')