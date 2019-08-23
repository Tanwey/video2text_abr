import torch
from models.transformer import create_look_ahead_mask
from train import accuracy_metrics
import random
from utils import BeamSearch
import numpy as np

class Captioner:
    def __init__(self, model, sp, tar_max_seq_length, top_n=1):
        self.model = model.eval()
        self.sp = sp
        self.tar_max_seq_length = tar_max_seq_length
        self.top_n = top_n
        self.start_token = torch.LongTensor([self.sp.piece_to_id('<s>')]).view(1, 1)
        self.end_token = torch.LongTensor([self.sp.piece_to_id('</s>')]).view(1, 1)
        
    def _evaluate(self, feature, inp_key_padding_mask, mem_key_padding_mask):
        output = self.start_token
        
        for i in range(1, self.tar_max_seq_length):
            tar_attn_mask = create_look_ahead_mask(i)
            prediction = self.model(feature, output, inp_key_padding_mask, mem_key_padding_mask=mem_key_padding_mask, tar_attn_mask=tar_attn_mask)
            prediction = prediction[:, -1:, :]
            rand_int = random.randint(0, self.top_n - 1)
            predicted_id = prediction.topk(self.top_n, dim=-1)[1][:, :, rand_int]
            if predicted_id == self.end_token:
                return output.squeeze(0)
            output = torch.cat((output, predicted_id), dim=1)
        return output.squeeze(0)

    def id_to_string(self, predicted_ids):
        predicted_ids_list = [int(id) for id in predicted_ids.numpy()[predicted_ids.numpy() != 0]]
        return self.sp.decode_ids(predicted_ids_list)
    
    def caption_video(self, feature, inp_key_padding_mask, mem_key_padding_mask=None):
        if mem_key_padding_mask is None:
            mem_key_padding_mask = inp_key_padding_mask
        predicted_ids = self._evaluate(feature, inp_key_padding_mask, mem_key_padding_mask)
        predicted_string = self.id_to_string(predicted_ids)
        return predicted_string

    def caption_video_from_dataloader(self, dataloader, count=10):
        dataiter = iter(dataloader)
        for i in range(count):
            sample = dataiter.next()
            feature = sample[0].transpose(0, 1)
            caption = sample[1]
            video_file = sample[2]
            inp_key_padding_mask = sample[3]
            mem_key_padding_mask = sample[5]
            predicted_string = self.caption_video(feature, inp_key_padding_mask, mem_key_padding_mask)
            print('video: {}\ncaption origin: {}\ncaption predict: {}\n'.format(video_file, self.id_to_string(caption), predicted_string))
        
    
class BeamSearchCaptioner(BeamSearch):
    def __init__(self, model, sp, tar_max_seq_length, k, num_required=None, least_length=3):
        self.sp = sp
        self.start_id = self.sp.piece_to_id('<s>')
        self.end_id = self.sp.piece_to_id('</s>')
        super(BeamSearchCaptioner, self).__init__(k, model, self.start_id, self.end_id, tar_max_seq_length, num_required, least_length)
    
    def id_to_string(self, predicted_ids):
        return self.sp.decode_ids(predicted_ids)
    
    def caption_video(self, feature, inp_key_padding_mask, mem_key_padding_mask):
        model_inp = {'inp': feature.transpose(0, 1), 'inp_key_padding_mask': inp_key_padding_mask, 'mem_key_padding_mask': mem_key_padding_mask}
        predict_ids = super(BeamSearchCaptioner, self).__call__(model_inp)
        return self.id_to_string(predict_ids)
    
    def caption_video_from_dataloader(self, dataloader, count=10):
        dataiter = iter(dataloader)
        for i in range(count):
            sample = dataiter.next()
            feature = sample[0]
            caption = sample[1]
            caption = [int(id) for id in caption.numpy()[caption.numpy() != 0]]
            video_file = sample[2]
            print('video: {}\ncaption origin: {}'.format(video_file, self.id_to_string(caption)))
            inp_key_padding_mask = sample[3]
            mem_key_padding_mask = sample[5]
            predicted_string = self.caption_video(feature, inp_key_padding_mask, mem_key_padding_mask)
            print('caption predict: {}\n'.format(predicted_string))