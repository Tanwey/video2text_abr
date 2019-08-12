import torch
from models.transformer import create_look_ahead_mask

class Captioner:
    def __init__(self, model, sp, tar_max_seq_length):
        self.model = model.eval()
        self.sp = sp
        self.tar_max_seq_length = tar_max_seq_length
        
    def _evaluate(self, feature, inp_key_padding_mask, mem_key_padding_mask):
        start_token = torch.LongTensor([self.sp.piece_to_id('<s>')]).view(1, 1)
        end_token = torch.LongTensor([self.sp.piece_to_id('</s>')]).view(1, 1)
        output = start_token
        for i in range(1, self.tar_max_seq_length):
            tar_attn_mask = create_look_ahead_mask(i)
            prediction = self.model(feature, output, inp_key_padding_mask, mem_key_padding_mask=mem_key_padding_mask, tar_attn_mask=tar_attn_mask)
            prediction = prediction[:, -1:, :]
            predicted_id = prediction.max(dim=-1)[1]
            if predicted_id == end_token:
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
        for i in range(10):
            sample = dataiter.next()
            feature = sample[0].transpose(0, 1)
            caption = sample[1]
            video_file = sample[2]
            inp_key_padding_mask = sample[3]
            mem_key_padding_mask = sample[5]
            predicted_string = self.caption_video(feature, inp_key_padding_mask, mem_key_padding_mask)
            print('video: {} caption origin: {} caption predict: {}'.format(video_file, self.id_to_string(caption), predicted_string))