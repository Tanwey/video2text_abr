import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.mask import create_look_ahead_mask


class BeamNode:
    def __init__(self, token_id, pre_node=None, log_prob=None, length=None):
        self.token_id = token_id
        self.pre_node = pre_node
        if log_prob is None:
            log_prob = 0
        self.log_prob = log_prob
        if length is None:
            length = 1
        self.length = length

    def score(self):
        ''' Get the score of node
            Return:
              log_prb / count of node
        '''
        return self.log_prob / (self.length - 1 + 1e-6)

    def get_ids(self):
        '''
            Returns:
              token_ids: List[int]
        '''
        token_ids = []
        current_node = self
        while True:
            token_ids.insert(0, current_node.token_id)
            if current_node.pre_node is None:
                break
            current_node = current_node.pre_node
        return token_ids


class BeamSearch:
    def __init__(self, k, model, start_id, end_id, max_length, min_length=None, num_required=None):
        '''
            Args:
              k: Beam search size
              model: Neural Net to beam search
              start_id: Tokenizer id of start token <s>
              end_id: Tokenizer id of end token </s>
              max_length: Maximum length of ids (include <s> and </s>)
              min_length: Minimun lenght of ids (include <s> and </s>)
              num_required: Searching ends when count of saved nodes are larger than num_required
                default None

            If you want to reuse this change <1>
        '''
        self.k = k
        self.model = model
        self.start_id = start_id
        self.end_id = end_id
        self.max_length = max_length
        if min_length is None:
            min_length = 2
        self.min_length = min_length
        if num_required is None:
            num_required == 99999999
        self.num_required = num_required

        self.leaf_nodes = [BeamNode(start_id)]
        self.count = 1
        self.end_nodes = []

    def _step(self, model_inp):
        '''
            BATCH SIZE MUST BE 1
            Model should return the tensor of shape (batch, seq, vocab_size)
        '''

        next_leaf_nodes = []
        for i, leaf_node in enumerate(self.leaf_nodes):
            # Save to end_nodes if end id is predicted
            if leaf_node.token_id == self.end_id and leaf_node.length > self.min_length:
                self.end_nodes.append(leaf_node)

            # Predict log prob
            # <1> start
            model_inp['tar'] = torch.tensor(
                leaf_node.get_ids(), dtype=torch.int64).unsqueeze(0)
            model_inp['tar_attn_mask'] = create_look_ahead_mask(self.count)
            # <1> end
            prediction = F.log_softmax(self.model(**model_inp), dim=-1)

            top_k = prediction.topk(self.k, dim=-1)
            value_top_k = top_k[0][:, -1, :].squeeze(0).tolist()
            argtop_k = top_k[1][:, -1, :].squeeze(0).tolist()
            for value, token_id in zip(value_top_k, argtop_k):
                log_prob = leaf_node.log_prob + value
                if len(next_leaf_nodes) == self.k:
                    index = None
                else:
                    index = 0

                for j in range(self.k):
                    # When next_leaf_nodes are not full and prob is biggest
                    if len(next_leaf_nodes) == j:
                        index = j
                        break

                    if log_prob <= next_leaf_nodes[j].log_prob:
                        break

                    index = j

                if index is not None:
                    if len(next_leaf_nodes) == self.k:
                        next_leaf_nodes.pop(0)
                    next_leaf_nodes.insert(index, BeamNode(
                        token_id, leaf_node, log_prob, leaf_node.length + 1))

        self.leaf_nodes = next_leaf_nodes
        self.count += 1

    def __call__(self, model_inp):
        self.leaf_nodes = [BeamNode(self.start_id)]
        self.count = 1
        self.end_nodes = []

        for i in range(self.max_length - 1):
            self._step(model_inp)
            if len(self.end_nodes) >= self.num_required:
                break
            print('step: {}, end_nodes: {}'.format(
                self.count, len(self.end_nodes)))

        # If end token is not predicted
        if len(self.end_nodes) == 0:
            print(self.count, 'no end node')
            best_node = self.leaf_nodes[0]
            for node in self.leaf_nodes[1:]:
                if best_node.score() < node.score():
                    best_node = node
            return best_node.get_ids()

        best_node = self.end_nodes[0]
        for end_node in self.end_nodes[1:]:
            if best_node.score() < end_node.score():
                best_node = end_node

        return best_node.get_ids()
