import torch.nn as nn
import torch.nn.functional as F


class EmbeddingModel(nn.Module):

    def __init__(self, vocab_size):
        CONTEXT_SIZE = 2
        EMBEDDING_DIM = 10
        super(EmbeddingModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, EMBEDDING_DIM)
        self.linear1 = nn.Linear(CONTEXT_SIZE * EMBEDDING_DIM, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs
