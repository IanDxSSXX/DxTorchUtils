from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_attention_mask_pad(q_words, k_words=None, padding_idx=0):
    """
    根据q和k的句子长度，生成padding mask
    :param k_words: 要作为key的words (batch_size, len_key_sentence)
    :param q_words: 要作为query的words (batch_size, len_query_sentence)
    :param padding_idx: padding对应的序号
    :return:
    """
    if k_words is None:
        batch_size, len_query_sentence = q_words.shape
        len_key_sentence = len_query_sentence
    else:
        batch_size, len_query_sentence = q_words.shape
        _, len_key_sentence = k_words.shape

    attention_mask_pad = q_words.data.eq(padding_idx).unsqueeze(1)
    # attention_mask_pad (batch_size, 1, len_query_sentence)
    attention_mask_pad = attention_mask_pad.expand(batch_size, len_key_sentence, len_query_sentence)
    # attention_mask_pad (batch_size, len_key_sentence, len_query_sentence)

    return attention_mask_pad


def get_attention_mask_subsequence(words):
    """
    subsequent的操作是每次只能多用一个词，后面的词全屏蔽
    :param words:
    :return:
    """
    batch_size, len_sentence = words.shape
    # torch.triu 下三角矩阵 diagonal=1即从索引为1开始；
    # [[False,  True,  True,  True]
    #  [False, False,  True,  True]
    #  [False, False, False,  True]
    #  [False, False, False, False]]
    attention_mask_subsequence = torch.triu(
        torch.ones((batch_size, len_sentence, len_sentence), dtype=bool),
        diagonal=1
    )

    return attention_mask_subsequence


def get_attention_mask_subsequence_pad(words):
    """
    :param words:
    :return:
    """
    attention_mask_pad = get_attention_mask_pad(words)
    attention_mask_subsequence = get_attention_mask_subsequence(words)

    # 逻辑或，如果要因为padding而mask掉或因为subsequence而mask掉，则为True
    attention_mask_subsequence_pad = torch.logical_or(attention_mask_pad, attention_mask_subsequence)

    return attention_mask_subsequence_pad


class PositionalEncoding(nn.Module):
    def __init__(self, len_embedding, rate_dropout=0.1, max_len=5000):
        """
        :param len_embedding:
        :param max_len: 为了使得生成和句子数一样多的position encoding，len_sentence < max_len，然后截取前len_sentence个
        :param rate_dropout:
        """
        super(PositionalEncoding, self).__init__()

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        if len_embedding % 2 == 0:
            div_term_odd = div_term_even = torch.exp(
                (torch.arange(0, len_embedding, 2, dtype=torch.float) * -(math.log(10000.0) / len_embedding))
            )
        else:
            div_term_odd = torch.exp(
                (torch.arange(0, len_embedding, 2, dtype=torch.float) * -(math.log(10000.0) / len_embedding))
            )
            div_term_even = torch.exp(
                (torch.arange(0, len_embedding - 1, 2, dtype=torch.float) * -(math.log(10000.0) / len_embedding))
            )

        pe = torch.randn(max_len, len_embedding)
        pe[:, 0::2] = torch.sin(position * div_term_odd)
        pe[:, 1::2] = torch.cos(position * div_term_even)
        pe = torch.unsqueeze(pe, 0)
        # pe: (1, len_embedding, max_len)

        self.register_buffer("pe", pe)
        self.len_embedding = len_embedding
        self.dropout = nn.Dropout(rate_dropout)


    def forward(self, word_embedding):
        """
        :param word_embedding:
        :return:
        """
        # -------------------- word embedding 和 positional coding 相加，再dropout ------------------------------------ #
        embedding = word_embedding + self.pe[:, :word_embedding.size(1)]

        outputs = self.dropout(embedding)
        # ------------------------------------------------------------------------------------------------------------ #

        return outputs


class MultiHeadAttention(nn.Module):
    def __init__(self, len_embedding, num_heads, rate_dropout=0.1, len_vec=0):
        super(MultiHeadAttention, self).__init__()
        if len_vec == 0:
            assert int(len_embedding / num_heads) != 0, \
                "Please set Q/K/V vector length, or make len_embedding / num_heads > 0"
            len_vec = int(len_embedding / num_heads)

        self.len_embedding = len_embedding
        self.num_heads = num_heads
        self.len_vec = len_vec

        self.W_Q = nn.Linear(len_embedding, len_vec * num_heads)
        self.W_K = nn.Linear(len_embedding, len_vec * num_heads)
        self.W_V = nn.Linear(len_embedding, len_vec * num_heads)
        self.fc = nn.Linear(num_heads * len_vec, len_embedding)
        self.dropout = nn.Dropout(rate_dropout)

        self.layer_norm = nn.LayerNorm(len_embedding)

    def forward(self, Q, K, V, attention_mask):
        """
        :param Q: (batch_size, len_sentence, len_embedding)
        :param K: (batch_size, len_sentence, len_embedding)
        :param V: (batch_size, len_sentence, len_embedding)
        :param attention_mask: padding 或者 sequence mask (batch_size, sentence_len, sentence_len)
        :return:
        """
        residual, batch_size = Q, Q.size(0)

        # -------------------- 单头到多头 ----------------------------------------------------------------------------- #
        # Q / K / V: (batch_size, sentence_len, len_embedding)
        # attention_mask: (batch_size, sentence_len, sentence_len)
        Qs, Ks, Vs, attention_mask = self.multiHead(Q, K, V, attention_mask, batch_size)
        # Q / K / V: (batch_size, num_heads, len_sentence, len_vec)
        # attention_mask: (batch_size, num_heads, sentence_len, sentence_len)
        # ------------------------------------------------------------------------------------------------------------ #

        # -------------------- 矩阵相乘算attention -------------------------------------------------------------------- #
        x = self.scaledDotProduct(Qs, Ks, Vs, attention_mask)
        # x: (batch_size, num_heads, len_sentence, len_vec)
        # attention: (batch_size, num_heads, len_vec, len_sentence)
        # ------------------------------------------------------------------------------------------------------------ #

        # -------------------- 全连接让输入size和输出size相等，并做残差 -------------------------------------------------- #
        outputs = self.fullyConnected(x, residual, batch_size)
        # outputs: (batch_size, len_sentence, len_embedding)
        # ------------------------------------------------------------------------------------------------------------ #

        return outputs


    def multiHead(self, Q, K, V, attention_mask, batch_size):
        """
        把单头变成多头
        :param Q: (batch_size, len_sentence, len_embedding)
        :param K: (batch_size, len_sentence, len_embedding)
        :param V: (batch_size, len_sentence, len_embedding)
        :param attention_mask: (batch_size, sentence_len, sentence_len)
        :param batch_size:
        :return:
        """
        #   Q / K / V  (batch_size, sentence_len, len_embedding)
        # -----W-----> (batch_size, len_sentence, num_heads * len_vec)
        # ----view---> (batch_size, len_sentence, num_heads, len_vec)
        # -transpose-> (batch_size, num_heads, len_sentence, len_vec)
        Qs = self.W_Q(Q).view(batch_size, -1, self.num_heads, self.len_vec).transpose(1, 2)
        Ks = self.W_K(K).view(batch_size, -1, self.num_heads, self.len_vec).transpose(1, 2)
        Vs = self.W_V(V).view(batch_size, -1, self.num_heads, self.len_vec).transpose(1, 2)

        # attention_mask: (batch_size, sentence_len, sentence_len)
        attention_mask = attention_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        # attention_mask: (batch_size, num_heads, sentence_len, sentence_len)

        return Qs, Ks, Vs, attention_mask


    def scaledDotProduct(self, Qs, Ks, Vs, attention_mask):
        """
        矩阵相乘算attention
        :param Qs: (batch_size, num_heads, len_sentence, len_vec)
        :param Ks: (batch_size, num_heads, len_sentence, len_vec)
        :param Vs: (batch_size, num_heads, len_sentence, len_vec)
        :param attention_mask: padding 或者 sequence mask (batch_size, sentence_len, sentence_len)
        :return:
        """
        # attention_scores: (batch_size, num_heads, len_vec, len_vec)
        attention_scores = torch.matmul(Qs, Ks.transpose(-1, -2)) / np.sqrt(self.len_vec)

        # 如果mask是False，就是padding，给他变成负无穷
        attention_scores.masked_fill_(attention_mask, float("-inf"))
        # softmax得attention概率
        attention = F.softmax(attention_scores, -1)
        # outputs:
        #       (batch_size, num_heads, len_sentence, len_vec) x (batch_size, num_heads, len_vec, len_vec)
        #    -> (batch_size, num_heads, len_sentence, len_vec)
        outputs = torch.matmul(attention, Vs)

        return outputs


    def fullyConnected(self, x, residual, batch_size):
        """
        全连接让输入size和输出size相等，并做残差
        :param x: (batch_size, num_heads, len_sentence, len_vec)
        :param residual: (batch_size, num_heads, len_sentence, len_vec)
        :param batch_size:
        :return:
        """
        # x: (batch_size, num_heads, len_sentence, len_vec)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.len_vec)
        # x: (batch_size, len_sentence, num_heads * len_vec)
        x = self.fc(x)
        # x: (batch_size, len_sentence, len_embedding)
        outputs = self.layer_norm(x + residual)
        # outputs: (batch_size, len_sentence, len_embedding)

        return outputs



class PositionWiseFeedForward(nn.Module):
    def __init__(self, len_embedding, len_feedforward, activation, rate_dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.fc = nn.Sequential(
            OrderedDict([
                ("fc0", nn.Linear(len_embedding, len_feedforward)),
                ("activation", activation),
                ("dropout0", nn.Dropout(rate_dropout)),
                ("fc1", nn.Linear(len_feedforward, len_embedding)),
                ("dropout1", nn.Dropout(rate_dropout)),
            ])
        )

        self.layer_norm = nn.LayerNorm(len_embedding)

    def forward(self, inputs):
        """
        :param inputs: (batch_size, len_sentence, len_embedding)
        :return:
        """
        residual = inputs
        # -------------------- 全连接层让输入与输出size相等并做残差 ------------------------------------------------------ #
        # inputs: (batch_size, len_sentence, len_embedding)
        x = self.fc(inputs)
        # x: (batch_size, len_sentence, len_embedding)
        outputs = self.layer_norm(x + residual)
        # outputs: (batch_size, len_sentence, len_embedding)
        # ------------------------------------------------------------------------------------------------------------ #

        return outputs



class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        len_embedding,
        num_heads,
        len_feedforward,
        activation,
        rate_dropout=0.1,
        len_vec=0,
    ):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(len_embedding, num_heads, rate_dropout, len_vec)
        self.feed_forward = PositionWiseFeedForward(len_embedding, len_feedforward, activation, rate_dropout)


    def forward(self, source_embedding, attention_mask_pad):
        """
        :param source_embedding: (batch_size, len_sentence, len_embedding)
        :param attention_mask_pad: padding mask (batch_size, sentence_len, sentence_len)
        :return:
        """
        # -------------------- 根据source words embedding做attention并做padding的mask，排除padding的影响 ---------------- #
        x = self.self_attention(source_embedding, source_embedding, source_embedding, attention_mask_pad)
        # x: (batch_size, len_sentence, len_embedding)
        # ------------------------------------------------------------------------------------------------------------ #

        # -------------------- 做feed forward 得到输出和输入一样的size -------------------------------------------------- #
        outputs = self.feed_forward(x)
        # outputs: (batch_size, len_sentence, len_embedding)
        # ------------------------------------------------------------------------------------------------------------ #

        return outputs



class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        len_embedding,
        num_heads,
        len_feedforward,
        activation,
        rate_dropout=0.1,
        len_vec=0,
    ):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(len_embedding, num_heads, rate_dropout, len_vec)
        self.decoder_encoder_attention = MultiHeadAttention(len_embedding, num_heads, rate_dropout, len_vec)
        self.feed_forward = PositionWiseFeedForward(len_embedding, len_feedforward, activation, rate_dropout)


    def forward(self, target_embedding, encoder_output, attention_mask_subsequence_pad, attention_mask_pad):
        """
        :param target_embedding: (batch_size, len_sentence, len_embedding)
        :param encoder_output: (batch_size, len_sentence, len_embedding)
        :param attention_mask_subsequence_pad: subsequence mask (batch_size, sentence_len, sentence_len)
        :param attention_mask_pad: padding mask (batch_size, sentence_len, sentence_len)
        :return:
        """
        # -------------------- 根据target words embedding做attention并做padding的mask，排除padding的影响 ---------------- #
        x = self.self_attention(target_embedding, target_embedding, target_embedding, attention_mask_subsequence_pad)
        # x: (batch_size, len_sentence, len_embedding)
        # ------------------------------------------------------------------------------------------------------------ #

        # -------------------- 根据source words embedding做attention并做padding的mask，排除padding的影响 ---------------- #
        x = self.decoder_encoder_attention(x, encoder_output, encoder_output, attention_mask_pad)
        # x: (batch_size, len_sentence, len_embedding)
        # ------------------------------------------------------------------------------------------------------------ #

        # -------------------- 做feed forward 得到输出和输入一样的size -------------------------------------------------- #
        outputs = self.feed_forward(x)
        # outputs: (batch_size, len_sentence, len_embedding)
        # ------------------------------------------------------------------------------------------------------------ #

        return outputs



class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_vocabularies,
        len_embedding,
        num_encoder_layers,
        num_heads,
        len_feedforward,
        activation,
        padding_idx=1,
        rate_dropout=0.1,
        len_vec=0,
    ):
        super(TransformerEncoder, self).__init__()
        self.word_embedding = nn.Embedding(num_vocabularies, len_embedding, padding_idx)
        self.positional_encoding = PositionalEncoding(len_embedding, rate_dropout)

        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(len_embedding, num_heads, len_feedforward, activation, rate_dropout, len_vec)
             for _ in range(num_encoder_layers)]
        )

    def forward(self, source_words):
        """
        :param source_words: (batch_size, len_source_sentence)
        :return:
        """
        # -------------------- self attention层的padding mask -------------------------------------------------------- #
        attention_mask_pad = get_attention_mask_pad(source_words)
        # ------------------------------------------------------------------------------------------------------------ #

        # -------------------- word embedding 和 positional encoding ------------------------------------------------- #
        word_embedding = self.word_embedding(source_words)
        x = self.positional_encoding(word_embedding)
        # ------------------------------------------------------------------------------------------------------------ #

        # -------------------- 多层encoder --------------------------------------------------------------------------- #
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, attention_mask_pad)
        # ------------------------------------------------------------------------------------------------------------ #

        return x



class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_vocabularies,
        len_embedding,
        num_encoder_layers,
        num_heads,
        len_feedforward,
        activation,
        padding_idx=1,
        rate_dropout=0.1,
        len_vec=0,
    ):
        super(TransformerDecoder, self).__init__()
        self.word_embedding = nn.Embedding(num_vocabularies, len_embedding, padding_idx)
        self.positional_encoding = PositionalEncoding(len_embedding, rate_dropout)

        self.decoder_layers = nn.ModuleList(
            [TransformerDecoderLayer(len_embedding, num_heads, len_feedforward, activation, rate_dropout, len_vec)
             for _ in range(num_encoder_layers)]
        )

    def forward(self, source_words, target_words, encoder_output):
        """
        :param source_words: (batch_size, len_source_sentence)
        :param target_words: (batch_size, len_target_sentence)
        :param encoder_output: (batch_size, len_source_sentence, len_embedding)
        :return:
        """
        # -------------------- self attention层的mask，padding+subsequence -------------------------------------------- #
        attention_mask_subsequence_pad = get_attention_mask_subsequence_pad(target_words)
        # ------------------------------------------------------------------------------------------------------------ #

        # -------------------- encode-decode 层的 padding mask ------------------------------------------------------- #
        attention_mask_pad = get_attention_mask_pad(source_words, target_words)
        # ------------------------------------------------------------------------------------------------------------ #

        # -------------------- word embedding 和 positional encoding ------------------------------------------------- #
        word_embedding = self.word_embedding(target_words)
        x = self.positional_encoding(word_embedding)
        # ------------------------------------------------------------------------------------------------------------ #

        # -------------------- 多层decoder --------------------------------------------------------------------------- #
        for decode_layer in self.decoder_layers:
            x = decode_layer(x, encoder_output, attention_mask_subsequence_pad, attention_mask_pad)
        # ------------------------------------------------------------------------------------------------------------ #

        return x


class Config:
    def __init__(self, config_yaml_file_path: str = "../resources/configs/config_default.yaml"):
        """
        传入配置的YAML文件路径，配置
        :param config_yaml_file_path:
        """
        with open(config_yaml_file_path, "r", encoding="utf-8") as config_file:
            self.config_content: str = config_file.read()
            self.config: dict = yaml.load(self.config_content, Loader=yaml.FullLoader)

        self.model: ModelConfig = ModelConfig(self.config_value("model", ModelConfig()))


    def config_value(self, key_name: str, default=None) -> str:
        """
        在字典中找对应键的值，如果没有，用默认值
        :param key_name:
        :param default:
        :return:
        """
        if key_name in self.config.keys():
            return self.config[key_name]
        if default is None:
            assert key_name in self.config.keys(), "No value {} in YAML Config file. Please Check!!".format(key_name)

        return default


    def __repr__(self):
        """
        重新修改打印值
        :return:
        """
        return "Config: \n" + self.config_content



class ModelConfig:
    def __init__(self, model_config=None):
        """
        :contrib-- num_source_vocabularies: 源词库的大小
        :contrib-- num_target_vocabularies: 目的词库的大小
        :contrib-- padding_idx: padding对应的序号
        :contrib-- len_embedding: word embedding向量的长度
        :contrib-- num_heads: 多头注意力的头数
        :contrib-- num_encoder_layers: encoder层数
        :contrib-- num_decoder_layers: decoder层数
        :contrib-- len_feedforward: feedforward中间层向量的长度
        :contrib-- len_vec: 多头注意力每个头中 Q/K/V 向量的长度
        :contrib-- activation: 激活层的函数，默认relu，可选["gelu", "relu", "relu6"]
        """
        if model_config is None:
            model_config = {}
        self.config: dict = model_config

        self.num_source_vocabularies: int = self.config_value("num_source_vocabularies", 3000)
        self.num_target_vocabularies: int = self.config_value("num_target_vocabularies", 3000)
        self.padding_idx: int = self.config_value("padding_idx", 0)
        self.len_embedding: int = self.config_value("len_embedding", 512)
        self.num_heads: int = self.config_value("num_heads", 8)
        self.num_encoder_layers: int = self.config_value("num_encoder_layers", 6)
        self.num_decoder_layers: int = self.config_value("num_decoder_layers", 6)
        self.len_feedforward: int = self.config_value("len_feedforward", 2048)
        self.rate_dropout: float = self.config_value("rate_dropout", 0.1)
        self.len_vec: int = self.config_value("len_vec", 0)
        activation_name: str = self.config_value("activation", "relu")
        if activation_name == "relu":
            self.activation = nn.ReLU(True)
        elif activation_name == "gelu":
            self.activation = nn.GELU()
        elif activation_name == "relu6":
            self.activation = nn.ReLU6(True)
        else:
            raise RuntimeError(
                "No activation called \"{}\", optional [\"gelu\", \"relu\", \"relu6\"]".format(activation_name)
            )


    def config_value(self, key_name: str, default=None) -> str:
        """
        在字典中找对应键的值，如果没有，用默认值
        :param key_name:
        :param default:
        :return:
        """
        if key_name in self.config.keys():
            return self.config[key_name]
        if default is None:
            assert key_name in self.config.keys(), \
                "No value \"{}\" in YAML ModelConfig file. Please Check!!".format(key_name)

        return default


    def __repr__(self):
        """
        重新修改打印值
        :return:
        """
        content = yaml.dump(self.config)
        return "Model Config: \n" + content



class Transformer(nn.Module):
    def __init__(self, config: Config):
        super(Transformer, self).__init__()
        self.config = config
        self.model_config = config.model

        self.encoder = TransformerEncoder(
            config.model.num_source_vocabularies,
            config.model.len_embedding,
            config.model.num_encoder_layers,
            config.model.num_heads,
            config.model.len_feedforward,
            config.model.activation,
            config.model.padding_idx,
            config.model.rate_dropout,
            config.model.len_vec
        )

        self.decoder = TransformerDecoder(
            config.model.num_target_vocabularies,
            config.model.len_embedding,
            config.model.num_decoder_layers,
            config.model.num_heads,
            config.model.len_feedforward,
            config.model.activation,
            config.model.padding_idx,
            config.model.rate_dropout,
            config.model.len_vec
        )

        self.projection = nn.Linear(config.model.len_embedding, config.model.num_target_vocabularies)

    def forward(self, source_words, target_words):
        source_outputs = self.encoder(source_words)

        target_outputs = self.decoder(source_words, target_words, source_outputs)

        outputs = self.projection(target_outputs)

        return outputs
