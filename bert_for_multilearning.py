from transformers import BertPreTrainedModel, BertModel
from torch import nn
from torch.nn import CrossEntropyLoss
# from transformers.file_utils import add_start_docstrings


BERT_START_DOCSTRING = r"""
    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Parameters:
        config (:class:`~transformers.BertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

BERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`transformers.BertTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.encode_plus` for details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`, defaults to :obj:`None`):
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            :obj:`1` indicates the head is **not masked**, :obj:`0` indicates the head is **masked**.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
"""

# @add_start_docstrings(
#     """Bert Model with a token classification head on top (a linear layer on top of
#     the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. """,
#     BERT_START_DOCSTRING,
# )
class BertForMultitaskLearning1(BertPreTrainedModel):
    def __init__(self, config, num_tag_labels, num_relation_labels,
                 sent_type_loss_weight=0.0, tag_loss_weight=1.0, relation_loss_weight=0.1):
        super().__init__(config)
        self.sent_type_loss_weight = sent_type_loss_weight
        self.tag_loss_weight = tag_loss_weight
        self.relation_loss_weight = relation_loss_weight
        self.num_tag_labels = num_tag_labels
        self.num_relation_labels = num_relation_labels
        self.num_class_labels = 2

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.tag_classifier = nn.Linear(config.hidden_size, self.num_tag_labels)
        self.relation_classifier = nn.Linear(config.hidden_size, self.num_relation_labels)
        self.class_classifier = nn.Linear(config.hidden_size, self.num_class_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        sent_type_labels=None,
        tag_labels=None,
        relation_labels=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :
            Classification loss.
        scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`)
            Classification scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import BertTokenizer, BertForTokenClassification
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForTokenClassification.from_pretrained('bert-base-uncased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)

        loss, scores = outputs[:2]

        """

        loss = 0
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output, pooled_output = outputs[0], outputs[1]

        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)

        tag_logits = self.tag_classifier(sequence_output)
        relation_logits = self.relation_classifier(sequence_output)
        class_logits = self.class_classifier(pooled_output)

        outputs = (class_logits,
                   tag_logits,
                   relation_logits) + outputs[2:]  # add hidden states and attention if they are here

        if sent_type_labels is not None:
            loss_fct = CrossEntropyLoss()
            loss += self.sent_type_loss_weight * loss_fct(class_logits.view(-1, self.num_class_labels), sent_type_labels.view(-1))

        active_loss = attention_mask.view(-1) == 1
        if tag_labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            active_labels = tag_labels.view(-1)[active_loss]
            active_logits = tag_logits.view(-1, self.num_tag_labels)[active_loss]
            loss += self.tag_loss_weight * loss_fct(active_logits, active_labels)

        if relation_labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            active_labels = relation_labels.view(-1)[active_loss]
            active_logits = relation_logits.view(-1, self.num_relation_labels)[active_loss]
            loss += self.relation_loss_weight * loss_fct(active_logits, active_labels)

        if loss > 0:
            outputs = loss

        return outputs  # (loss), scores, (hidden_states), (attentions)


# @add_start_docstrings(
#     """Bert Model with a token classification head on top (a linear layer on top of
#     the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. """,
#     BERT_START_DOCSTRING,
# )
class BertForMultitaskLearning2(BertPreTrainedModel):
    def __init__(self, config, num_tag_labels, num_relation_labels,
                 sent_type_loss_weight=1, tag_loss_weight=1, relation_loss_weight=1):
        super().__init__(config)
        self.sent_type_loss_weight = sent_type_loss_weight
        self.tag_loss_weight = tag_loss_weight
        self.relation_loss_weight = relation_loss_weight
        self.num_tag_labels = num_tag_labels
        self.num_relation_labels = num_relation_labels
        self.num_class_labels = 2

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.tag_classifier = nn.Linear(config.hidden_size, config.num_tag_labels)
        self.relation_classifier = nn.Linear(config.hidden_size, config.num_relation_labels)
        self.class_classifier = nn.Linear(config.hidden_size, self.num_class_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        sent_type_labels=None,
        tag_labels=None,
        relation_labels=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :
            Classification loss.
        scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`)
            Classification scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import BertTokenizer, BertForTokenClassification
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForTokenClassification.from_pretrained('bert-base-uncased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)

        loss, scores = outputs[:2]

        """

        loss = 0
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output, pooled_output = outputs[0], outputs[1]

        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)

        tag_logits = self.tag_classifier(sequence_output)
        relation_logits = self.relation_classifier(pooled_output)
        class_logits = self.class_classifier(pooled_output)

        outputs = (class_logits,
                   tag_logits,
                   relation_logits) + outputs[2:]  # add hidden states and attention if they are here

        if sent_type_labels is not None:
            loss_fct = CrossEntropyLoss()
            loss += self.sent_type_loss_weight * loss_fct(class_logits.view(-1, self.num_class_labels),
                                                      sent_type_labels.view(-1))

        active_loss = attention_mask.view(-1) == 1
        if tag_labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            active_labels = tag_labels.view(-1)[active_loss]
            active_logits = tag_logits.view(-1, self.num_tag_labels)[active_loss]
            loss += self.tag_loss_weight * loss_fct(active_logits, active_labels)

        if relation_labels is not None:
            loss_fct = CrossEntropyLoss()
            loss += self.relation_loss_weight * loss_fct(relation_logits.view(-1, self.num_relation_labels), relation_labels.view(-1))

        if loss > 0:
            outputs = (loss,) + outputs
        return outputs  # (loss), scores, (hidden_states), (attentions)