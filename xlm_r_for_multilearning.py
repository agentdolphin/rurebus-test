from transformers.modeling_xlm_roberta import XLMRobertaModel, XLMRobertaConfig
from transformers.modeling_roberta import RobertaModel
from transformers.modeling_bert import BertPreTrainedModel
from torch import nn
from torch.nn import CrossEntropyLoss
import torch


XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "xlm-roberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-base-pytorch_model.bin",
    "xlm-roberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-large-pytorch_model.bin",
    "xlm-roberta-large-finetuned-conll02-dutch": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta"
                                                 "-large-finetuned-conll02-dutch-pytorch_model.bin",
    "xlm-roberta-large-finetuned-conll02-spanish": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta"
                                                   "-large-finetuned-conll02-spanish-pytorch_model.bin",
    "xlm-roberta-large-finetuned-conll03-english": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta"
                                                   "-large-finetuned-conll03-english-pytorch_model.bin",
    "xlm-roberta-large-finetuned-conll03-german": "https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta"
                                                  "-large-finetuned-conll03-german-pytorch_model.bin",
}

#
# class RobertaClassificationHead(nn.Module):
#     """Head for sentence-level classification tasks."""
#
#     def __init__(self, config, num_labels=2):
#         super().__init__()
#         self.num_labels = num_labels
#         self.dense = nn.Linear(config.hidden_size, config.hidden_size)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.out_proj = nn.Linear(config.hidden_size, self.num_labels)
#
#     def forward(self, features, **kwargs):
#         x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
#         x = self.dropout(x)
#         x = self.dense(x)
#         x = torch.tanh(x)
#         x = self.dropout(x)
#         x = self.out_proj(x)
#         return x


class XLMRMultiLearning(BertPreTrainedModel):
    config_class = XLMRobertaConfig
    pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config, num_tag_labels, num_relation_labels,
                 sent_type_loss_weight=0.0, tag_loss_weight=1.0, relation_loss_weight=0.1):
        super().__init__(config)
        self.sent_type_loss_weight = sent_type_loss_weight
        self.tag_loss_weight = tag_loss_weight
        self.relation_loss_weight = relation_loss_weight
        self.num_tag_labels = num_tag_labels
        self.num_relation_labels = num_relation_labels
        # self.num_class_labels = 2

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.tag_classifier = nn.Linear(config.hidden_size, self.num_tag_labels)
        self.relation_classifier = nn.Linear(config.hidden_size, self.num_relation_labels)
        # self.class_classifier = RobertaClassificationHead(config, num_labels=self.num_class_labels)

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

        loss = 0
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]
        # class_logits = self.class_classifier(sequence_output)
        sequence_output = self.dropout(sequence_output)
        tag_logits = self.tag_classifier(sequence_output)
        relation_logits = self.relation_classifier(sequence_output)

        outputs = (tag_logits,
                   relation_logits) + outputs[2:]  # add hidden states and attention if they are here
        #
        # if sent_type_labels is not None:
        #     loss_fct = CrossEntropyLoss()
        #     loss += self.sent_type_loss_weight * loss_fct(class_logits.view(-1, self.num_class_labels),
        #                                                   sent_type_labels.view(-1))

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
