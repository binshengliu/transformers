from transformers import T5ForSequenceClassification, T5Tokenizer, T5Config
import torch

tokenizer = T5Tokenizer.from_pretrained('t5-small')
config = T5Config.from_pretrained('t5-small', num_labels=2)
model = T5ForSequenceClassification.from_pretrained('t5-small', config=config)

text = ["Hello, my dog is cute", "Second test sentence"]
inputs = tokenizer.batch_encode_plus(text, return_attention_masks=True, return_tensors='pt')
inputs.pop('token_type_ids', None)

labels = torch.tensor([1, 0], dtype=torch.long)
# train
loss, logits = model(**inputs, labels=labels)
# predict
logits = model(**inputs)[0]
preds = logits.argmax(dim=-1)
for pred, input in zip(preds, text):
    print('{} {}'.format(pred, input))
