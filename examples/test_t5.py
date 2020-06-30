from transformers import T5Tokenizer as Tokenizer
from transformers import T5ForConditionalGeneration as Model

arch = 't5-base'

tokenizer = Tokenizer.from_pretrained(arch)
model = Model.from_pretrained(arch)

text = ["my dog is", "my cat is"]
inputs = tokenizer.batch_encode_plus(
    text, return_attention_masks=True, return_tensors='pt')
outputs = model.generate(inputs['input_ids'], num_beams=1)
for one in outputs:
    print(tokenizer.decode(one))

# config = T5Config.from_pretrained(
#     't5-small', eos_token_ids=tokenizer.eos_token_id)

# model = T5LMForVariants.from_pretrained('t5-small', config=config)
# inputs.pop('token_type_ids', None)
# print(inputs)
# candidates = inputs['input_ids'].new_tensor(
#     [[12, 13, 14, 1, 0, 0, 0, 0, 0, 0],
#      [56, 57, 58, 59, 60, 61, 62, 63, 64, 1]])
# outputs = model(inputs['input_ids'], candidates=candidates)
