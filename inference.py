from fairseq.models.transformer import TransformerModel
import transformers

model = TransformerModel.from_pretrained("./wmt19.ru-en.single_model", "model.pt")

result = model.translate("Я - это ты.")

print(result)

