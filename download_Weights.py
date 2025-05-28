from transformers import PaliGemmaForConditionalGeneration, AutoProcessor

model = PaliGemmaForConditionalGeneration.from_pretrained("google/paligemma-3b-pt-224")
processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224")


model.save_pretrained("./local_paligemma")
processor.save_pretrained("./local_paligemma")