from transformers import AutoTokenizer, AutoModelForCausalLM
import note_seq

tokenizer = AutoTokenizer.from_pretrained("TristanBehrens/js-fakes-4bars")
model = AutoModelForCausalLM.from_pretrained("TristanBehrens/js-fakes-4bars")

input_ids = tokenizer.encode("PIECE_START STYLE=JSFAKES GENRE=JSFAKES TRACK_START INST=48 BAR_START NOTE_ON=61 NOTE_ON=60 NOTE_ON=61", return_tensors="pt")
#print(tokenizer("PIECE_START STYLE=JSFAKES GENRE=JSFAKES TRACK_START INST=48 BAR_START NOTE_ON=61 NOTE_ON=60 NOTE_ON=61"))

generated_ids = model.generate(input_ids, max_length=500)
generated_sequence = tokenizer.decode(generated_ids[0])
print(generated_sequence)

