from loadmodels import audio_model,audio_processor
from VilT import  answer
from IPython.display import Audio
import scipy

inputs = audio_processor(
    text= answer,
    return_tensors="pt",
)
sampling_rate = audio_model.generation_config.sample_rate

speech_values = audio_model.generate(**inputs, do_sample=True)
Audio(speech_values.cpu().numpy().squeeze(), rate=sampling_rate)

scipy.io.wavfile.write("bark_out.wav", rate=sampling_rate, data=speech_values.cpu().numpy().squeeze())

