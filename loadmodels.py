from transformers import ViltProcessor,ViltForQuestionAnswering
from bark import SAMPLE_RATE, generate_audio, preload_models
from transformers import AutoProcessor, AutoModel
from main import st

@st.cache_data
def model_functions():
    vit_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    vit_model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

    audio_processor = AutoProcessor.from_pretrained("suno/bark-small")
    audio_model = AutoModel.from_pretrained("suno/bark-small")
    return vit_model,vit_processor,audio_model,audio_processor

vit_model,vit_processor,audio_model,audio_processor = model_functions()