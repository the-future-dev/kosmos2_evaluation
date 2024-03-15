from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
from os import path
from torchvision.transforms import ToTensor

model_id = "microsoft/kosmos-2-patch14-224"
model = AutoModelForVision2Seq.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

def test_inference():
    prompt = "<grounding>The subject "
    image = Image.open('dataset/test.jpeg')

    inputs = processor(text=prompt, images=image, return_tensors="pt")

    generated_ids = model.generate(
        pixel_values=inputs["pixel_values"],
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        image_embeds=None,
        image_embeds_position_mask=inputs["image_embeds_position_mask"],
        use_cache=True,
        max_new_tokens=1,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    processed_text = processor.post_process_generation(generated_text, cleanup_and_extract=False)

    return processed_text


def single_image_classification(image, prompt="<grounding>#1 Detect the main object in the image #2 The image contains a", max_new_tokens=30):
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    
    generated_ids = model.generate(
        pixel_values=inputs["pixel_values"],
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        image_embeds=None,
        image_embeds_position_mask=inputs["image_embeds_position_mask"],
        use_cache=True,
        max_new_tokens=max_new_tokens,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    processed_text = processor.post_process_generation(generated_text, cleanup_and_extract=False)

    return processed_text

def batch_image_classification(images, prompt="<grounding>#1 Detect the main object in the image #2 The image contains a"):
    inputs = processor(text=[prompt]*len(images), images=images, return_tensors="pt", padding=True)
    
    generated_ids = model.generate(
        pixel_values=inputs["pixel_values"],
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        image_embeds=None,
        image_embeds_position_mask=inputs["image_embeds_position_mask"],
        use_cache=True,
        max_new_tokens=30,
    )
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=False)

    processed_texts = [processor.post_process_generation(generated_text, cleanup_and_extract=False) for generated_text in generated_texts]

    return processed_texts

if __name__ == '__main__':
    test_inference()
