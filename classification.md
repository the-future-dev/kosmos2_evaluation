# kosmos_01
### idea: direct the LLM to classify
classes = dataset.classes
classes.insert(0,'undefined')

results = []

options = "\n".join(classes)
class_prompt = f"Given the image provided, please identify the object it represents. Choose only one word from the following options that best describes the object in the image: {options}"
grounding = "<grounding> The word that best describes the object in the image is"
prompt = f"{class_prompt}\n{grounding}"

<!-- # describe_then_derive
### idea: let the LLM free to understand the image and provide a great text explaination then use a text classifier to detect the class 
grounding = "Given the image provided, please identify the objects it contains. Then describe in detail the image content. <grounding>The image of a " -->

# kosmos_02
1. ask the model to internally formulate a description of the image as it's a task it has been specifically trained for
2. perform zero-shot classification

options = "\n".join(classes)
class_prompt = f"<grounding>Given the image provided, please identify the objects it represents, then internally describe the image content in a simple phrase. Finally choose only one word from the following options that best describes the object in the image: {options}"
grounding = "Constraining the otput dictionary to the previously provided options, the most probable single word to describe the image main object is:"
prompt = f'{class_prompt} {grounding}'
