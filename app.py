import streamlit as st
from PIL import Image
from torchvision import datasets
import firebase_admin
from firebase_admin import credentials, firestore
import json
import kosmos
import bart

firebase_table = "kosmos2_miniImageNet"

if not firebase_admin._apps:
    cred = credentials.Certificate('serviceAccount.json')
    firebase_admin.initialize_app(cred)

db = firestore.client()

st.title('KOSMOS2 Model Evaluation')

if st.button('Mini-ImageNet'):
    st.write("Elaboration")
    dataset = datasets.ImageFolder(root='dataset/mini-ImageNet')

    folder_names = dataset.classes
    with open('dataset/mini-ImageNet/imagenet_class_index.json') as f:
        class_index = json.load(f)
    
    classes = []
    for folder in folder_names:
        for key, value in class_index.items():
            if value[0] == folder:
                classes.append(value[1])
    
    all_classes = [class_index[str(i)][1] for i in range(len(class_index))]
    
    bart.init_w_labels(classes)

    options = ", ".join(classes)
    class_prompt = f"<grounding> Given the image provided, please identify and describe the objects it represents. After analyzing the image content, select only one label from the list provided below. This label should best describe the primary object in the image. Please ensure that only one label is chosen, and it must be from the list provided"
    grounding = "Please note that the chosen label must be from the list provided and only one label should be selected. After analyzing the image, the single most probable label that best describes the primary object in the provided image is "
    prompt = f'{class_prompt}: \n\n {options} \n\n {grounding}'

    print(f"Prompt: {prompt}")

    stop_inference = False
    
    if st.button('Stop Inference'):
        stop_inference = True

    last_index = db.collection(firebase_table).document('last_index').get().to_dict().get('index', 0)

    for i in range(last_index, len(dataset)):
        image, label = dataset[i]

        pred_string = kosmos.single_image_classification(
            image,
            prompt,
            max_new_tokens=20,
        )

        parsed_output = bart.preprocess_string_from_model(prompt, pred_string)
        pred_label = bart.classify_string(parsed_output)

        doc_ref = db.collection(firebase_table).document()
        doc_ref.set({
            'True Label': classes[label],
            'Predicted Text': pred_string,
            'Processed Output': parsed_output,
            'Predicted Label': pred_label,
        })

        print(f"{i}: {classes[label]} | {parsed_output} | {pred_label}")
        db.collection(firebase_table).document('last_index').set({'index': i})
