import requests
import json
import numpy as np

def emotion_detector(text_to_analyse):
    url = 'https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'
    header = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}
    myobj = { "raw_document": { "text": text_to_analyse } }
    response = requests.post(url, json = myobj, headers = header)
    
    formatted_response = json.loads(response.text)

    if response.status_code == 400:
        output = {
            'anger': None,
            'disgust': None,
            'fear': None,
            'joy': None,
            'sadness': None,
            'dominant_emotion' : None
        }
        return output
    else:
        anger_score = formatted_response['emotionPredictions'][0]['emotion']['anger']
        disgust_score = formatted_response['emotionPredictions'][0]['emotion']['disgust']
        fear_score = formatted_response['emotionPredictions'][0]['emotion']['fear']
        joy_score = formatted_response['emotionPredictions'][0]['emotion']['joy']
        sadness_score = formatted_response['emotionPredictions'][0]['emotion']['sadness']
        
        scores = np.array([anger_score, disgust_score, fear_score, joy_score, sadness_score])
        emotions = ['anger', 'disgust', 'fear', 'joy', 'sadness']
        output = {
            'anger': anger_score,
            'disgust': disgust_score,
            'fear': fear_score,
            'joy': joy_score,
            'sadness': sadness_score,
            'dominant_emotion' : emotions[np.argmax(scores)]
        }
        return output