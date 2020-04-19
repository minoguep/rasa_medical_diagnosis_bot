from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from diagnosis_functions import encode_symptom, create_illness_vector, get_diagnosis


class ActionDiagnoseSymptoms(Action):

    def name(self) -> Text:
        return "action_diagnose_symptoms"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        symptoms = tracker.get_slot("symptom")

        # encode each symptom
        encoded_symptoms = [encode_symptom(symptom) for symptom in symptoms]

        # create a binary vector of symptoms to compare to each each documented illnedd
        illness_vector = create_illness_vector(encoded_symptoms)

        # perform diagnosis
        diagnosis_string = get_diagnosis(illness_vector)

        dispatcher.utter_message(text=diagnosis_string)
