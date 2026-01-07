import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

import json

cred = credentials.Certificate("credentials.json")

firebase_admin.initialize_app(cred)

db = firestore.client()


def get_fog():
    fog_ref = db.collection('fog')
    docs = fog_ref.stream()

    fog_dict = {}
    for doc in docs:
        fog_data = doc.to_dict()
        fog_dict[fog_data['name']] = {
            "id" : doc.id,
            "ip" : fog_data['ip'],
            "status" : 0,
            "last_updated": None,
            "cpu": None,
            "ram": None,
            "disk": None
        }
    return fog_dict