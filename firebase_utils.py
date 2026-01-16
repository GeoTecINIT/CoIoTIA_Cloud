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
    fog_id_name = {}
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
        fog_id_name[doc.id] = fog_data['name']
    return fog_dict, fog_id_name


def get_fog_of_regions(user, domain):
    regions_ref = db.collection('users').document(user).collection('domains').document(domain).collection('regions')
    docs = regions_ref.stream()

    fog_regions = {}
    for doc in docs:
        polygon = json.loads(doc.to_dict().get('polygon'))
        fog_regions[polygon["properties"]["name"]] = doc.to_dict().get('fog')

    return fog_regions