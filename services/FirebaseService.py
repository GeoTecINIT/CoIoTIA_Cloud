import firebase_admin
from firebase_admin import credentials, firestore, auth
from fastapi import HTTPException
import json

class FirebaseService:
    def __init__(self, cred_path: str = "credentials.json"):
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
        self.db = firestore.client()

    def verify_firebase_token(self, authorization: str) -> str:
        try:
            token = authorization.replace("Bearer ", "")
            decoded = auth.verify_id_token(token)
            return decoded["uid"]
        except auth.ExpiredIdTokenError:
            raise HTTPException(status_code=401, detail="Token expirado")
        except auth.InvalidIdTokenError:
            raise HTTPException(status_code=401, detail="Token inválido")
        except Exception:
            raise HTTPException(status_code=401, detail="Error de autenticación")


    def get_fog(self):
        fog_ref = self.db.collection('fog')
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


    def get_fog_of_regions(self, user: str, domain: str):
        regions_ref = self.db.collection('users').document(user).collection('domains').document(domain).collection('regions')
        docs = regions_ref.stream()

        fog_regions = {}
        for doc in docs:
            polygon = json.loads(doc.to_dict().get('polygon'))
            region_name = polygon.get("properties", {}).get("name", doc.id)
            fog_regions[region_name] = doc.to_dict().get('fog')

        return fog_regions