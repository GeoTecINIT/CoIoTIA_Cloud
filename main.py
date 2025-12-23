from waitress import serve
from cloud import app, connect_mqtt

connect_mqtt()

if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=8080)