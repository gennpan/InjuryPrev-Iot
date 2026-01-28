from flask import Flask, jsonify, render_template, request
import pandas as pd
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Feature di interesse che mostro nella dash
FEATURES = [
    "date",
    "speed_mean", "speed_max", "speed_std",
    "acc_norm_mean", "acc_norm_max", "acc_norm_std",
    "gyro_norm_mean", "gyro_norm_max",
    "n_samples"
]

PLAYER_CSV_FILES = {
    "1": [
        "2f23d7d5-2326-49ce-b9c8-5a6303f785c5.csv",
        "8d723104-f773-83c1-3458-a748e9bb17bc.csv"
    ]
}


def load_player_data(player_id, csv_file, limit=10000):
    player_dir = os.path.join(DATA_DIR, str(player_id))
    csv_path = os.path.join(player_dir, csv_file)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV non trovato: {csv_file} per player_id={player_id}")

    df = pd.read_csv(csv_path)

    return df

@app.route("/")
def statistics_page():
    return render_template("statistics.html")

@app.route("/api/player/<player_id>/files")
def api_player_files(player_id):
    files = PLAYER_CSV_FILES.get(player_id, [])

    available_files = [f for f in files if os.path.exists(os.path.join(DATA_DIR, player_id, f))]
    return jsonify(available_files)


@app.route("/api/player/<player_id>")
def api_player_features(player_id):
    csv_file = request.args.get("file")
    if not csv_file:
        return jsonify({"error": "Devi specificare il parametro 'file'"}), 400

    try:
        df = load_player_data(player_id, csv_file, limit=1000)
        return jsonify(df.to_dict(orient="records"))
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404


if __name__ == "__main__":
    app.run(debug=True)