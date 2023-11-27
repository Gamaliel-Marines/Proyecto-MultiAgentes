from flask import Flask, request, jsonify
from main import *

app = Flask(__name__)
model = FoodCollector(20, 20, 5)

@app.route('/init', methods=['GET'])
def init():
    model = FoodCollector(20, 20, 5)

    return jsonify(
        {
            "agents": model.agent_positions,
            "food": model.known_food_positions,
            "deposit_cell": model.known_deposit_pos,

        }
    )

@app.route('/step', methods=['POST'])
def step():
    model.step()

    return jsonify(
        {
            "agents": model.agent_positions,
            "food": model.known_food_positions,
            "deposit_cell": model.known_deposit_pos,

        }
    )


@app.route('/food', methods=['POST'])
def food():
    model.add_food(model)
    return jsonify(
        {
            "message": "food added"
        }
    )

if __name__ == '__main__':
    app.run(debug=True)