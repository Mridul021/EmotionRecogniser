from flask import Flask, render_template, request, send_from_directory
def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__)

    # Simple route
    @app.route('/')
    def hello_world():
        return jsonify({
            "status": "success",
            "message": "Hello World!"
        })

    return app  # do not forget to return the app


APP = create_app()

if __name__ == '__main__':
    APP.run(host='0.0.0.0', port=5000, debug=True)
    APP.run(debug=True)