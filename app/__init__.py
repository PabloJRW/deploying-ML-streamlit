from flask import Flask


def create_app():
    app= Flask(__name__)
    app.config.from_mapping(
        DEBUG = True,
        SECRETE_KEY = 'dev',
        SQLALCHEMY_DATABASE_URI = "sqlite:///inventario.db"
    )


    from app.main import main_bp


    app.register_blueprint(main_bp)


    return app  