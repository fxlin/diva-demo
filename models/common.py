from sqlalchemy import create_engine, MetaData
from sqlalchemy.engine.url import URL
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from variables import DEFAULT_POSTGRES_PASSWORD, DEFAULT_POSTGRES_USER, DEFAULT_POSTGRES_DB
from variables import DEFAULT_POSTGRES_HOST, DEFAULT_POSTGRES_PORT

# engine = create_engine('postgresql://dbuser:dbpassword@localhost:5432/sqlalchemy-orm-tutorial')
# 'sqlite:///sqlalchemy_example.db'
new_url = URL(drivername="postgres",
              username=DEFAULT_POSTGRES_USER,
              password=DEFAULT_POSTGRES_PASSWORD,
              host=DEFAULT_POSTGRES_HOST,
              port=DEFAULT_POSTGRES_PORT,
              database=DEFAULT_POSTGRES_DB)
engine = create_engine(new_url)
db_session = scoped_session(
    sessionmaker(autocommit=False, autoflush=False, bind=engine))

Base = declarative_base()
Base.query = db_session.query_property()


def init_db():
    import models.camera
    import models.video
    import models.frame
    import models.element
    Base.metadata.create_all(bind=engine)


def drop_all_tables(engineURL):
    new_engine = create_engine(engineURL)
    meta = MetaData(new_engine)
    meta.reflect()
    meta.drop_all()
