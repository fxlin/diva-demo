from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, scoped_session
from variables import DEFAULT_POSTGRES_PASSWORD, DEFAULT_POSTGRES_USER, DEFAULT_POSTGRES_DB
from variables import DEFAULT_POSTGRES_HOST, DEFAULT_POSTGRES_PORT

# engine = create_engine('postgresql://dbuser:dbpassword@localhost:5432/sqlalchemy-orm-tutorial')
# FIXME use real db
# 'sqlite:///sqlalchemy_example.db'
new_url = URL(drivername="postgres",
              username=DEFAULT_POSTGRES_USER,
              password=DEFAULT_POSTGRES_PASSWORD,
              host=DEFAULT_POSTGRES_HOST,
              port=DEFAULT_POSTGRES_PORT,
              database=DEFAULT_POSTGRES_DB)
engine = create_engine(new_url)
Base = declarative_base()


def session_factory() -> Session:
    Base.metadata.create_all(engine)
    # use session_factory() to get a new Session
    return scoped_session(
        sessionmaker(bind=engine, autocommit=False, autoflush=False))
