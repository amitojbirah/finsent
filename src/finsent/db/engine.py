from sqlalchemy import create_engine

from finsent.config import get_database_url

engine = create_engine(get_database_url())
