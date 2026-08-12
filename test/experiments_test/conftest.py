import pytest
from sqlalchemy.orm import sessionmaker

from krrood.ormatic.utils import create_engine, drop_database


@pytest.fixture(scope="function")
def experiments_testing_session():
    """
    An in-memory database session against ``experiments.orm.ormatic_interface``'s
    ``Base``, for tests that persist domain objects (e.g. Montessori sorting results)
    via ``to_dao`` and query them back.
    """
    import experiments.orm.ormatic_interface as ormatic_interface

    engine = create_engine("sqlite:///:memory:")
    session = sessionmaker(engine)()
    ormatic_interface.Base.metadata.create_all(bind=session.bind)
    yield session
    drop_database(session.bind)
    session.close()
    engine.dispose()
