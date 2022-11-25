from abc import abstractmethod
from google.oauth2 import service_account
import pandas as pd
import pandas_gbq
import psycopg2


class DataExtractor:
    def __init__(
            self,
            args
    ):
        self.args = args
        self.conn_type = args.conn_type
        self.database = args.database
        self.server = args.server
        self.user = args.user
        self.password = args.password
        self.port = args.port

        self.ent_extract = args.ent_extract
        self.rel_extract = args.rel_extract
        self.feat_extract = args.feat_extract

        self.conn = None
        self.path = args.path

        self._setup_conn()

    def _setup_conn(self):
        if self.ent_extract or self.rel_extract or self.feat_extract:
            try:
                if self.conn_type == 'SQL':
                    self.conn = self._get_sql_conn()

                elif self.conn_type == 'GBQ':
                    self.conn = self._get_gbq_conn()
            except Exception:
                exit(f"Connection to database: {self.database} could not be established, can\'t continue")
        else:
            print('All extraction arguments set to false, using local parquet files\n')

    def _get_sql_conn(self):
        return psycopg2.connect(
            host=self.server,
            database=self.database,
            user=self.user,
            password=self.password,
            port=self.port,
            connect_timeout=5)

    def _get_gbq_conn(self):
        return service_account.Credentials.from_service_account_file(
            f'{self.path["config_fold"]}mimiciv_gbq.json'
        )

    def extract(self, func, *args):
        df = None
        if self.conn_type == 'SQL':
            df = pd.read_sql_query(
                func(*args),
                self.conn)

        elif self.conn_type == 'GBQ':
            df = pandas_gbq.read_gbq(
                func(*args),
                project_id=self.args.project_id,
                credentials=self.conn)
        else:
            exit("Connection type not supported")

        # save df as parquet file
        if df is not None:
            df.to_parquet(f'{self.path["output_fold"]}{func.__name__}.parquet')
            print(f'Read and wrote query: {func.__name__}')
        else:
            exit(f"No data extracted with function: {func.__name__}")

    @abstractmethod
    def run_queries(self):
        pass

    def __del__(self):
        if self.conn == 'SQL' and self.conn is not None:
            self.conn.close()
