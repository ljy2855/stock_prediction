import yaml
import os
from dotenv import load_dotenv


class ConfigLoader:
    """
    설정 파일을 불러와 경로를 관리하는 클래스
    """
    def __init__(self, config_path="config/paths.yaml", secret_path=".env"):
        self.config_path = config_path
        self.secret_path = secret_path
        self.config = self._load_config()
        self.secret = self._load_secret()

    def _load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        with open(self.config_path, "r") as file:
            return yaml.safe_load(file)
    
    def _load_secret(self):
        if not os.path.exists(self.secret_path):
            raise FileNotFoundError(f"Secret file not found: {self.secret_path}")
        # Load environment variables from .env file
        load_dotenv(self.secret_path)

    def get_path(self, *keys):
        """
        YAML 설정 파일에서 경로를 반환
        :param keys: YAML 키 순서대로 입력 (예: 'paths', 'raw_data', 'stock_data')
        :return: 해당 경로
        """
        value = self.config
        for key in keys:
            value = value.get(key, {})
            if not value:
                raise KeyError(f"Key {key} not found in config file.")
        return value
    
    def get_secret(self, secret_name):
        """
        .env 파일에서 비밀 값을 반환
        :param secret_name: 환경 변수 이름
        :return: 해당 환경 변수 값
        """
        value = os.getenv(secret_name)
        if value is None:
            raise KeyError(f"Secret {secret_name} not found in environment variables.")
        return value

config = ConfigLoader()