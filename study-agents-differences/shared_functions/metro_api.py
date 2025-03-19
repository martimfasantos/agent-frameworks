import httpx
import json
import inspect
from typing import List, Union

from shared_functions.base_module import BaseModule

# Private API - needs a token
# Get one at https://api.metrolisboa.pt/store/apis/info?name=EstadoServicoML&version=1.0.1&provider=admin
# By default, token will exist ONLY for 3600 seconds!!!

class MetroAPI(BaseModule):

    token = "9e94ac40-21c6-31f2-8ef0-7277e4a2ab9f"

    @staticmethod
    def get_state_subway() -> str:
        """
        Useful function to get the information about the state of the subway.
        """
        url = "https://api.metrolisboa.pt:8243/estadoServicoML/1.0.1/estadoLinha/todos"
        headers: dict = {
            "Accept": "application/json",
            "Authorization": f"Bearer {MetroAPI.token}"
        }
        response = httpx.get(url, headers=headers, verify=False)

        if response.status_code == 200:
            return json.dumps(response.json())
        else:
            return f"Failed to get state subway information: {response.status_code}"
        
    @staticmethod
    def get_times_next_two_subways_in_station(station: str) -> Union[List[int], str]:
        """
        Useful to get the time (in seconds) of the next two subways in a station.
        """
        url = f"https://api.metrolisboa.pt:8243/estadoServicoML/1.0.1/tempoEspera/Estacao/{station}"
        headers: dict = {
            "Accept": "application/json",
            "Authorization": f"Bearer {MetroAPI.token}"
        }
        response = httpx.get(url, headers=headers, verify=False)

        if response.status_code == 200:
            data = response.json()
            tempos_chegada = [int(item["tempoChegada1"]) for item in data["resposta"]]
            lowest_tempos = sorted(tempos_chegada)[:2]
            return json.dumps({"times": lowest_tempos, "metric": "seconds"})
        else:
            return f"Failed to get time for the next two subways in station: {response.status_code}"
