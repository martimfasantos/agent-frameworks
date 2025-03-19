import httpx
import json
import inspect

from shared_functions.base_module import BaseModule

# Public API

class F1API(BaseModule):

    @staticmethod
    def get_driver_info(driver_number: int, session_key: int = 9158) -> str:
        """
        Useful function to get F1 driver information.
        """
        url = f"https://api.openf1.org/v1/drivers?driver_number={driver_number}&session_key={session_key}"
        response = httpx.get(url)
        
        if response.status_code == 200:
            return json.dumps(response.json())
        else:
            return f"Failed to get driver information: {response.status_code}"
    
