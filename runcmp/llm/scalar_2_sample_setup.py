"""
SCALAR 2 Sample SETUP Script: Connecting to AIDP Scalar 2 APIs

This script serves as a primer for transitioning from Scalar 1 to Scalar 2 (AI Gateway) and demonstrates how to configure
and authenticate to Scalar 2 services. Please see scalar_2_sample.py for how to make the actual LLM calls.

This script includes:
1. Environment-specific setup for connecting to AIDP Scalar 2 APIs.
2. Configuration and initialization of authentication mechanisms, such as PingFed and Secure Credentials Vault (SCV).
3. Example prompts, models, and scopes used in Scalar 2 integrations.
4. Comprehensive utilities for handling OS-specific configurations and environment variables.

Key features covered in this script:
- Setting environment variables such as `client_id`, `SCV_KEY`, `SCV_NAMESPACE`, `prompt`, and model-specific details.
- Using secure authentication methods (e.g., SCV secrets or existing tokens) to connect to Scalar 2 via PingFed.
- Specific instructions to configure both Windows and Linux environments for AIDP Scalar 2 integration.
- Information about CA certificate bundle paths and their OS-specific handling.

Before running this script:
- Ensure you have access to the Scalar 2 environment (e.g., ETS Dev, WM Dev, QA environments) and relevant credentials.
- Properly configure environment variables via a `.env` file or system-wide settings.
- Follow additional onboarding steps for Linux (if required) using the instructions provided in this script.

Environment Setup Examples:
TODO: Please follow the README.md in this folder.

@Author: timeng
@Date: 2025-06-02
"""


#####################################################################################################################

import httpx
from aidevplatformaccess.core import PingFedAssertionTokenCredential, AIDevPlatformAuth, PingFedGateway, \
    PingFedAssertionAsyncTokenCredential
from dotenv import load_dotenv
from openai import OpenAI

print("######### Setting up Scalar 2 #########")

############### AIDP Utilities ###############
############### DO NOT CHANGE ANY OF THE CODE IN THE FOLLOWING SECTION. You MUST include it in your code, however. ###############
############### YOU CAN OPTIONALLY REFACTOR THIS INTO A DIFFERENT FILE, e.g. aidp_scalar_utilities.py ###############

from httpx import Client
import sys
if sys.platform != "win32":
    verify = "/etc/pki/ca-trust/certs/internal-ca-chain.crt"
else:
    verify = "//fileshare/pki/internal-ca-chain.crt"

http_client = Client(verify=verify)

import os
import platform
from scvlib import SecureCredentialsVault
from aidevplatformaccess.abstractions import IPingFedGateway, IPingFedTokenManager
from aidevplatformaccess.core import PingFedTokenManager, ScvJsonWebKeyProvider, BasicPingFedTokenManager
from aidevplatformaccess.scv import ScvClient


def get_env_var(env_var_name: str, is_required: bool = False) -> str:
    """
    Retrieves the value of the specified environment variable.

    Args:
        env_var_name (str): The name of the environment variable.
        is_required (bool, optional): Specifies whether the environment variable is required.
            If set to True and the environment variable is not set, a ValueError is raised.
            Defaults to False.

    Returns:
        str: The value of the environment variable.

    Raises:
        ValueError: If the environment variable is required and not set.

    """
    env_var = os.getenv(env_var_name)
    if is_required and (env_var is None or env_var.isspace()):
        raise ValueError(f"Environment variable {env_var_name} is not set.")
    return env_var


def create_pingfed_token_manager_using_scv_secret(pingfed_gateway: IPingFedGateway,
                                                  scv_url: str,
                                                  scv_namespace: str,
                                                  scv_key: str) -> IPingFedTokenManager:
    """
    Creates a PingFedTokenManager using a Secure Credentials Vault (SCV) secret.

    Args:
        pingfed_gateway (IPingFedGateway): The PingFedGateway object.
        scv_url (str): The URL of the Secure Credentials Vault.
        scv_namespace (str): The namespace of the SCV secret.
        scv_key (str): The key of the SCV secret.

    Returns:
        IPingFedTokenManager: The PingFedTokenManager object.

    Raises:
        ValueError: If scv_namespace is empty or whitespace.
        ValueError: If scv_key is empty or whitespace.
    """

    if scv_namespace is None or not scv_namespace.strip():
        raise ValueError("scv_namespace cannot be empty or whitespace")

    if scv_key is None or not scv_key.strip():
        raise ValueError("scv_key cannot be empty or whitespace")

    secure_credentials_vault = SecureCredentialsVault(scv_url) # Replace with this to run on mks: secure_credentials_vault = SecureCredentialsVault(env="genpop", scv_env="prod")
    scv_client = ScvClient(secure_credentials_vault)

    security_key_provider = ScvJsonWebKeyProvider(scv_client, scv_namespace, scv_key)

    return PingFedTokenManager(security_key_provider, pingfed_gateway)


def create_pingfed_token_manager_using_existing_token(access_token: str) -> IPingFedTokenManager:
    """
    Creates a PingFedTokenManager using an existing access token.

    Args:
        access_token (str): The access token to be used for authentication.

    Returns:
        IPingFedTokenManager: An instance of the PingFedTokenManager.

    Raises:
        ValueError: If the access_token is empty or contains only whitespace.
    """
    if access_token is None or not access_token.strip():
        raise ValueError("access_token cannot be empty or whitespace")

    return BasicPingFedTokenManager(access_token, 3600)



def get_os_specific_ca_cert_bundle() -> str:
    """
    Returns the path of the CA certificate bundle specific to the operating system.

    Returns:
        str: The path of the CA certificate bundle.

    Raises:
        ValueError: If the operating system platform is not supported.
    """
    os_platform_case_folded = platform.system().casefold()

    if os_platform_case_folded == "windows".casefold():
        ca_cert_bundle = "\\\\fileshare\\pki\\root-ca.crt"
    elif os_platform_case_folded == "linux".casefold():
        ca_cert_bundle = "/etc/pki/ca-trust/certs/internal-ca-chain.crt"
    else:
        raise ValueError(f"Unsupported OS platform: {os_platform_case_folded}")

    return ca_cert_bundle

ca_cert_bundle = get_os_specific_ca_cert_bundle()
pingfed_http_client = httpx.Client(verify=ca_cert_bundle)
pingfed_http_client_async = httpx.AsyncClient(verify=ca_cert_bundle)

############### END AIDP Utilities ###############




############### AIDP Configuration ###############
############### Modify the below section as appropriate to set up your Scalar 2 client. ###############

# TODO: Set this to true if you are using Azure.
azure = False
api_version = "2024-02-01"

# TODO: Change your model name/embeddings name as appropriate. You don't have to do this in this file - this is only for an example.
completions_model_name = "gpt-4o"
completions_model_name_azure = "gpt-35-turbo"
embeddings_model_name = "text-embedding-ada-002"

######### DO NOT CHANGE THESE! #########
# Wealth Management Dev environment
ai_dev_platform_wm_dev_url = "https://ai-gateway-wm-dev.example.com/openai/v1/"
ai_dev_platform_wm_dev_url_azure = f"https://ai-gateway-wm-dev.example.com/azure/openai/deployments/{completions_model_name_azure}"
ai_dev_platform_wm_dev_scope = "urn:api:ops-dev.00000000-0000-0000-0000-000000000001/.app"

# Wealth Management QA environment
ai_dev_platform_wm_qa_url = "https://ai-gateway-wm-qa.example.com/openai/v1/"
ai_dev_platform_wm_qa_url_azure = f"https://ai-gateway-wm-qa.example.com/azure/openai/deployments/{completions_model_name_azure}"
ai_dev_platform_wm_qa_scope = "urn:api:ops-qa.00000000-0000-0000-0000-000000000002/.app"

# WM Prod environment
ai_dev_platform_wm_prod_url = "https://ai-gateway-wm.example.com/openai/v1"
ai_dev_platform_wm_prod_scope = "urn:api:prod.00000000-0000-0000-0000-000000000003/.app"

# ETS Dev environment
ai_dev_platform_ets_dev_url = "https://ai-gateway-dev.example.com/openai/v1"
ai_dev_platform_ets_dev_url_azure = f"https://ai-gateway-dev.example.com/azure/openai/deployments/{completions_model_name_azure}"
ai_dev_platform_ets_dev_scope = "urn:api:ops-dev.00000000-0000-0000-0000-000000000004/.app"

# ETS QA environment
ai_dev_platform_ets_qa_url = "https://ai-gateway-qa.example.com"
ai_dev_platform_ets_qa_scope = "urn:api:ops-qa.00000000-0000-0000-0000-000000000005/.app"

# ETS Prod environment
ai_dev_platform_ets_prod_url = "https://ai-gateway.example.com/openai/v1"
ai_dev_platform_ets_prod_scope = "urn:api:prod.00000000-0000-0000-0000-000000000006/.app"


# TODO: Change this to your proper pingfed.env file. We've included a sample_secrets.env to show you what to put. Name your secrets as PINGFED_CLIENT_ID_<ENV> and REGISTRATION_ID_<ENV>
# You do not need to have a .env file long term - this is just a sample. Ultimately this will go in your config file or app_config/gitops repo.
from pathlib import Path
parent_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(parent_dir))
secrets_path = parent_dir / 'secrets.env'
# Load the environment variables from the pingfed.env file
print(secrets_path)
load_dotenv(secrets_path)


######### DO NOT CHANGE THESE! #########
# For the reference, see your organization's internal webauth documentation.

client_id_dev = get_env_var("PINGFED_CLIENT_ID_DEV")
registration_id_dev = get_env_var("REGISTRATION_ID_DEV")

client_id_qa = get_env_var("PINGFED_CLIENT_ID_QA")
registration_id_qa = get_env_var("REGISTRATION_ID_QA")

client_id_prod = get_env_var("PINGFED_CLIENT_ID_PROD")
registration_id_prod = get_env_var("REGISTRATION_ID_PROD")

assertion_audience_dev = "https://auth-dev.example.com"
assertion_audience_qa = "https://auth-qa.example.com"
assertion_audience_prod = "https://auth.example.com"

pingfed_url_dev = "https://auth-dev.example.com/as/token.oauth2"
pingfed_url_qa = "https://auth-qa.example.com/as/token.oauth2"
pingfed_url_prod = "https://auth.example.com/as/token.oauth2"

scv_namespace_dev = "AUTH/PINGFEDERATE-OIDC/OPS-DEV"
scv_namespace_qa = "AUTH/PINGFEDERATE-OIDC/OPS-QA"
scv_namespace_prod="AUTH/PINGFEDERATE-OIDC/OPS-PROD"

SCV_KEY_DEV = f"oidc/ops-dev/{registration_id_dev}/credentials"
SCV_KEY_QA = f"oidc/ops-qa/{registration_id_qa}/credentials"
SCV_KEY_PROD = f"oidc/prod/{registration_id_prod}/credentials"

scv_url_prod = "https://credential-vault.example.com"
scv_url = scv_url_prod

# TODO: Change these depending on your environment.

# Dev setup
assertion_audience = assertion_audience_dev
scv_namespace = scv_namespace_dev
pingfed_url = pingfed_url_dev
scv_key = SCV_KEY_DEV
client_id = client_id_dev

ai_dev_platform_url = ai_dev_platform_ets_dev_url  
ai_dev_platform_url_azure = ai_dev_platform_ets_dev_url_azure
ai_dev_platform_scope = ai_dev_platform_ets_dev_scope

# QA setup
# assertion_audience = assertion_audience_qa
# scv_namespace = scv_namespace_qa
# pingfed_url = pingfed_url_qa
# scv_key = SCV_KEY_QA
# client_id = client_id_qa
# ai_dev_platform_url = ai_dev_platform_wm_qa_url
# ai_dev_platform_url_azure = ai_dev_platform_wm_qa_url_azure
# ai_dev_platform_scope = ai_dev_platform_wm_qa_scope

# Prod setup
# assertion_audience = assertion_audience_prod
# scv_namespace = scv_namespace_prod
# scv_key = SCV_KEY_PROD
# client_id = client_id_prod
# ai_dev_platform_url = ai_dev_platform_wm_prod_url
# # ai_dev_platform_url_azure = ai_dev_platform_wm_prod_url_azure
# ai_dev_platform_scope = ai_dev_platform_wm_prod_scope


############### AIDP Token Generation ###############
############### DO NOT CHANGE ANY OF THE CODE IN THE FOLLOWING SECTION. ###############


pingfed_gateway = PingFedGateway(pingfed_url, pingfed_http_client, pingfed_http_client_async)

# Using client credential from SCV
pingfed_token_manager = create_pingfed_token_manager_using_scv_secret(pingfed_gateway,
                                                                      scv_url,
                                                                      scv_namespace,
                                                                      scv_key)

# If you have an existing token, you can use this function instead.
#existing_token = ""
#pingfed_token_manager = create_pingfed_token_manager_using_existing_token(existing_token)


ping_credential = PingFedAssertionTokenCredential(pingfed_token_manager,
                                                  client_id,
                                                  assertion_audience)

ping_credential_async = PingFedAssertionAsyncTokenCredential(pingfed_token_manager,
                                                              client_id,
                                                              assertion_audience)


############### End AIDP Token Generation ###############


############### AIDP HTTPX client: PICK ME ###############
# TODO: Depending on your use case, pick sync or async as appropriate
ai_dev_platform_http_client = httpx.Client(verify=ca_cert_bundle)
ai_dev_platform_async_http_client = httpx.AsyncClient(verify=ca_cert_bundle)

# TODO: Set your appropriate scope depending on which URL you are using
ai_dev_platform_http_client.auth = AIDevPlatformAuth(ai_dev_platform_scope, ping_credential)
ai_dev_platform_async_http_client.auth = AIDevPlatformAuth(ai_dev_platform_scope, token_credential_async=ping_credential_async)
############### End AIDP HTTPX client ###############

# Optional - you can print your token out this way.
print(f"Token: {str(ping_credential.get_token(ai_dev_platform_scope).token)}")

print("######### Scalar 2 setup DONE #########")
