from importlib import resources

import yaml

from alpha_lab.agents.agent_definition import AgentDefinition
from alpha_lab.agents import registry
from alpha_lab.config import split_frontmatter_from_config_body

AGENTS_DIR = resources.files(registry)


def load_agent(agent_id: str) -> AgentDefinition:
    """
    Load an agent definition from a YAML markdown file.

    Reads an agent configuration file from the AGENTS_DIR directory, parses its
    YAML frontmatter and markdown body, and returns an AgentDefinition object.

    Args:
        agent_id (str): The identifier of the agent to load. Corresponds to the filename
                        (without extension) in the agents directory.

    Returns:
        AgentDefinition: An AgentDefinition object containing the agent's configuration,
                         including name, description, tools, web search settings, reasoning
                         effort, logging configuration, and prompt information.

    Raises:
        FileNotFoundError: If the agent markdown file does not exist.
        KeyError: If required fields are missing from the YAML frontmatter
                  (name, description, allowed-tools at top level; prompt_source, log_name under metadata).
        yaml.YAMLError: If the frontmatter is not valid YAML.

    Notes:
        - The agent file must be in markdown format with YAML frontmatter.
        - Leading newlines are stripped from the prompt body.
        - The prompt_body is only populated when prompt_source is "inline".
        - Optional metadata fields default to: include_web_search=False, min_report_attempts=2,
          reasoning_effort=None.
    """
    text = (AGENTS_DIR / f"{agent_id}.md").read_text()
    frontmatter, body = split_frontmatter_from_config_body(text)
    frontmatter = yaml.safe_load(frontmatter)
    return AgentDefinition.build_from_config(frontmatter, body)


__all__ = ["AgentDefinition", "AGENTS_DIR", "load_agent"]
