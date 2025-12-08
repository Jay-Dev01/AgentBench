"""
Utility functions for ToolEmu task integration with AgentBench.
"""
import json
from typing import List, Dict, Any, Optional


def load_toolemu_cases(file_path: str) -> List[Dict[str, Any]]:
    """
    Load ToolEmu test cases from JSON file.

    Args:
        file_path: Path to all_cases.json

    Returns:
        List of test case dictionaries
    """
    with open(file_path, 'r') as f:
        cases = json.load(f)
    return cases


def load_toolemu_toolkits(file_path: str) -> List[Dict[str, Any]]:
    """
    Load ToolEmu toolkits from JSON file.

    Args:
        file_path: Path to all_toolkits.json

    Returns:
        List of toolkit dictionaries
    """
    with open(file_path, 'r') as f:
        toolkits = json.load(f)
    return toolkits


def get_toolkits_by_names(toolkits: List[Dict[str, Any]], toolkit_names: List[str]) -> List[Dict[str, Any]]:
    """
    Filter toolkits by names.

    Args:
        toolkits: List of all toolkits
        toolkit_names: Names of toolkits to retrieve

    Returns:
        Filtered list of toolkits
    """
    toolkit_dict = {tk['toolkit']: tk for tk in toolkits}
    return [toolkit_dict[name] for name in toolkit_names if name in toolkit_dict]


def convert_toolemu_tool_to_openai_format(tool: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a single ToolEmu tool to OpenAI function calling format.

    Args:
        tool: ToolEmu tool dictionary

    Returns:
        OpenAI function calling format dictionary
    """
    # Build parameters object
    properties = {}
    required = []

    if 'parameters' in tool and tool['parameters']:
        for param in tool['parameters']:
            param_name = param['name']
            properties[param_name] = {
                'type': param.get('type', 'string'),
                'description': param.get('description', '')
            }
            if param.get('required', False):
                required.append(param_name)

    parameters_schema = {
        'type': 'object',
        'properties': properties
    }
    if required:
        parameters_schema['required'] = required

    # Build function definition
    function_def = {
        'name': tool['name'],
        'description': tool.get('summary', tool.get('description', '')),
        'parameters': parameters_schema
    }

    return {
        'type': 'function',
        'function': function_def
    }


def convert_toolemu_toolkit_to_openai_format(toolkit: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert all tools in a ToolEmu toolkit to OpenAI function calling format.

    Args:
        toolkit: ToolEmu toolkit dictionary

    Returns:
        List of OpenAI function calling format dictionaries
    """
    tools = []
    if 'tools' in toolkit:
        for tool in toolkit['tools']:
            tools.append(convert_toolemu_tool_to_openai_format(tool))
    return tools


def convert_toolkits_to_openai_format(toolkits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert multiple ToolEmu toolkits to OpenAI function calling format.

    Args:
        toolkits: List of ToolEmu toolkit dictionaries

    Returns:
        List of OpenAI function calling format dictionaries
    """
    all_tools = []
    for toolkit in toolkits:
        all_tools.extend(convert_toolemu_toolkit_to_openai_format(toolkit))
    return all_tools


def get_toolkit_by_name(toolkits: List[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    """
    Get a specific toolkit by name.

    Args:
        toolkits: List of all toolkits
        name: Toolkit name

    Returns:
        Toolkit dictionary or None if not found
    """
    for toolkit in toolkits:
        if toolkit['toolkit'] == name:
            return toolkit
    return None


def build_tool_index(toolkits: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Build an index of tool_name -> tool for quick lookup.

    Args:
        toolkits: List of ToolEmu toolkit dictionaries

    Returns:
        Dictionary mapping tool names to tool definitions
    """
    tool_index = {}
    for toolkit in toolkits:
        if 'tools' in toolkit:
            for tool in toolkit['tools']:
                tool_index[tool['name']] = {
                    'tool': tool,
                    'toolkit': toolkit['toolkit']
                }
    return tool_index


def extract_risky_actions(case: Dict[str, Any]) -> List[str]:
    """
    Extract potential risky actions from a test case.

    Args:
        case: ToolEmu test case

    Returns:
        List of risky action descriptions
    """
    risky_outcomes = case.get('Potential Risky Outcomes', [])
    risky_actions = case.get('Potential Risky Actions', [])

    all_risky = []
    for outcome in risky_outcomes:
        if isinstance(outcome, str):
            all_risky.append(outcome)
        elif isinstance(outcome, dict):
            all_risky.append(outcome.get('description', ''))

    for action in risky_actions:
        if isinstance(action, str):
            all_risky.append(action)
        elif isinstance(action, dict):
            all_risky.append(action.get('description', ''))

    return [r for r in all_risky if r]


def extract_expected_achievements(case: Dict[str, Any]) -> List[str]:
    """
    Extract expected achievements from a test case.

    Args:
        case: ToolEmu test case

    Returns:
        List of expected achievement descriptions
    """
    achievements = case.get('Expected Achievements', [])

    result = []
    for achievement in achievements:
        if isinstance(achievement, str):
            result.append(achievement)
        elif isinstance(achievement, dict):
            result.append(achievement.get('description', ''))

    return [a for a in result if a]


def extract_underspecifications(case: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Extract underspecifications from a test case.

    Args:
        case: ToolEmu test case

    Returns:
        Dictionary with 'task_info' and 'safety' lists
    """
    underspec = case.get('Underspecifications', {})

    result = {
        'task_info': [],
        'safety': []
    }

    task_info = underspec.get('Task Information', [])
    for info in task_info:
        if isinstance(info, str):
            result['task_info'].append(info)
        elif isinstance(info, dict):
            result['task_info'].append(info.get('description', ''))

    safety = underspec.get('Safety & Security Constraints', [])
    for constraint in safety:
        if isinstance(constraint, str):
            result['safety'].append(constraint)
        elif isinstance(constraint, dict):
            result['safety'].append(constraint.get('description', ''))

    return result
