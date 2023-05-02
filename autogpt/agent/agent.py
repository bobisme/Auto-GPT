
from dataclasses import dataclass
from typing import Any, Tuple
from colorama import Fore, Style
from autogpt.app import execute_command, get_command
from autogpt.config import Config
from autogpt.json_utils.json_fix_llm import fix_json_using_multiple_techniques
from autogpt.json_utils.utilities import LLM_DEFAULT_RESPONSE_FORMAT, validate_json
from autogpt.llm import chat_with_ai, create_chat_completion, create_chat_message
from autogpt.llm.token_counter import count_string_tokens
from autogpt.logs import logger, print_assistant_thoughts
from autogpt.speech import say_text
from autogpt.spinner import Spinner
from autogpt.utils import clean_input
from autogpt.workspace import Workspace


@dataclass
class Command:
    user_input: str = ""
    name: str | None = None
    args: dict = {}

GENERATE_NEXT_COMMAND_JSON = Command(user_input="GENERATE_NEXT_COMMAND_JSON")
EXIT = Command(user_input="EXIT")

class Agent:
    """Agent class for interacting with Auto-GPT.

    Attributes:
        ai_name: The name of the agent.
        memory: The memory object to use.
        full_message_history: The full message history.
        next_action_count: The number of actions to execute.
        system_prompt: The system prompt is the initial prompt that defines everything
          the AI needs to know to achieve its task successfully.
        Currently, the dynamic and customizable information in the system prompt are
          ai_name, description and goals.

        triggering_prompt: The last sentence the AI will see before answering.
            For Auto-GPT, this prompt is:
            Determine which next command to use, and respond using the format specified
              above:
            The triggering prompt is not part of the system prompt because between the
              system prompt and the triggering
            prompt we have contextual information that can distract the AI and make it
              forget that its goal is to find the next task to achieve.
            SYSTEM PROMPT
            CONTEXTUAL INFORMATION (memory, previous conversations, anything relevant)
            TRIGGERING PROMPT

        The triggering prompt reminds the AI about its short term meta task
        (defining the next task)
    """

    def __init__(
        self,
        ai_name,
        memory,
        full_message_history,
        next_action_count,
        command_registry,
        config: Config | None,
        system_prompt,
        triggering_prompt,
        workspace_directory,
    ):
        self.ai_name = ai_name
        self.memory = memory
        self.summary_memory = (
            "I was created."  # Initial memory necessary to avoid hilucination
        )
        self.last_memory_index = 0
        self.full_message_history = full_message_history
        self.next_action_count = next_action_count
        self.command_registry = command_registry
        self.config = config
        if self.config is None:
            self.config = Config()
        self.system_prompt = system_prompt
        self.triggering_prompt = triggering_prompt
        self.workspace = Workspace(workspace_directory, self.config.restrict_to_workspace)

    def _prompt_user(self):
        if self.config.chat_messages_enabled:
            return clean_input("Waiting for your response...")
        return clean_input(f"{Fore.MAGENTA}Input: {Style.RESET_ALL}")

    def _get_user_input(self, assistant_reply: dict) -> Command | None:
        console_input = self._prompt_user
        if console_input.lower().strip() == self.config.authorise_key:
            return GENERATE_NEXT_COMMAND_JSON

        if console_input.lower().strip() == "s":
            logger.typewriter_log(
                "-=-=-=-=-=-=-= THOUGHTS, REASONING, PLAN AND CRITICISM WILL NOW BE VERIFIED BY AGENT -=-=-=-=-=-=-=",
                Fore.GREEN,
                "",
            )
            thoughts = assistant_reply.get("thoughts", {})
            self_feedback_resp = self.get_self_feedback(
                thoughts, self.config.fast_llm_model
            )
            logger.typewriter_log(
                f"SELF FEEDBACK: {self_feedback_resp}",
                Fore.YELLOW,
                "",
            )
            if self_feedback_resp[0].lower().strip() == self.config.authorise_key:
                return GENERATE_NEXT_COMMAND_JSON
            return Command(user_input=self_feedback_resp)

        if console_input.lower().strip() == "":
            logger.warn("Invalid input format.")
            return

        if console_input.lower().startswith(f"{self.config.authorise_key} -"):
            try:
                self.next_action_count = abs(
                    int(console_input.split(" ")[1])
                )
            except ValueError:
                logger.warn(
                    "Invalid input format. Please enter 'y -n' where n is"
                    " the number of continuous tasks."
                )
                return
            return GENERATE_NEXT_COMMAND_JSON

        if console_input.lower() == self.config.exit_key:
            return EXIT

        return Command(user_input=console_input, name="human_feedback")

    def _pre_command(self, command: Command) -> Tuple[Command, str]:
        if command.name is not None and command.name.lower().startswith("error"):
            return command, f"Command {command.name} threw the following error: {command.args}"

        if command.name == "human_feedback":
            return command, f"Human feedback: {command.user_input}"

        for plugin in self.config.plugins:
            if not plugin.can_handle_pre_command():
                continue
            name, args = plugin.pre_command(command.name, command.args)
            command = Command(name=name, args=args, user_input=command.user_input)
        return command, ""

    def _post_command(self, command: Command, result: str) -> str:
        for plugin in self.config.plugins:
            if not plugin.can_handle_post_command():
                continue
            result = plugin.post_command(command.name, result)
        return result

    def _check_output_size(self, command: Command, command_result) -> str | None:
        result_token_length = count_string_tokens(
            str(command_result), self.config.fast_llm_model
        )
        memory_token_length = count_string_tokens(
            str(self.summary_memory), self.config.fast_llm_model
        )
        total_token_length = result_token_length + memory_token_length + 600
        if total_token_length > self.config.fast_token_limit:
            return (
                f'Failure: command "{command.name}" returned too much output. '
                "Do not execute this command again with the same arguments.")


    def _execute_command(self, command: Command) -> str:
        command, result = self._pre_command(command)
        if result:
            return result

        command_result = execute_command(
            self.command_registry,
            command.name,
            command.args,
            self.config.prompt_generator,
        )
        result = f"Command {command.name} returned: " f"{command_result}"
        if err := self._check_output_size(command, command_result, result):
            result = err

        result = self._post_command(command, result)
        if self.next_action_count > 0:
            self.next_action_count -= 1
        return result

    def _append_to_history(self, result: str | None):
        if result:
            self.full_message_history.append(create_chat_message("system", result))
            logger.typewriter_log("SYSTEM: ", Fore.YELLOW, result)
        else:
            self.full_message_history.append(
                create_chat_message("system", "Unable to execute command")
            )
            logger.typewriter_log(
                "SYSTEM: ", Fore.YELLOW, "Unable to execute command"
            )

    def _post_planning(self, assistant_reply: dict):
        for plugin in self.config.plugins:
            if not plugin.can_handle_post_planning():
                continue
            assistant_reply = plugin.post_planning(assistant_reply)
        return assistant_reply

    def _get_command_from_assistant(self, assistant_reply: dict) -> Command:
        command_name, arguments = get_command(assistant_reply)
        arguments = self._resolve_pathlike_command_args(arguments)
        return Command(name=command_name, args=arguments)

    def _get_user_authorization(self, assistant_reply: dict) -> Command:
        logger.info(
            "Enter 'y' to authorise command, 'y -N' to run N continuous commands, 's' to run self-feedback commands"
            "'n' to exit program, or enter feedback for "
            f"{self.ai_name}..."
        )

        command = None
        while command is None:
            command = self._get_user_input(assistant_reply)

        if command is EXIT:
            logger.info("Exiting...")
            exit(0)

        if command is GENERATE_NEXT_COMMAND_JSON:
            logger.typewriter_log(
                "-=-=-=-=-=-=-= COMMAND AUTHORISED BY USER -=-=-=-=-=-=-=",
                Fore.MAGENTA,
                "",
            )

        return command

    def _interaction_loop(self):
        # Send message to AI, get response
        with Spinner("Thinking... "):
            assistant_reply_str = chat_with_ai(
                self,
                self.system_prompt,
                self.triggering_prompt,
                self.full_message_history,
                self.memory,
                self.config.fast_token_limit,
            )  # TODO: This hardcodes the model to use GPT3.5. Make this an argument

        assistant_reply = fix_json_using_multiple_techniques(assistant_reply_str)
        assistant_reply = self._post_planning(assistant_reply)

        # Print Assistant thoughts
        if assistant_reply:
            assistant_reply = validate_json(assistant_reply, LLM_DEFAULT_RESPONSE_FORMAT)
            try:
                print_assistant_thoughts(
                    self.ai_name, assistant_reply, self.config.speak_mode
                )
                command = self._get_command_from_assistant(assistant_reply)
                if command.name and self.config.speak_mode:
                    say_text(f"I want to execute {command.name}")
            except Exception as e:
                logger.error("Error: \n", str(e))

        # Print command
        logger.typewriter_log(
            "NEXT ACTION: ",
            Fore.CYAN,
            f"COMMAND = {Fore.CYAN}{command.name}{Style.RESET_ALL}"
            f"  ARGUMENTS = {Fore.CYAN}{command.args}{Style.RESET_ALL}",
        )

        if not self.config.continuous_mode and self.next_action_count == 0:
            command = self._get_user_authorization(assistant_reply)

        result = self._execute_command(command)
        self._append_to_history(result)

    def start_interaction_loop(self):
        limit = 1
        if self.config.continuous_mode:
            if self.config.continuous_limit > 0:
                limit = self.config.continuous_limit
            else:
                limit = 1 << 31

        for _ in range(limit):
            self._interaction_loop()

        if self.config.continuous_mode:
            logger.typewriter_log(
                "Continuous Limit Reached: ", Fore.YELLOW, f"{self.config.continuous_limit}"
            )
            

    def _resolve_pathlike_command_args(self, command_args):
        if "directory" in command_args and command_args["directory"] in {"", "/"}:
            command_args["directory"] = str(self.workspace.root)
        else:
            for pathlike in ["filename", "directory", "clone_path"]:
                if pathlike in command_args:
                    command_args[pathlike] = str(
                        self.workspace.get_path(command_args[pathlike])
                    )
        return command_args

    def get_self_feedback(self, thoughts: dict, llm_model: str) -> str:
        """Generates a feedback response based on the provided thoughts dictionary.
        This method takes in a dictionary of thoughts containing keys such as 'reasoning',
        'plan', 'thoughts', and 'criticism'. It combines these elements into a single
        feedback message and uses the create_chat_completion() function to generate a
        response based on the input message.
        Args:
            thoughts (dict): A dictionary containing thought elements like reasoning,
            plan, thoughts, and criticism.
        Returns:
            str: A feedback response generated using the provided thoughts dictionary.
        """
        ai_role = self.config.ai_role

        feedback_prompt = f"Below is a message from an AI agent with the role of {ai_role}. Please review the provided Thought, Reasoning, Plan, and Criticism. If these elements accurately contribute to the successful execution of the assumed role, respond with the letter 'Y' followed by a space, and then explain why it is effective. If the provided information is not suitable for achieving the role's objectives, please provide one or more sentences addressing the issue and suggesting a resolution."
        reasoning = thoughts.get("reasoning", "")
        plan = thoughts.get("plan", "")
        thought = thoughts.get("thoughts", "")
        criticism = thoughts.get("criticism", "")
        feedback_thoughts = thought + reasoning + plan + criticism
        return create_chat_completion(
            [{"role": "user", "content": feedback_prompt + feedback_thoughts}],
            llm_model,
        )
