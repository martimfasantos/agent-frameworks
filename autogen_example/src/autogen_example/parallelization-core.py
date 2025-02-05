import asyncio
from dataclasses import dataclass
from autogen_core import (
    AgentId,
    ClosureAgent,
    ClosureContext,
    DefaultTopicId,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    TopicId,
    TypeSubscription,
    default_subscription,
    message_handler,
    type_subscription,
)

# AUTOGEN 0.4 - WORKING!

# Issue: https://github.com/microsoft/autogen/issues/5359
# Discussion: https://github.com/microsoft/autogen/discussions/5337

@dataclass
class Task:
    task_id: str


@dataclass
class TaskResponse:
    task_id: str
    result: str


def calculator(a: float, b: float, operator: str) -> str:
    try:
        if operator == '+':
            return str(a + b)
        elif operator == '-':
            return str(a - b)
        elif operator == '*':
            return str(a * b)
        elif operator == '/':
            if b == 0:
                return 'Error: Division by zero'
            return str(a / b)
        else:
            return 'Error: Invalid operator. Please use +, -, *, or /'
    except Exception as e:
        return f'Error: {str(e)}'


TASK_RESULTS_TOPIC_TYPE = "task-results"
task_results_topic_id = TopicId(type=TASK_RESULTS_TOPIC_TYPE, source="default")


@type_subscription(topic_type="urgent")
class UrgentProcessor(RoutedAgent):
    @message_handler
    async def on_task(self, message: Task, ctx: MessageContext) -> None:
        print(f"Urgent processor starting task {message.task_id}")
        await asyncio.sleep(1)  # Simulate work
        print(f"Urgent processor finished task {message.task_id}")

        task_response = TaskResponse(task_id=message.task_id, result="Results by Urgent Processor")
        await self.publish_message(task_response, topic_id=task_results_topic_id)


@type_subscription(topic_type="normal")
class NormalProcessor(RoutedAgent):
    @message_handler
    async def on_task(self, message: Task, ctx: MessageContext) -> None:
        print(f"Normal processor starting task {message.task_id}")
        await asyncio.sleep(3)  # Simulate work
        print(f"Normal processor finished task {message.task_id}")

        task_response = TaskResponse(task_id=message.task_id, result="Results by Normal Processor")
        await self.publish_message(task_response, topic_id=task_results_topic_id)


async def main():
    # -------------------------
    # Run the task in parallel
    # -------------------------
    runtime = SingleThreadedAgentRuntime()

    await UrgentProcessor.register(runtime, "urgent_processor", lambda: UrgentProcessor("Urgent Processor"))
    await NormalProcessor.register(runtime, "normal_processor", lambda: NormalProcessor("Normal Processor"))

    runtime.start()

    await runtime.publish_message(Task(task_id="normal-1"), topic_id=TopicId(type="normal", source="default"))
    await runtime.publish_message(Task(task_id="urgent-1"), topic_id=TopicId(type="urgent", source="default"))

    await runtime.stop_when_idle()

    # -------------------------
    # Collect the results
    # -------------------------
    queue = asyncio.Queue[TaskResponse]()

    async def collect_result(_agent: ClosureContext, message: TaskResponse, ctx: MessageContext) -> None:
        await queue.put(message)

    CLOSURE_AGENT_TYPE = "collect_result_agent"
    await ClosureAgent.register_closure(
        runtime,
        CLOSURE_AGENT_TYPE,
        collect_result,
        subscriptions=lambda: [TypeSubscription(topic_type=TASK_RESULTS_TOPIC_TYPE, agent_type=CLOSURE_AGENT_TYPE)],
    )

    await runtime.publish_message(Task(task_id="normal-1"), topic_id=TopicId(type="normal", source="default"))
    await runtime.publish_message(Task(task_id="urgent-1"), topic_id=TopicId(type="urgent", source="default"))

    # Print the responses
    while not queue.empty():
        print(await queue.get())


if __name__ == "__main__":
    asyncio.run(main())