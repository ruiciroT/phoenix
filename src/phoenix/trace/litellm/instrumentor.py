from trace.exporter import _OpenInferenceExporter
from typing import Any, Dict

from config import get_env_project_name
from litellm.integrations.custom_logger import CustomLogger
from openinference.semconv.resource import ResourceAttributes
from openinference.semconv.trace import SpanAttributes
from opentelemetry import trace as trace_api
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Initialize the TracerProvider once globally
resource = Resource.create({ResourceAttributes.PROJECT_NAME: get_env_project_name()})
tracer_provider = trace_sdk.TracerProvider(resource=resource)
tracer_provider.add_span_processor(BatchSpanProcessor(_OpenInferenceExporter()))
trace_api.set_tracer_provider(tracer_provider)


class LitellmInstrumentor:
    def __init__(self) -> None:
        self.tracer = trace_api.get_tracer(__name__)
        print("Initialized LitellmInstrumentor")

    def log_interaction(
        self,
        input_args: Dict[str, Any],
        response: Any,
        session_name: str = "default",
        parent_span: Any = None,
    ) -> None:
        print("Starting log_interaction")
        context = trace_api.set_span_in_context(parent_span) if parent_span else None
        with self.tracer.start_as_current_span(session_name, context=context) as span:
            print("Span started")
            span.update_name(response.object)
            span.set_attribute(SpanAttributes.INPUT_VALUE, input_args["messages"][0]["content"])
            span.set_attribute(SpanAttributes.LLM_MODEL_NAME, input_args["model"])
            span.set_attribute(
                SpanAttributes.OUTPUT_VALUE, response.choices[0]["message"]["content"]
            )
            span.set_attribute(
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT, response.usage["prompt_tokens"]
            )
            span.set_attribute(
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, response.usage["completion_tokens"]
            )
            span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_TOTAL, response.usage["total_tokens"])
            span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")

            span.set_status(trace_api.StatusCode.OK)
            span.add_event("interaction_details_registered!")
            print(f"Span created: {span}")


class PhoenixLiteLLMHandler(CustomLogger):
    def __init__(self, instrumentor: LitellmInstrumentor) -> None:
        self.instrumentor = instrumentor

    def log_success_event(
        self, kwargs: Dict[str, Any], response_obj: Any, start_time: float, end_time: float
    ) -> None:
        self.instrumentor.log_interaction(kwargs, response_obj)
