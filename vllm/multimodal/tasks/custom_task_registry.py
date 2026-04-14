# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)


def register_enabled_engine_core_tasks(engine: Any, vllm_config: Any) -> None:
    """Register optional custom task hooks without hard-wiring them into core."""
    if envs.VLLM_ENABLE_HUNYUAN_IMAGE3_TASK:
        from vllm.multimodal.tasks.hunyuan_image3_orchestrator import (
            register_hunyuan_image3_engine_core_hooks,
        )

        register_hunyuan_image3_engine_core_hooks(engine, vllm_config)
        logger.info("Hunyuan Image3 task is enabled")


def register_enabled_model_runner_tasks(model_runner: Any) -> None:
    """Register optional model-runner callables for enabled custom tasks."""
    if envs.VLLM_ENABLE_HUNYUAN_IMAGE3_TASK:
        from vllm.multimodal.tasks.hunyuan_image3_orchestrator import (
            register_hunyuan_image3_modelrunner_task_callable,
        )

        register_hunyuan_image3_modelrunner_task_callable(model_runner)
