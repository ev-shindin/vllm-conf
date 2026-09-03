# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dev endpoint for changing a running engine's prefill/decode role.

Sits beside /collective_rpc rather than reusing it: that route reaches the
workers, and half of a role change belongs to EngineCore. The scheduler caches
its token budget at init and lives in that process, so the budget for the new
role can only be moved from there -- and the low-latency buffer is linear in
that budget, which makes it the largest memory decision in the switch.

    curl -X POST localhost:8000/switch_pd_role \\
         -d '{"backend": "deepep_low_latency", "max_num_batched_tokens": 256}'

Gated behind VLLM_SERVER_DEV_MODE with the other dev routes. It reconfigures a
live engine, so it has no business on a default-exposed surface.
"""

import json
from http import HTTPStatus

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.post("/switch_pd_role")
async def switch_pd_role(raw_request: Request):
    try:
        body = await raw_request.json()
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=f"JSON decode error: {e}",
        ) from e

    backend = body.get("backend")
    if backend is None:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Missing 'backend' in request body",
        )

    try:
        result = await engine_client(raw_request).switch_pd_role(
            backend=backend,
            max_num_tokens=body.get("max_num_tokens"),
            max_num_batched_tokens=body.get("max_num_batched_tokens"),
        )
    except Exception as e:
        # The switch refuses before mutating anything, so a rejection here means
        # the engine is untouched. Report why rather than a bare 500: the
        # reasons are actionable -- an incompatible weight layout, a budget the
        # buffer cannot cover, a buffer that will not fit.
        logger.warning("switch_pd_role refused: %s", e)
        raise HTTPException(
            status_code=HTTPStatus.CONFLICT.value, detail=str(e)
        ) from e

    return JSONResponse(content=result)


def attach_router(app: FastAPI):
    app.include_router(router)
